"""DVC data-management tools. Only an allow-list of DVC/git sub-commands can ever run (no shell)."""
from __future__ import annotations

import json
import re
import subprocess
import sys

import yaml

from agent.config import get_settings
from agent.tools.base import PermissionDenied, ToolError, tool

ALLOWED = {"dvc": {"status", "data", "diff", "list", "checkout", "remote"},
           "git": {"rev-parse", "log", "tag", "describe", "cat-file"}}
REV_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/~^@-]{0,99}$")


def _run(binary: str, args: list[str], timeout: int = 60) -> str:
    if binary not in ALLOWED or not args or args[0] not in ALLOWED[binary]:
        raise PermissionDenied(f"command '{binary} {args[:1]}' is not on the approved list")
    if binary == "dvc" and args[0] == "remote" and args[1:2] != ["list"]:
        raise PermissionDenied("only 'dvc remote list' is permitted")
    cmd = [sys.executable, "-m", "dvc", *args] if binary == "dvc" else ["git", *args]
    try:
        p = subprocess.run(cmd, cwd=get_settings().repo_root, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as e:
        raise ToolError(f"{binary} is not installed") from e
    except subprocess.TimeoutExpired as e:
        raise ToolError(f"{binary} {args[0]} timed out") from e
    if p.returncode != 0:
        raise ToolError(f"{binary} {args[0]} failed: {(p.stderr or p.stdout).strip()[:400]}")
    return p.stdout


def _rev(rev: str) -> str:
    if not REV_RE.match(rev):
        raise PermissionDenied(f"invalid revision string '{rev}'")
    return rev


def _dvc_files() -> list:
    root = get_settings().repo_root
    skip = {".git", ".dvc", ".venv", "mlruns", "node_modules", "artifacts_work"}
    return [p for p in root.rglob("*.dvc") if not skip & set(p.relative_to(root).parts)]


@tool("dvc_list_datasets", "List datasets tracked by DVC (path, md5, size).", "dvc")
def list_datasets() -> dict:
    root, out = get_settings().repo_root, []
    for f in _dvc_files():
        for o in yaml.safe_load(f.read_text()).get("outs", []):
            out.append({"dvc_file": str(f.relative_to(root)), "path": str((f.parent / o["path"]).relative_to(root)),
                        "md5": o.get("md5"), "size": o.get("size")})
    return {"datasets": out, "remotes": _remotes()}


def _remotes() -> list[str]:
    try:
        return [x for x in _run("dvc", ["remote", "list"]).splitlines() if x]
    except ToolError:
        return []


@tool("dvc_get_dataset_version", "Current recorded dataset version (md5), Git commit and version tag.", "dvc",
      {"path": {"type": "string"}})
def get_dataset_version(path: str | None = None) -> dict:
    from training.pipeline import dataset_provenance
    path = path or get_settings().dvc_dataset
    prov = dataset_provenance(path)
    if prov["dvc.dataset_md5"] == "unknown":
        raise ToolError(f"{path} is not tracked by DVC (no {path}.dvc file)")
    head = _run("git", ["rev-parse", "--short", "HEAD"]).strip()
    return {"path": path, "md5": prov["dvc.dataset_md5"], "dataset_version": prov["dvc.dataset_version"],
            "dataset_commit": prov["dvc.git_commit"], "git_head": head,
            "workspace_matches_recorded": prov["dvc.data_matches_recorded"] == "true"}


@tool("dvc_check_changes", "Detect whether DVC-tracked data differs from what is recorded in Git.", "dvc")
def check_changes() -> dict:
    raw = _run("dvc", ["data", "status", "--json"])
    status = json.loads(raw or "{}")
    unc = status.get("uncommitted", {}) or {}
    modified = [*unc.get("modified", []), *unc.get("deleted", []), *unc.get("added", [])]
    missing = status.get("not_in_cache", []) or []
    return {"changed": bool(modified), "modified_or_deleted": modified, "not_in_cache": missing,
            "raw_status": status}


@tool("dvc_compare_versions", "Compare tracked data between two Git revisions (tags/commits).", "dvc",
      {"rev_a": {"type": "string"}, "rev_b": {"type": "string"}}, ("rev_a", "rev_b"))
def compare_versions(rev_a: str, rev_b: str) -> dict:
    raw = _run("dvc", ["diff", _rev(rev_a), _rev(rev_b), "--json"])
    return {"rev_a": rev_a, "rev_b": rev_b, "diff": json.loads(raw or "{}")}


@tool("dvc_explain_reproduce", "Explain (without executing) how to reproduce a previous dataset version.", "dvc",
      {"rev": {"type": "string"}, "path": {"type": "string"}}, ("rev",))
def explain_reproduce(rev: str, path: str | None = None) -> dict:
    path = path or get_settings().dvc_dataset
    full = _run("git", ["rev-parse", "--verify", f"{_rev(rev)}^{{commit}}"]).strip()
    steps = [f"git checkout {rev} -- {path}.dvc   # restore the pointer file recorded at that revision",
             f"dvc checkout {path}                 # restore data from local cache",
             f"dvc pull {path}                     # only if the cache lacks it: fetch from the DVC remote",
             f"git checkout HEAD -- {path}.dvc && dvc checkout {path}   # return to current version"]
    return {"rev": rev, "commit": full, "steps": steps}


@tool("dvc_checkout", "Restore the DVC-tracked dataset to the version recorded at a Git revision. CHANGES THE WORKSPACE.",
      "dvc", {"rev": {"type": "string"}, "path": {"type": "string"}, "confirm": {"type": "boolean"}},
      ("rev",), mutating=True)
def checkout(rev: str, path: str | None = None, confirm: bool = False) -> dict:
    path = path or get_settings().dvc_dataset
    _rev(rev)
    if ".." in path or path.startswith("/"):
        raise PermissionDenied("path must be relative to the repository")
    _run("git", ["rev-parse", "--verify", f"{rev}^{{commit}}"])
    # git restore of the .dvc pointer is performed with a fixed, validated command (not user-controlled flags)
    p = subprocess.run(["git", "checkout", rev, "--", f"{path}.dvc"], cwd=get_settings().repo_root,
                       capture_output=True, text=True, timeout=30)
    if p.returncode != 0:
        raise ToolError(f"git checkout failed: {p.stderr.strip()[:300]}")
    out = _run("dvc", ["checkout", path])
    return {"rev": rev, "path": path, "result": out.strip() or "checked out"}


@tool("dvc_run_provenance", "Dataset version recorded for an MLflow run or experiment id ('experiment 25').",
      "dvc", {"run_ref": {"type": "string"}}, ("run_ref",))
def run_provenance(run_ref: str) -> dict:
    from agent.tools.mlflow_tools import _call, get_client
    c = get_client()
    run = None
    try:
        run = c.get_run(run_ref)
    except Exception:  # noqa: BLE001
        pass
    if run is None:
        runs = _call(c.search_runs, [run_ref] if run_ref.isdigit() else [e.experiment_id for e in _call(c.search_experiments)],
                     filter_string="" if run_ref.isdigit() else f"tags.mlflow.runName = '{run_ref.replace(chr(39), '')}'",
                     order_by=["attributes.start_time DESC"], max_results=1)
        run = runs[0] if runs else None
    if run is None:
        raise ToolError(f"no run or experiment matching '{run_ref}'")
    t = run.data.tags
    if "dvc.dataset_md5" not in t:
        raise ToolError(f"run {run.info.run_id} has no recorded DVC provenance tags")
    return {"run_id": run.info.run_id, "dataset_path": t.get("dvc.dataset_path"), "dataset_md5": t["dvc.dataset_md5"],
            "dataset_version": t.get("dvc.dataset_version"), "git_commit": t.get("dvc.git_commit"),
            "data_matched_recorded_hash": t.get("dvc.data_matches_recorded")}
