"""Runs the five demo scenarios against whatever infrastructure is reachable and stores raw tool evidence in docs/evidence/."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.orchestrator import Orchestrator  # noqa: E402

out = Path("docs/evidence")
out.mkdir(parents=True, exist_ok=True)
SCENARIOS = {
    "1_dataset_provenance": "Which dataset version was used for experiment 1?",
    "2_model_selection": "Compare the last five training runs and recommend the best candidate based on F1 score",
    "3_pipeline_execution": "Trigger the model training pipeline and report whether execution succeeded.",
    "4_deployment_diagnosis": "Why is my prediction API unavailable?",
    "5_multi_tool": "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary.",
}
o = Orchestrator()
for name, q in SCENARIOS.items():
    r = o.handle(q)
    (out / f"{name}.json").write_text(json.dumps({"question": q, **r.model_dump()}, indent=2, default=str))
    print(f"== {name}: {q}\n{r.answer}\n")
