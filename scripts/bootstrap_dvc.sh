#!/usr/bin/env bash
# Creates two dataset versions (data-v1, data-v2) with a LOCAL DVC remote and one MLflow run per version.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
[ -d .git ] || git init -q -b main
git config user.email >/dev/null 2>&1 || git config user.email "student@example.com"
git config user.name  >/dev/null 2>&1 || git config user.name  "student"
[ -d .dvc ] || $PY -m dvc init -q
$PY -m dvc remote add -f -d localstore .dvc-storage >/dev/null
rm -f data/customer_churn.csv
git add -A . >/dev/null; git commit -qm "chore: project scaffold" || true

for V in v1 v2; do
  $PY scripts/make_dataset.py --version $V
  $PY -m dvc add data/customer_churn.csv -q
  git add data/customer_churn.csv.dvc data/.gitignore .dvc/config
  git commit -qm "data: dataset $V" || true
  git tag -f data-$V >/dev/null
  $PY -m dvc push -q
  $PY -m training.train
done
git add -A . >/dev/null; git commit -qm "chore: dvc.yaml and tracked state" || true
echo "bootstrap complete"; git tag --list 'data-v*'
