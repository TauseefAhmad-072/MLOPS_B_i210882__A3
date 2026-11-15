#!/usr/bin/env bash
# Build the inference image from a freshly exported model and deploy to a local cluster (kind or minikube).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m training.train --export-only
docker build -t mlops-inference:1.0.0 .
if command -v kind >/dev/null && kind get clusters 2>/dev/null | grep -q .; then kind load docker-image mlops-inference:1.0.0
elif command -v minikube >/dev/null; then minikube image load mlops-inference:1.0.0
else echo "No kind/minikube detected - Docker Desktop Kubernetes shares the local image store."; fi
kubectl apply -f kubernetes/namespace.yaml -f kubernetes/configmap.yaml -f kubernetes/deployment.yaml -f kubernetes/service.yaml
kubectl -n mlops-agent rollout status deploy/ml-inference-api --timeout=120s
kubectl -n mlops-agent get pods,svc
