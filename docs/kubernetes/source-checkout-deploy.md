---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Source Checkout Deployment
---

# Deploy from a Source Checkout to a Real Kubernetes Cluster

This guide starts from an AIPerf source checkout and deploys that code to a real Kubernetes cluster. It builds a container image, pushes it to a registry, installs or upgrades the AIPerf operator with Helm, then runs a benchmark against a real inference endpoint.

This guide does **not** use the mock server path. Use it when you have a real OpenAI-compatible endpoint already running, or when you want to deploy a real Dynamo/vLLM endpoint first.

## Prerequisites

You need:

- An AIPerf source checkout.
- `kubectl` configured for the target cluster.
- Helm v3.
- Docker or another container builder that can build and push OCI images.
- Registry credentials for the image repository you will use.
- Permission to install CRDs, create the operator namespace, create benchmark namespaces, and create JobSet workloads.
- JobSet installed on the cluster. If it is not installed, install it before the AIPerf operator.
- A real OpenAI-compatible inference endpoint reachable from benchmark pods, or permission to deploy one.

Check the cluster first:

```bash
kubectl cluster-info
kubectl get nodes
kubectl api-resources | grep -i jobset
```

For GPU benchmarks, verify GPU resources are allocatable:

```bash
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu
```

## 1. Set up the checkout

Install the local development environment from the checkout:

```bash
make first-time-setup
```

Use the local CLI through `uv run` until the package is installed somewhere else:

```bash
uv run aiperf --help
uv run aiperf kube --help
```

The build script derives the image tag from the checkout version, branch, and
commit automatically. You can preview it without building:

```bash
source ci/scripts/lib/common.sh
aiperf_ci_init
aiperf_ci_resolve_refs
printf '%s\n' "${AIPERF_IMAGE_REF}"
```

## 2. Build and push the AIPerf image

The repository provides a Harbor-oriented BuildKit entrypoint. It uses the same image for the operator containers and benchmark JobSet pods unless you explicitly override `defaults.image` in the Helm chart.

The generated image tag is deterministic:

```text
{version}-{prefix}-{commit sha first 4 characters}
```

For example, version `v0.13.0` on branch `develop` at commit `1ec5f974...` becomes:

```text
v0.13.0-dev-1ec5
```

The default prefix is the branch name, with `develop` normalized to `dev`. Override it with `IMAGE_TAG_PREFIX`.

Run the scripts using Bash (including when previewing a tag). Log in to Harbor
and build the complete runtime image:

```bash
docker login harbor.weizhipin.com
PUSH_IMAGE=1 make docker-ci
```

The script defaults to these internal sources:

- Base Python image: `harbor.weizhipin.com/arsenal-ai/python:3.12-slim`
- uv image: `harbor.weizhipin.com/arsenal-ai/uv:latest`
- Python packages: Tsinghua PyPI mirror
- CPU PyTorch wheels: SJTU mirror
- Debian packages: Aliyun Debian mirror
- BuildKit image: `harbor.weizhipin.com/arsenal-ai/moby/buildkit:buildx-stable-1`
- Image repository: `harbor.weizhipin.com/arsenal-ai/aiperf`

Runtime requirements are exported from `uv.lock` with their exact pins and
hashes, then installed from these mirrors without re-resolving the lockfile.
Changing `UV_DEFAULT_INDEX` alongside `uv sync --locked` would invalidate the
lock's original index identity. `PYPI_INDEX_URL` and `PYTORCH_INDEX_URL` override
the two trusted indexes; installation checks lockfile hashes even when
searching both indexes.

FFmpeg source archives still default to `https://ffmpeg.org/releases`, because
the tested domestic paths did not serve valid archives. Override
`FFMPEG_SOURCE_URL` with a reachable release-directory mirror if available.

Useful overrides:

```bash
# Build without pushing the image; still use the registry cache when available.
make docker-ci

# Build the shorter operator/CI image without compiling ffmpeg.
ENABLE_FFMPEG=0 make docker-ci

# Use a custom branch prefix.
IMAGE_TAG_PREFIX=feature-dashboard PUSH_IMAGE=1 make docker-ci

# Force a clean build.
NO_CACHE=1 PUSH_IMAGE=1 make docker-ci
```

The build uses BuildKit registry cache at:

```text
harbor.weizhipin.com/arsenal-ai/aiperf-build-cache
```

It also uses cache mounts for the uv package cache and ffmpeg C/C++ compilation
cache. These cache mounts persist on the same BuildKit builder, but are not
exported as registry layer cache. Use a persistent builder (for example, on a
self-hosted runner) to retain them between CI runs. `PUSH_IMAGE=1` publishes both
the image and the branch/platform layer cache. Set `PUSH_CACHE=0` to disable
cache export independently. `NO_CACHE=1` ignores layer cache, but does not erase
the builder's uv/ccache mounts and still permits exporting newly built layers.

Reuse a builder with `BUILDX_BUILDER`; its existing BuildKit configuration is
preserved. `BUILDKIT_IMAGE` controls newly created builders. The script selects
the builder without changing Docker's global current-builder setting.

For a cluster with a different architecture, set the target platform explicitly:

```bash
BUILD_PLATFORM=linux/amd64 PUSH_IMAGE=1 make docker-ci
```

The implementation is in `ci/scripts/build-image.sh`, with tag and BuildKit
helpers under `ci/scripts/lib/`. GitHub Actions uses the same script in
`.github/workflows/build-image.yml` and is manually triggered. Configure the
`HARBOR_USERNAME` and `HARBOR_PASSWORD` repository secrets, and ensure the runner
can reach Harbor. Set the `AIPERF_BUILD_RUNNER` repository variable to an internal
self-hosted runner label if hosted runners cannot reach it. The default runner
is `ubuntu-latest`.

For a source copy without `.git`, pass `AIPERF_BRANCH` and the original
`AIPERF_SHA`; CI automatically uses `GITHUB_SHA` when provided. Builds from an
uncommitted working tree still use HEAD's tag, so commit before publishing an
image intended to identify an exact source revision. A short four-character
SHA may collide, so the tag is a naming convention, not a cryptographic identity.

## 3. Create an image pull secret if the registry is private

Skip this section if every node can pull the image without a secret.

Create the operator namespace first:

```bash
kubectl create namespace aiperf-system --dry-run=client -o yaml | kubectl apply -f -
```

For Harbor:

```bash
kubectl create secret docker-registry aiperf-registry \
  --namespace aiperf-system \
  --docker-server=harbor.weizhipin.com \
  --docker-username="${HARBOR_USERNAME}" \
  --docker-password="${HARBOR_PASSWORD}"
```

For GHCR:

```bash
kubectl create secret docker-registry aiperf-registry \
  --namespace aiperf-system \
  --docker-server=ghcr.io \
  --docker-username="${GITHUB_USER}" \
  --docker-password="${GITHUB_TOKEN}"
```

The Helm install below references this secret for the operator. Benchmark jobs also need pull access in the namespace you run them in. Kubernetes secrets are namespace-scoped, so create the same pull secret there:

```bash
kubectl create namespace <your-namespace> --dry-run=client -o yaml | kubectl apply -f -

kubectl create secret docker-registry aiperf-registry \
  --namespace <your-namespace> \
  --docker-server=<registry-host> \
  --docker-username=<username> \
  --docker-password=<token>
```

## 4. Install or upgrade the AIPerf operator

Resolve the built image reference and split it into repository and tag for Helm
(run this block in Bash):

```bash
source ci/scripts/lib/common.sh
aiperf_ci_init
aiperf_ci_resolve_refs
export AIPERF_IMAGE="${AIPERF_IMAGE_REF}"
export AIPERF_IMAGE_REPOSITORY="${AIPERF_IMAGE%:*}"
export AIPERF_IMAGE_TAG="${AIPERF_IMAGE##*:}"
```

Install the operator:

```bash
helm upgrade --install aiperf-operator deploy/helm/aiperf-operator \
  --namespace aiperf-system \
  --create-namespace \
  --set image.repository="${AIPERF_IMAGE_REPOSITORY}" \
  --set image.tag="${AIPERF_IMAGE_TAG}" \
  --set image.pullPolicy=IfNotPresent
```

If you created `aiperf-registry`, include it in the Helm release:

```bash
helm upgrade --install aiperf-operator deploy/helm/aiperf-operator \
  --namespace aiperf-system \
  --create-namespace \
  --set image.repository="${AIPERF_IMAGE_REPOSITORY}" \
  --set image.tag="${AIPERF_IMAGE_TAG}" \
  --set image.pullPolicy=IfNotPresent \
  --set 'imagePullSecrets[0].name=aiperf-registry'
```

The chart default for `defaults.image` is empty, which means AIPerfJob pods use `<image.repository>:<image.tag>`, falling back to the chart's `appVersion` when `image.tag` is also empty. Set `defaults.image` only when benchmark pods should run a different image from the operator.

Wait for the operator and results server:

```bash
kubectl rollout status deploy/aiperf-operator -n aiperf-system --timeout=180s
kubectl get pods -n aiperf-system
```

Run Helm tests if the cluster can pull the chart's test image:

```bash
helm test aiperf-operator -n aiperf-system
```

## 5. Run against an existing real endpoint

Use this path when your inference server is already deployed in the cluster or reachable from the cluster network.

Set the endpoint and model:

```bash
export MODEL="Qwen/Qwen3-0.6B"
export ENDPOINT_URL="http://vllm.default.svc.cluster.local:8000/v1"
```

Run cluster-side preflight checks:

```bash
uv run aiperf kube preflight \
  --image "${AIPERF_IMAGE}" \
  --endpoint-url "${ENDPOINT_URL}" \
  --workers 8
```

Submit a benchmark:

```bash
uv run aiperf kube profile \
  --model "${MODEL}" \
  --url "${ENDPOINT_URL}" \
  --image "${AIPERF_IMAGE}" \
  --total-workers 8 \
  --request-count 1000 \
  --concurrency 50 \
  --streaming
```

For private benchmark images, pass the pull secret name when submitting jobs:

```bash
uv run aiperf kube profile \
  --model "${MODEL}" \
  --url "${ENDPOINT_URL}" \
  --image "${AIPERF_IMAGE}" \
  --image-pull-secrets aiperf-registry \
  --total-workers 8 \
  --request-count 1000 \
  --concurrency 50 \
  --streaming
```

For repeatable runs, put the benchmark configuration in YAML and pass `--config benchmark.yaml`; see [End-to-End Workflow](workflow.md) for the `init` -> `validate` -> `preflight` -> `profile` sequence.

## 6. Optional: deploy a real Dynamo/vLLM endpoint first

Skip this section if you already have a real endpoint.

Install the Dynamo platform chart if your cluster does not already have it:

```bash
helm upgrade --install dynamo-platform \
  oci://nvcr.io/nvidia/ai-dynamo/dynamo-platform \
  --version 1.1.0 \
  --namespace dynamo-system \
  --create-namespace \
  --set dynamo-operator.webhook.enabled=false \
  --set grove.enabled=false \
  --set kai-scheduler.enabled=false
```

Deploy an aggregated Dynamo vLLM server by applying a `DynamoGraphDeployment` manifest such as the aggregated vLLM example in [Getting Started on Kubernetes](getting-started.md#step-2-deploy-a-dynamo-inference-server):

```bash
kubectl create namespace dynamo-server --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f dynamo-server.yaml
```

Wait for the frontend and worker pods to become ready:

```bash
kubectl get pods -n dynamo-server -w
```

Use the Dynamo service URL as the benchmark endpoint:

```bash
export ENDPOINT_URL="http://dynamo-agg-frontend.dynamo-server.svc:8000/v1"

uv run aiperf kube preflight \
  --image "${AIPERF_IMAGE}" \
  --endpoint-url "${ENDPOINT_URL}" \
  --workers 8

uv run aiperf kube profile \
  --model "${MODEL}" \
  --url "${ENDPOINT_URL}" \
  --image "${AIPERF_IMAGE}" \
  --total-workers 8 \
  --request-count 1000 \
  --concurrency 50 \
  --streaming
```

## 7. Monitor and retrieve results

Watch progress:

```bash
uv run aiperf kube list
```

Reattach to a detached run:

```bash
uv run aiperf kube attach
```

Download results:

```bash
uv run aiperf kube results --output ./aiperf-results
```

Port-forward the results server and dashboard:

```bash
kubectl port-forward -n aiperf-system svc/aiperf-operator 8081:8081
```

Then open `http://localhost:8081`.

## 8. Upgrade after source changes

After changing source code, repeat the build and push with a new immutable tag:

```bash
PUSH_IMAGE=1 make docker-ci
source ci/scripts/lib/common.sh
aiperf_ci_init
aiperf_ci_resolve_refs
export AIPERF_IMAGE="${AIPERF_IMAGE_REF}"

export AIPERF_IMAGE_REPOSITORY="${AIPERF_IMAGE%:*}"
export AIPERF_IMAGE_TAG="${AIPERF_IMAGE##*:}"

helm upgrade aiperf-operator deploy/helm/aiperf-operator \
  --namespace aiperf-system \
  --reuse-values \
  --set image.repository="${AIPERF_IMAGE_REPOSITORY}" \
  --set image.tag="${AIPERF_IMAGE_TAG}"
```

Wait for the new operator pod before submitting new jobs:

```bash
kubectl rollout status deploy/aiperf-operator -n aiperf-system --timeout=180s
```

## Troubleshooting

### ImagePullBackOff

Check the failing pod and events:

```bash
kubectl describe pod -n aiperf-system -l app.kubernetes.io/name=aiperf-operator
uv run aiperf kube debug
```

Common fixes:

- Confirm the image was pushed with the exact tag used by Helm or `aiperf kube profile`.
- Create the pull secret in both `aiperf-system` and the benchmark namespace.
- Pass `--image-pull-secrets aiperf-registry` to `aiperf kube profile` for private benchmark images.
- Use `--set image.pullPolicy=Always` while testing mutable tags; prefer immutable tags for normal use.

### Operator is running but jobs use an old image

The operator image and default benchmark image come from the Helm release. Check the rendered CRD default:

```bash
kubectl get crd aiperfjobs.aiperf.nvidia.com \
  -o jsonpath='{.spec.versions[?(@.name=="v1alpha1")].schema.openAPIV3Schema.properties.spec.properties.image.default}{"\n"}'
```

If you set `defaults.image`, it overrides the benchmark image independently of `image.repository` and `image.tag`.

### Endpoint check fails

Verify the endpoint from inside the cluster:

```bash
kubectl run curl-check \
  --rm -it \
  --restart=Never \
  --image=curlimages/curl:latest \
  -- curl -sf "${ENDPOINT_URL}/models"
```

If the server is still starting and you intentionally want the benchmark to wait until worker runtime, set `skipEndpointCheck: true` in the AIPerfJob YAML. Do not use this to hide a wrong service name or namespace.

### RBAC or namespace errors

The Helm chart creates benchmark RBAC in the release namespace and in any namespaces listed in `benchmarkRbacNamespaces`. If you run jobs in a different namespace, add that namespace to `benchmarkRbacNamespaces`:

```bash
helm upgrade aiperf-operator deploy/helm/aiperf-operator \
  --namespace aiperf-system \
  --reuse-values \
  --set 'benchmarkRbacNamespaces[0]=team-a-benchmarks'
```

## Related Documentation

- [Getting Started on Kubernetes](getting-started.md) -- First benchmark walkthrough and Dynamo manifest examples.
- [End-to-End Workflow](workflow.md) -- Full `init` -> `validate` -> `preflight` -> `profile` -> `results` lifecycle.
- [Production Deployments](production.md) -- CI/CD, Kueue, private registries, multi-tenancy, and operations patterns.
- [Kubernetes Configuration Reference](configuration.md) -- CRD fields, Helm values, and AIPerfJob configuration.
- [Monitoring and Troubleshooting](monitoring.md) -- Watch, debug, logs, and common failure modes.
