# Edge Inference and Model Compression

[![CI/CD](https://github.com/avinashjaiengineer/edge-inference-model-compression/actions/workflows/ci.yml/badge.svg)](https://github.com/avinashjaiengineer/edge-inference-model-compression/actions/workflows/ci.yml)

Train a Random Forest on the scikit-learn Breast Cancer dataset, compress it, export it to ONNX, and serve it on CPU with FastAPI and Docker. Then measure, test and monitor it.

> This is a learning demo, not a medical diagnostic tool.

| Stage | What | Where |
|---|---|---|
| 1 | CPU inference, compression, ONNX export and benchmarking | `src/train_model.py`, `src/benchmark.py`, `src/api.py` |
| 2 | Memory profiling and repeatable load tests | `src/profile_memory.py`, `src/load_test.py` |
| 3 | GitHub Actions CI/CD with tests and model-artifact validation | `.github/workflows/ci.yml`, `src/validate_artifacts.py` |
| 4 | Prometheus metrics, Grafana dashboard, latency alerts | `/metrics` in `src/api.py`, `docker-compose.yml`, `monitoring/` |

## Quick start (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python .\src\train_model.py           # writes artifacts/
python -m src.validate_artifacts      # gate: ONNX validity, agreement, accuracy, size
python .\src\benchmark.py             # reports/benchmark_results.csv
python -m src.profile_memory          # reports/memory_results.csv
python -m pytest -q                   # 7 tests
uvicorn src.api:app --host 127.0.0.1 --port 8000
python -m src.load_test --url http://127.0.0.1:8000 --label local
```

Endpoints: `GET /`, `GET /health`, `GET /features`, `POST /predict`, `GET /metrics` (Prometheus). Interactive docs are at `/docs`.

## Stage 1: Compression and ONNX

| Model    | Configuration                                   | Test accuracy |
|----------|--------------------------------------------------|---------------|
| baseline | 300 trees, unrestricted depth                    | 0.9474        |
| compact  | 60 trees, `max_depth=6`, `min_samples_leaf=2`    | 0.9561        |

Measured on the dev laptop (Windows 11, CPU only, scikit-learn 1.7.2, ONNX Runtime 1.31.0):

| Model    | Runtime      | Size (KB) | Accuracy | Agreement vs sklearn | p50 (ms) | p95 (ms) | Batch throughput (samples/s) |
|----------|--------------|-----------|----------|----------------------|----------|----------|------------------------------|
| baseline | scikit-learn | 948.3     | 0.9474   | 1.00                 | 40.78    | 51.49    | 2,315                        |
| baseline | ONNX Runtime | 391.0     | 0.9474   | 1.00                 | 0.027    | 0.034    | 325,608                      |
| compact  | scikit-learn | 161.7     | 0.9561   | 1.00                 | 18.65    | 19.88    | 6,376                        |
| compact  | ONNX Runtime | 63.2      | 0.9561   | 1.00                 | 0.021    | 0.022    | 1,422,635                    |

- **Compression helped on every axis here.** In ONNX form the compact model is about 6x smaller than the baseline (63 KB vs 391 KB). It was slightly *more* accurate on this split, probably because limiting depth reduces overfitting. That won't hold on every dataset.
- **ONNX export preserved behaviour.** Labels match scikit-learn on 100% of test records. Probabilities differ by at most 4e-7.
- **The scikit-learn latency is inflated by `n_jobs=-1`.** That setting dispatches to a thread pool on every `predict()`. For a single record, that overhead dwarfs the tree evaluation.
- **Quantization was not applied.** INT8 quantization targets matrix ops (linear and neural-network models), and tree ensembles don't benefit the same way.

## Stage 2: Memory and load

### Memory (`python -m src.profile_memory`)

Each model/runtime pair is measured in a fresh subprocess that imports only what it needs. Values are RSS in MB on the dev laptop:

| Model    | Runtime      | Runtime import | Model load | Growth over 1,000 single + 100 batch predictions | Peak RSS |
|----------|--------------|----------------|------------|-----------------------------------|----------|
| baseline | scikit-learn | 119.0          | 3.5        | 0.07                              | 142.4    |
| baseline | ONNX Runtime | 32.6           | 8.4        | 0.27                              | 61.1     |
| compact  | scikit-learn | 119.1          | 0.7        | 0.37                              | 139.6    |
| compact  | ONNX Runtime | 32.5           | 7.2        | 0.23                              | 59.8     |

**The runtime matters more than the model.** An ONNX-only serving process peaks at about 60 MB. A process running scikit-learn peaks at about 140 MB, and 119 MB of that is importing scikit-learn/SciPy before any model loads. That gap is the main memory argument for exporting to ONNX on an edge device. Memory stays flat during inference, so there is no growth per request.

### Load (`python -m src.load_test`)

The load test is repeatable by construction:

- a fixed request count and concurrency
- payloads drawn with a fixed seed from the held-out test split
- a warm-up phase that isn't measured
- several runs, with the spread between them reported

Each invocation writes `reports/load_test_<label>.json` and appends a row to `reports/load_test_history.csv`.

| Where the client ran | Requests x runs | Concurrency | Throughput | Client p50 | Client p95 | p95 stdev | Errors |
|---|---|---|---|---|---|---|---|
| Inside the EC2 container (no network) | 2000 x 3 | 16 | 283 req/s | 24.6 ms | 190 ms | 7.9 ms | 0 |
| Dev laptop to the same machine | 2000 x 3 | 16 | 277 req/s | 25 ms | 190 ms | 67 ms | 0 |
| Dev laptop to EC2 over the internet | 1000 x 3 | 32 | 100 req/s | 296 ms | 412 ms | 131 ms | 0 |

Prometheus measured the server-side `/predict` p95 at **8.7 ms**, and the ONNX Runtime inference p95 at **0.67 ms**. Most of the client-side p95 is **queueing**. A single uvicorn worker serves about 280 req/s, so with 16 requests in flight each one waits for the others (Little's law: 16 / 283 ≈ 57 ms on average, with a long tail). Over the internet, the round trip to `us-east-1` dominates.

Running the client on the server host gives the most repeatable numbers (p95 stdev under 8 ms). Use that setup when comparing changes.

## Stage 3: CI/CD (GitHub Actions)

`.github/workflows/ci.yml` runs on every push and pull request.

1. **Train, validate, test**
   - Train both models and export them to ONNX.
   - `python -m src.validate_artifacts` blocks the build unless every check passes:
     - ONNX checker passes
     - float32 input of width 30 with a dynamic batch dimension
     - label and probability outputs, with ZipMap disabled
     - 100% label agreement with scikit-learn
     - probability difference ≤ 1e-4
     - accuracy ≥ 0.93
     - `compact.onnx` ≤ 200 KB and smaller than the baseline
   - Run pytest.
   - Run the benchmark and memory profile, and post both tables to the job summary.
   - Upload the models and reports as build artifacts.
2. **Monitoring config:** `promtool check config` and `promtool test rules` (alert unit tests), plus a JSON check of the dashboard.
3. **Container**
   - Build the image, start it, and wait for the Docker `HEALTHCHECK` to report `healthy`.
   - Run a 500-request smoke load test that must have zero errors.
   - Check `/metrics`.
   - On pushes to `main`, publish `ghcr.io/avinashjaiengineer/edge-inference:{sha,latest}`.

Model artifacts are not committed. CI rebuilds them from code, and training is deterministic (`random_state=42`).

## Stage 4: Monitoring

```bash
echo "GRAFANA_ADMIN_PASSWORD=<choose one>" > .env    # git-ignored
docker compose up -d
```

| Service    | URL                    | Notes |
|------------|------------------------|-------|
| API        | http://HOST:8000       | `/metrics` exposes the metrics below |
| Prometheus | http://HOST:9090       | Scrapes every 5 s. Rules are in `monitoring/prometheus/alerts.yml`. |
| Grafana    | http://HOST:3000       | User `admin`, password from `.env`. The data source and the *Edge Inference API* dashboard are provisioned automatically. |

**Metrics**

- `http_requests_total{method,path,status}`
- `http_request_duration_seconds{method,path}`, a histogram measured by middleware
- `model_inference_duration_seconds`, which times `session.run` only
- `model_predictions_total{prediction}`
- `model_info`
- the standard `process_*` CPU and memory metrics

Paths are recorded as route templates, so arbitrary URLs can't explode label cardinality.

**Alerts** (each one is covered by `monitoring/prometheus/alerts_test.yml`):

| Alert | Condition | For |
|---|---|---|
| `PredictLatencyP95High` | server-side `/predict` p95 > 50 ms | 2m |
| `InferenceLatencyP95High` | ONNX Runtime p95 > 5 ms | 2m |
| `PredictErrorRateHigh` | `/predict` 5xx rate > 1% | 5m |
| `EdgeInferenceDown` | scrape target down | 1m |

Alerts are evaluated by Prometheus and appear in the Prometheus UI and on the dashboard's alert timeline (built from Prometheus's `ALERTS` series). Alertmanager isn't included. To send notifications (Slack, email, PagerDuty), add an Alertmanager service and point `alerting:` in `prometheus.yml` at it.

## Deployment: AWS EC2 (torn down)

The full compose stack (API, Prometheus, Grafana) ran on one Amazon Linux 2023 `t3.small` in `us-east-1` on 2026-10-11. All the measurements in this README were taken there. Every resource was deleted when the project ended: the instance, its 20 GB disk, the security group and the key pair. Nothing is left running or billing.

To redeploy from scratch:

1. Launch an Amazon Linux 2023 instance (`t3.small` or larger) with `deploy/ec2-user-data.sh` as user data, which installs Docker.
2. Install the Compose plugin under `/usr/local/lib/docker/cli-plugins`. It was v2.29.7 here.
3. Allow ports 22, 8000, 9090 and 3000 from your own IP only.

Then, from Git Bash in the project root, after training and creating `.env`:

```bash
H=ec2-user@<instance-ip>
scp -r src artifacts requirements.txt Dockerfile .dockerignore docker-compose.yml monitoring .env $H:~/app/
ssh $H "cd ~/app && sudo docker build -t edge-inference:1.0 . && sudo docker compose up -d --no-build"
```

Set `MSYS_NO_PATHCONV=1` for any `aws` command in Git Bash, so it doesn't rewrite AWS paths such as `/aws/service/...`.

## Conclusion

| Question | Answer from the measurements |
|---|---|
| Did compression cost accuracy? | No. 60 trees at depth 6 scored 0.9561 vs 0.9474 for 300 unrestricted trees, and the model is 6x smaller (63 KB vs 391 KB ONNX). |
| Did ONNX conversion change predictions? | No. Labels agree 100%, probabilities differ by at most 4e-7, and CI blocks any build where that stops being true. |
| Where does the memory go? | Mostly into the runtime. An ONNX-only process peaks at ~60 MB, against ~140 MB with scikit-learn loaded. |
| Where does the latency go? | Mostly into queueing. Inference p95 is ~0.2 ms and server p95 is 9 ms, but client p95 is 190 ms at concurrency 16 on one worker (~285 req/s). |
| Is it observable? | Yes. Request and inference latency, errors, predictions by class, CPU and memory are all on one Grafana dashboard. The `EdgeInferenceDown` alert went pending → firing → resolved in a live outage test. |

Next steps, if the project continues:

- Run one uvicorn worker per vCPU to cut queueing.
- Add Alertmanager so alerts notify someone.
- Add a promtool test for `InferenceLatencyP95High`.
- Alert on a shift in the predicted-class ratio as a cheap drift signal.

## Environment notes

- `scikit-learn` is pinned to **1.7.2**. On the dev machine, Windows Smart App Control blocks a compiled DLL (`_argkmin`) shipped in scikit-learn 1.9.1, which stops `sklearn.ensemble` from importing. The same pin is used in CI and Docker so that training and serving match.
- Only load `.joblib` files you created or trust.
