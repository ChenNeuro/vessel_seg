# Perf Tooling

## 1) Collect benchmark artifact

```bash
python scripts/perf/collect_benchmark.py \
  --name baseline_normal20 \
  --centerline-csv outputs/vmtk_centerlines/compare_summary.csv \
  --output outputs/benchmark/baseline_normal20.json
```

## 2) Compare baseline vs candidate

```bash
python scripts/perf/compare_benchmark.py \
  --baseline outputs/benchmark/baseline_normal20.json \
  --candidate outputs/benchmark/candidate_normal20.json \
  --contract configs/perf_contract.example.json \
  --output outputs/benchmark/compare_normal20.json
```

`compare_benchmark.py` returns non-zero when regression is detected.

## GitHub Actions

Workflow file: `.github/workflows/perf-gate.yml`

1. `Perf Tooling Smoke`: runs on GitHub hosted runner for every PR.
2. `Perf Contract Gate`: runs on `self-hosted` when:
3. Manual trigger (`workflow_dispatch`), or
4. PR has label `run-perf-gate`.

See `docs/github_perf_ci_setup.md` for setup details.
