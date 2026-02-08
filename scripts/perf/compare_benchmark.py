#!/usr/bin/env python3
"""Compare two benchmark artifacts against a configurable performance contract."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_CONTRACT: Dict[str, Any] = {
    "aggregate_metric_rules": [
        {"name": "pred2gt_mean", "direction": "lower", "rel_tol": 0.0, "abs_tol": 0.0, "required": True},
        {"name": "pred2gt_p95", "direction": "lower", "rel_tol": 0.0, "abs_tol": 0.0, "required": True},
        {"name": "gt2pred_mean", "direction": "lower", "rel_tol": 0.0, "abs_tol": 0.0, "required": True},
        {"name": "coverage_pred@1mm", "direction": "higher", "rel_tol": 0.0, "abs_tol": 0.0, "required": True},
        {"name": "coverage_gt@1mm", "direction": "higher", "rel_tol": 0.0, "abs_tol": 0.0, "required": True},
    ],
    "runtime_rules": [
        {"name": "wall_time_sec", "direction": "lower", "rel_tol": 0.0, "abs_tol": 0.0, "required": False},
        {"name": "peak_rss_mb", "direction": "lower", "rel_tol": 0.0, "abs_tol": 0.0, "required": False},
    ],
    "per_case_rules": [],
    "require_same_case_set": True,
}


def _parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(num) or math.isinf(num):
        return None
    return num


def _allowed_delta(base: float, rel_tol: float, abs_tol: float) -> float:
    return abs_tol + abs(base) * rel_tol


def _check_rule(
    *,
    scope: str,
    name: str,
    direction: str,
    baseline_value: Optional[float],
    candidate_value: Optional[float],
    rel_tol: float,
    abs_tol: float,
    required: bool,
) -> Dict[str, Any]:
    check: Dict[str, Any] = {
        "scope": scope,
        "name": name,
        "direction": direction,
        "baseline": baseline_value,
        "candidate": candidate_value,
        "rel_tol": rel_tol,
        "abs_tol": abs_tol,
        "required": required,
        "passed": True,
        "reason": "ok",
    }

    if baseline_value is None or candidate_value is None:
        if required:
            check["passed"] = False
            check["reason"] = "missing_value"
        else:
            check["reason"] = "skipped_missing_value"
        return check

    allowed = _allowed_delta(baseline_value, rel_tol=rel_tol, abs_tol=abs_tol)
    check["allowed_delta"] = allowed
    check["delta"] = candidate_value - baseline_value

    if direction == "lower":
        passed = candidate_value <= baseline_value + allowed
    elif direction == "higher":
        passed = candidate_value >= baseline_value - allowed
    else:
        check["passed"] = False
        check["reason"] = f"invalid_direction:{direction}"
        return check

    check["passed"] = bool(passed)
    if not passed:
        check["reason"] = "regression"
    return check


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _get_aggregate_metric(artifact: Dict[str, Any], metric_name: str) -> Optional[float]:
    centerline = artifact.get("centerline_summary") or {}
    aggregates = centerline.get("aggregates") or {}
    metric = aggregates.get(metric_name) or {}
    return _parse_float(metric.get("mean"))


def _get_runtime_metric(artifact: Dict[str, Any], metric_name: str) -> Optional[float]:
    runtime = artifact.get("runtime_summary") or {}
    metric = runtime.get(metric_name) or {}
    return _parse_float(metric.get("mean"))


def _case_map(artifact: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    centerline = artifact.get("centerline_summary") or {}
    per_case = centerline.get("per_case") or []
    result: Dict[str, Dict[str, Any]] = {}
    for row in per_case:
        case_name = row.get("case")
        if case_name:
            result[str(case_name)] = row
    return result


def _merge_contract(user_contract: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(DEFAULT_CONTRACT)
    for key, value in user_contract.items():
        merged[key] = value
    return merged


def _compare_case_sets(
    baseline_cases: Dict[str, Dict[str, Any]],
    candidate_cases: Dict[str, Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    base_set = set(baseline_cases.keys())
    cand_set = set(candidate_cases.keys())
    missing_in_candidate = sorted(base_set - cand_set)
    extra_in_candidate = sorted(cand_set - base_set)
    return missing_in_candidate, extra_in_candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare baseline vs candidate benchmark artifacts.")
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline benchmark JSON.")
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate benchmark JSON.")
    parser.add_argument(
        "--contract",
        type=Path,
        default=None,
        help="Optional contract JSON. If omitted, strict defaults are used.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional path to write compare report JSON.")
    parser.add_argument(
        "--no-fail-on-regression",
        action="store_true",
        help="Return exit code 0 even when checks fail.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    baseline = _load_json(args.baseline.expanduser().resolve())
    candidate = _load_json(args.candidate.expanduser().resolve())

    user_contract = {}
    if args.contract is not None:
        user_contract = _load_json(args.contract.expanduser().resolve())
    contract = _merge_contract(user_contract)

    checks: List[Dict[str, Any]] = []

    for rule in contract.get("aggregate_metric_rules", []):
        name = rule["name"]
        checks.append(
            _check_rule(
                scope="aggregate_metric",
                name=name,
                direction=rule.get("direction", "lower"),
                baseline_value=_get_aggregate_metric(baseline, name),
                candidate_value=_get_aggregate_metric(candidate, name),
                rel_tol=float(rule.get("rel_tol", 0.0)),
                abs_tol=float(rule.get("abs_tol", 0.0)),
                required=bool(rule.get("required", True)),
            )
        )

    for rule in contract.get("runtime_rules", []):
        name = rule["name"]
        checks.append(
            _check_rule(
                scope="runtime",
                name=name,
                direction=rule.get("direction", "lower"),
                baseline_value=_get_runtime_metric(baseline, name),
                candidate_value=_get_runtime_metric(candidate, name),
                rel_tol=float(rule.get("rel_tol", 0.0)),
                abs_tol=float(rule.get("abs_tol", 0.0)),
                required=bool(rule.get("required", False)),
            )
        )

    baseline_cases = _case_map(baseline)
    candidate_cases = _case_map(candidate)
    missing_cases, extra_cases = _compare_case_sets(baseline_cases, candidate_cases)

    if contract.get("require_same_case_set", True):
        if missing_cases or extra_cases:
            checks.append(
                {
                    "scope": "case_set",
                    "name": "case_set_match",
                    "direction": "equal",
                    "baseline": sorted(baseline_cases.keys()),
                    "candidate": sorted(candidate_cases.keys()),
                    "passed": False,
                    "reason": "case_set_mismatch",
                    "missing_in_candidate": missing_cases,
                    "extra_in_candidate": extra_cases,
                }
            )

    for rule in contract.get("per_case_rules", []):
        metric_name = rule["name"]
        direction = rule.get("direction", "lower")
        rel_tol = float(rule.get("rel_tol", 0.0))
        abs_tol = float(rule.get("abs_tol", 0.0))
        required = bool(rule.get("required", True))

        for case_name, base_row in baseline_cases.items():
            cand_row = candidate_cases.get(case_name)
            check = _check_rule(
                scope=f"per_case:{case_name}",
                name=metric_name,
                direction=direction,
                baseline_value=_parse_float(base_row.get(metric_name)),
                candidate_value=_parse_float(cand_row.get(metric_name) if cand_row else None),
                rel_tol=rel_tol,
                abs_tol=abs_tol,
                required=required,
            )
            checks.append(check)

    failed = [check for check in checks if not check.get("passed", False)]
    passed = len(failed) == 0

    report = {
        "schema_version": "1.0",
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "contract": contract,
        "passed": passed,
        "total_checks": len(checks),
        "failed_checks": len(failed),
        "checks": checks,
    }

    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote compare report: {output}")

    print(f"Compare result: {'PASS' if passed else 'FAIL'} ({len(checks) - len(failed)}/{len(checks)} checks passed)")
    if failed:
        for item in failed[:20]:
            print(
                f"- FAIL {item.get('scope')}::{item.get('name')} "
                f"(baseline={item.get('baseline')}, candidate={item.get('candidate')}, reason={item.get('reason')})"
            )

    if failed and not args.no_fail_on_regression:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
