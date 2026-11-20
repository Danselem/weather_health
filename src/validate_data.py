"""Lightweight data contract checks for the cleaned dataset."""

import json
from pathlib import Path

import pandas as pd

from src.config import load_config

NUMERIC_RANGES = {
    "Age": (0, 120),
    "Temperature (C)": (-60, 60),
    "Humidity": (0, 1),
    "Wind Speed (km/h)": (0, 200),
}
NON_BINARY = set(NUMERIC_RANGES)


def validate(path: str | Path, target: str) -> dict:
    df = pd.read_parquet(path)
    if "uuid" in df.columns:
        df = df.drop(columns=["uuid"])

    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(ok), "detail": detail})

    check("non_empty", len(df) > 0, f"{len(df)} rows")
    check("target_present", target in df.columns, target)
    check("no_missing_values", not df.isna().any().any(), f"{int(df.isna().sum().sum())} missing")

    features = [c for c in df.columns if c != target]
    binary = [c for c in features if c not in NON_BINARY]
    bad_binary = [c for c in binary if not set(df[c].unique()).issubset({0, 1})]
    check("binary_features", not bad_binary, f"invalid: {bad_binary[:5]}")

    for column, (low, high) in NUMERIC_RANGES.items():
        if column in df.columns:
            ok = df[column].between(low, high).all()
            check(f"range_{column}", bool(ok), f"[{low}, {high}]")

    if target in df.columns:
        counts = df[target].value_counts()
        check("at_least_two_classes", len(counts) >= 2, f"{len(counts)} classes")
        check("min_class_size", int(counts.min()) >= 2, f"min={int(counts.min())}")
    else:
        counts = pd.Series(dtype=int)

    feature_cols = [c for c in df.columns if c != target]
    conflicts = df.groupby(feature_cols, dropna=False)[target].nunique() if target in df else pd.Series()
    check(
        "no_conflicting_labels",
        int((conflicts > 1).sum()) == 0,
        f"{int((conflicts > 1).sum())} conflicting feature rows",
    )

    report = {
        "passed": all(c["passed"] for c in checks),
        "rows": int(len(df)),
        "features": len(feature_cols),
        "classes": int(len(counts)),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "checks": checks,
    }
    return report


def main() -> None:
    cfg = load_config("config")
    report = validate(cfg.data.interim_data_path, cfg.data.target_col)
    out = Path(cfg.reports.data_validation_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "rows": report["rows"], "classes": report["classes"]}))
    if not report["passed"]:
        failed = [c for c in report["checks"] if not c["passed"]]
        raise SystemExit(f"Data validation failed: {failed}")


if __name__ == "__main__":
    main()
