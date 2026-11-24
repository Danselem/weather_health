#!/usr/bin/env bash
# Copy selected DVC report figures into tracked documentation snapshots.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p images/reports
cp reports/ablation_metrics.png images/reports/ablation-metrics.png
cp reports/reliability_diagram.png images/reports/reliability-diagram.png
cp reports/slice_metrics.png images/reports/slice-metrics.png
cp reports/shap_summary.png images/reports/shap-summary.png
cp reports/confusion_matrix.png images/reports/confusion-matrix.png
echo "Published report figures to images/reports/"
