# ADR 0004: Drift monitoring and guarded retraining

## Status
Accepted

## Context
The original monitoring job re-scored training data, double-scaled features, and
had no alert or retrain path.

## Decision
Log every prediction, compare current traffic to a pinned reference with
Evidently tests, write metrics to Grafana, send optional webhook alerts, and
trigger retraining only after consecutive failed windows. Retraining never
auto-deploys.

## Consequences
- Monitoring reflects production traffic, not training data.
- Alerting requires optional webhook secrets.
- A human still approves deployment after the quality gate.
