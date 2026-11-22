# Load testing

Locust profile for `/predict`, `/feedback`, `/health`, and `/stats`.

The live API is rate-limited to 60 requests/minute per instance/IP, so benchmark
against a local server with the limiter disabled:

```bash
RATE_LIMIT_PER_MINUTE=0 uv run uvicorn service.app:app --port 18084
LOCUST_HOST=http://127.0.0.1:18084 make loadtest
```

Results are written to `reports/loadtest_*` (excluded from Git).

Optional API key:

```bash
LOADTEST_API_KEY=... LOCUST_HOST=... make loadtest
```

Do not run this as a CI gate; it is a manual performance measurement.

## Reference result

Warmed local API, 10 users, 20 seconds, rate limiting disabled:

- 317 requests
- 16.1 req/s
- 0 failures
- p50 / p95 / p99: 13 / 27 / 38 ms
