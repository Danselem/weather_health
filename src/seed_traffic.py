"""Send reference-like rows to the live API for demo monitoring windows."""

import argparse
import concurrent.futures
import os

import pandas as pd
import requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    frame = pd.read_csv("data/processed/x_test.csv").sample(
        n=min(args.count, 997), random_state=1024
    )
    records = frame.to_dict(orient="records")
    headers = {"Content-Type": "application/json"}
    if os.getenv("API_KEY"):
        headers["X-API-Key"] = os.environ["API_KEY"]

    def send(row: dict) -> int:
        for attempt in range(3):
            try:
                return requests.post(
                    f"{args.url.rstrip('/')}/predict",
                    json=row,
                    headers=headers,
                    timeout=30,
                ).status_code
            except requests.RequestException:
                if attempt == 2:
                    return 0
        return 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        codes = list(pool.map(send, records))

    ok = sum(code == 200 for code in codes)
    print(f"sent={len(codes)} ok={ok} errors={len(codes) - ok}")


if __name__ == "__main__":
    main()
