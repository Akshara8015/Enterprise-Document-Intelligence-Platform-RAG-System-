import json
import os

METRICS_FILE = "monitoring_metrics.json"

def init_metrics():
    if not os.path.exists(METRICS_FILE):
        metrics = {
            "total_queries": 0,
            "successful_answers": 0,
            "blocked_queries": 0,
            "low_confidence_answers": 0,
            "avg_confidence": 0.0,
            "avg_sources_used": 0,
            "avg_latency": 0.0
        }
        save_metrics(metrics)

def load_metrics():
    if not os.path.exists(METRICS_FILE):
        init_metrics()
    with open(METRICS_FILE, "r") as f:
        return json.load(f)


def save_metrics(metrics):
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

def update_metrics(
    confidence=None,
    sources_used=0,
    latency=0.0,
    blocked=False,
    success=False
):
    metrics = load_metrics()
    metrics["total_queries"] += 1

    if blocked:
        metrics["blocked_queries"] += 1

    if success:
        metrics["successful_answers"] += 1

    if confidence is not None:
        if confidence < 0.6:
            metrics["low_confidence_answers"] += 1

        # Running average
        n = metrics["successful_answers"]
        metrics["avg_confidence"] = round(
            ((metrics["avg_confidence"] * (n - 1)) + confidence) / max(n, 1),
            3
        )

    if sources_used > 0:
        n = metrics["successful_answers"]
        metrics["avg_sources_used"] = round(
            ((metrics["avg_sources_used"] * (n - 1)) + sources_used) / max(n, 1),
            2
        )

    if latency > 0:
        n = metrics["total_queries"]
        metrics["avg_latency"] = round(
            ((metrics["avg_latency"] * (n - 1)) + latency) / n,
            2
        )

    save_metrics(metrics)
