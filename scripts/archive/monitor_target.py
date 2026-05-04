import argparse
import csv
from pathlib import Path


def _to_float(value):
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _pick_score(row, metric):
    if metric:
        return _to_float(row.get(metric))
    if row.get("mode") == "cv":
        return _to_float(row.get("accuracy_mean"))
    return _to_float(row.get("accuracy"))


def _load_rows(summary_csv):
    with open(summary_csv, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _filter_hits(rows, threshold, metric):
    hits = []
    for row in rows:
        score = _pick_score(row, metric)
        if score is None:
            continue
        if score >= threshold:
            item = dict(row)
            item["score"] = score
            hits.append(item)
    hits.sort(key=lambda x: x["score"], reverse=True)
    return hits


def _write_md(path, hits, threshold, metric, total_rows):
    metric_label = metric if metric else "auto(mode-aware accuracy)"
    lines = [
        "# Target Monitor",
        "",
        f"Threshold: {threshold:.4f}",
        f"Metric: {metric_label}",
        f"Hits: {len(hits)} / {total_rows}",
        "",
        "| Rank | Score | Config | Tag | Exp | Mode | Timestamp |",
        "| --- | ---: | --- | --- | --- | --- | --- |",
    ]

    for idx, row in enumerate(hits, start=1):
        lines.append(
            "| {rank} | {score:.4f} | {config} | {tag} | {exp} | {mode} | {timestamp} |".format(
                rank=idx,
                score=row.get("score", 0.0),
                config=row.get("config", ""),
                tag=row.get("tag", ""),
                exp=row.get("exp", ""),
                mode=row.get("mode", ""),
                timestamp=row.get("timestamp", ""),
            )
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path, hits):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "score",
        "config",
        "tag",
        "exp",
        "mode",
        "timestamp",
        "accuracy",
        "accuracy_mean",
        "accuracy_std",
        "balanced_accuracy",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "macro_f1",
        "macro_f1_mean",
        "macro_f1_std",
    ]

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(hits, start=1):
            out = {k: row.get(k, "") for k in fieldnames}
            out["rank"] = idx
            out["score"] = f"{row.get('score', 0.0):.6f}"
            writer.writerow(out)


def main():
    parser = argparse.ArgumentParser(description="Monitor runs that meet accuracy target")
    parser.add_argument("--summary-csv", default="outputs/logs/summary.csv")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--metric", default="")
    parser.add_argument("--out-md", default="outputs/logs/target_hits.md")
    parser.add_argument("--out-csv", default="outputs/logs/target_hits.csv")
    args = parser.parse_args()

    summary_path = Path(args.summary_csv)
    if not summary_path.exists():
        print(f"Summary CSV not found: {summary_path}")
        print("No target monitoring report generated.")
        return

    rows = _load_rows(summary_path)
    hits = _filter_hits(rows, args.threshold, args.metric.strip() or None)
    _write_md(Path(args.out_md), hits, args.threshold, args.metric.strip() or None, len(rows))
    _write_csv(Path(args.out_csv), hits)

    print(f"Loaded rows: {len(rows)}")
    print(f"Target hits: {len(hits)}")
    print(f"Saved markdown: {args.out_md}")
    print(f"Saved csv: {args.out_csv}")


if __name__ == "__main__":
    main()
