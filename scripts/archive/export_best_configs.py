import argparse
import csv
from pathlib import Path
from collections import defaultdict


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


def _best_by_group(rows, group_by, metric):
    grouped = defaultdict(list)
    for row in rows:
        key = row.get(group_by, "") or "(empty)"
        grouped[key].append(row)

    best = {}
    for key, group_rows in grouped.items():
        scored = []
        for row in group_rows:
            score = _pick_score(row, metric)
            if score is not None:
                item = dict(row)
                item["score"] = score
                scored.append(item)
        if scored:
            best[key] = max(scored, key=lambda x: x["score"])
    return best


def _write_md(path, best_map, group_by, metric):
    metric_label = metric if metric else "auto(mode-aware accuracy)"
    lines = [
        "# Best Configs",
        "",
        f"Grouped by: {group_by}",
        f"Metric: {metric_label}",
        "",
        "| Group | Score | Config | Tag | Exp | Mode | Timestamp |",
        "| --- | ---: | --- | --- | --- | --- | --- |",
    ]

    for key in sorted(best_map.keys()):
        row = best_map[key]
        lines.append(
            "| {group} | {score:.4f} | {config} | {tag} | {exp} | {mode} | {timestamp} |".format(
                group=key,
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


def _write_csv(path, best_map):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
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
        "base_model_type",
        "meta_model_type",
    ]

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(best_map.keys()):
            row = best_map[key]
            out = {k: row.get(k, "") for k in fieldnames}
            out["group"] = key
            out["score"] = f"{row.get('score', 0.0):.6f}"
            writer.writerow(out)


def main():
    parser = argparse.ArgumentParser(description="Export best configurations by group from outputs/logs/summary.csv")
    parser.add_argument("--summary-csv", default="outputs/logs/summary.csv")
    parser.add_argument("--group-by", default="exp", choices=["exp", "tag", "mode", "config"])
    parser.add_argument("--metric", default="")
    parser.add_argument("--out-md", default="outputs/logs/best_configs.md")
    parser.add_argument("--out-csv", default="outputs/logs/best_configs.csv")
    args = parser.parse_args()

    summary_path = Path(args.summary_csv)
    if not summary_path.exists():
        print(f"Summary CSV not found: {summary_path}")
        print("No best config exported.")
        return

    rows = _load_rows(summary_path)
    best_map = _best_by_group(rows, args.group_by, args.metric.strip() or None)
    _write_md(Path(args.out_md), best_map, args.group_by, args.metric.strip() or None)
    _write_csv(Path(args.out_csv), best_map)

    print(f"Loaded rows: {len(rows)}")
    print(f"Groups with best configs: {len(best_map)}")
    print(f"Saved markdown: {args.out_md}")
    print(f"Saved csv: {args.out_csv}")


if __name__ == "__main__":
    main()
