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


def _load_rows(summary_csv):
    with open(summary_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _filter_rows(rows, tag=None, mode=None):
    filtered = rows
    if tag:
        filtered = [row for row in filtered if row.get("tag") == tag]
    if mode:
        filtered = [row for row in filtered if row.get("mode") == mode]
    return filtered


def _pick_metric(row, preferred_metric):
    if preferred_metric:
        return _to_float(row.get(preferred_metric))

    mode = row.get("mode")
    if mode == "cv":
        return _to_float(row.get("accuracy_mean"))
    return _to_float(row.get("accuracy"))


def _rank_rows(rows, metric_key):
    ranked = []
    for row in rows:
        score = _pick_metric(row, metric_key)
        if score is None:
            continue
        row_copy = dict(row)
        row_copy["score"] = score
        ranked.append(row_copy)

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


def _group_rows(ranked_rows, group_by):
    if not group_by:
        return {"all": ranked_rows}

    grouped = defaultdict(list)
    for row in ranked_rows:
        key = row.get(group_by, "") or "(empty)"
        grouped[key].append(row)
    return dict(grouped)


def _write_markdown(path, ranked_rows, metric_key, total_count):
    metric_label = metric_key if metric_key else "auto(mode-aware accuracy)"
    lines = [
        "# Experiment Leaderboard",
        "",
        f"Total ranked runs: {len(ranked_rows)} / {total_count}",
        f"Metric: {metric_label}",
        "",
        "| Rank | Timestamp | Tag | Exp | Mode | Score | Config |",
        "| --- | --- | --- | --- | --- | ---: | --- |",
    ]

    for idx, row in enumerate(ranked_rows, start=1):
        lines.append(
            "| {rank} | {timestamp} | {tag} | {exp} | {mode} | {score:.4f} | {config} |".format(
                rank=idx,
                timestamp=row.get("timestamp", ""),
                tag=row.get("tag", ""),
                exp=row.get("exp", ""),
                mode=row.get("mode", ""),
                score=row.get("score", 0.0),
                config=row.get("config", ""),
            )
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_grouped_markdown(path, grouped_rows, metric_key, total_count, top_k):
    metric_label = metric_key if metric_key else "auto(mode-aware accuracy)"
    lines = [
        "# Grouped Experiment Leaderboard",
        "",
        f"Total ranked runs: {sum(len(v) for v in grouped_rows.values())} / {total_count}",
        f"Metric: {metric_label}",
        "",
    ]

    for group_name in sorted(grouped_rows.keys()):
        rows = grouped_rows[group_name]
        lines.append(f"## Group: {group_name}")
        lines.append("")
        lines.append("| Rank | Timestamp | Tag | Exp | Mode | Score | Config |")
        lines.append("| --- | --- | --- | --- | --- | ---: | --- |")
        for idx, row in enumerate(rows[:top_k] if top_k > 0 else rows, start=1):
            lines.append(
                "| {rank} | {timestamp} | {tag} | {exp} | {mode} | {score:.4f} | {config} |".format(
                    rank=idx,
                    timestamp=row.get("timestamp", ""),
                    tag=row.get("tag", ""),
                    exp=row.get("exp", ""),
                    mode=row.get("mode", ""),
                    score=row.get("score", 0.0),
                    config=row.get("config", ""),
                )
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(path, ranked_rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not ranked_rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = [
        "rank",
        "timestamp",
        "tag",
        "exp",
        "mode",
        "score",
        "config",
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
        for idx, row in enumerate(ranked_rows, start=1):
            out = {k: row.get(k, "") for k in fieldnames}
            out["rank"] = idx
            out["score"] = f"{row.get('score', 0.0):.6f}"
            writer.writerow(out)


def main():
    parser = argparse.ArgumentParser(description="Summarize experiment runs from outputs/logs/summary.csv")
    parser.add_argument("--summary-csv", default="outputs/logs/summary.csv")
    parser.add_argument("--tag", default="")
    parser.add_argument("--mode", default="")
    parser.add_argument("--metric", default="", help="Column name to rank by, e.g. accuracy_mean")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--out-md", default="outputs/logs/leaderboard.md")
    parser.add_argument("--out-csv", default="outputs/logs/leaderboard.csv")
    parser.add_argument("--group-by", default="", choices=["", "exp", "tag", "mode", "config"])
    parser.add_argument("--out-grouped-md", default="outputs/logs/leaderboard_grouped.md")
    args = parser.parse_args()

    summary_path = Path(args.summary_csv)
    if not summary_path.exists():
        print(f"Summary CSV not found: {summary_path}")
        print("No rows to summarize.")
        return

    rows = _load_rows(summary_path)
    filtered = _filter_rows(rows, tag=args.tag.strip() or None, mode=args.mode.strip() or None)
    ranked = _rank_rows(filtered, args.metric.strip() or None)

    if args.top_k > 0:
        ranked = ranked[: args.top_k]

    _write_markdown(Path(args.out_md), ranked, args.metric.strip() or None, len(filtered))
    _write_csv(Path(args.out_csv), ranked)

    grouped_rows = _group_rows(_rank_rows(filtered, args.metric.strip() or None), args.group_by.strip() or None)
    _write_grouped_markdown(
        Path(args.out_grouped_md),
        grouped_rows,
        args.metric.strip() or None,
        len(filtered),
        args.top_k,
    )

    print(f"Loaded rows: {len(rows)}")
    print(f"Filtered rows: {len(filtered)}")
    print(f"Ranked rows: {len(ranked)}")
    print(f"Saved markdown: {args.out_md}")
    print(f"Saved csv: {args.out_csv}")
    print(f"Saved grouped markdown: {args.out_grouped_md}")


if __name__ == "__main__":
    main()
