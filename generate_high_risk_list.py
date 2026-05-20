import argparse
import csv
from pathlib import Path
from typing import Dict, List


STATUS_PRIORITY = {"篡改": 3, "可疑": 2, "未定位": 1, "正常": 0}
TIER_PRIORITY = {"A": 3, "B": 2, "C": 1, "WATCH": 0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a high-risk review list from amount batch summary CSV.")
    parser.add_argument("--summary-csv", required=True, help="Path to amount_batch_summary.csv")
    parser.add_argument("--output-dir", required=True, help="Directory to write report files")
    parser.add_argument("--top-n", type=int, default=80, help="Max rows to keep in the high-risk list")
    return parser.parse_args()


def risk_tier(row: Dict[str, str]) -> str:
    status = row["top_status"]
    confidence = float(row["top_confidence"] or 0.0)
    reason = row["top_reason"]
    readable_amount = row.get("top_readable_amount", "no") == "yes"

    if status == "篡改":
        if confidence >= 0.99 or (not readable_amount and "局部字体风格异常" in reason):
            return "A"
        if confidence >= 0.95:
            return "B"
        return "C"
    if status == "未定位":
        return "C"
    return "WATCH"


def should_keep(row: Dict[str, str]) -> bool:
    status = row["top_status"]
    confidence = float(row["top_confidence"] or 0.0)
    needs_review = row.get("needs_review", "no") == "yes"

    if status in {"篡改", "未定位"}:
        return True
    if needs_review and confidence >= 0.49:
        return True
    return False


def load_rows(summary_csv: Path) -> List[Dict[str, str]]:
    with summary_csv.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    return rows


def write_csv(rows: List[Dict[str, str]], output_path: Path) -> None:
    if not rows:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: List[Dict[str, str]], output_path: Path, summary_csv: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 高危样本名单",
        "",
        f"- 来源: `{summary_csv}`",
        f"- 样本数: `{len(rows)}`",
        "",
        "| Rank | Tier | Image | Group | Status | Confidence | ReviewPriority | Text | Reason | Preview |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in rows:
        lines.append(
            f"| {row['rank']} | {row['risk_tier']} | {row['image_name']} | {row['group']} | {row['top_status']} | "
            f"{float(row['top_confidence']):.2%} | {float(row['review_priority']):.2f} | {row['top_text'][:24]} | "
            f"{row['top_reason'][:32]} | {row['preview_path']} |"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary_csv = Path(args.summary_csv)
    output_dir = Path(args.output_dir)
    rows = load_rows(summary_csv)

    shortlisted = []
    for row in rows:
        if not should_keep(row):
            continue
        enriched = dict(row)
        enriched["risk_tier"] = risk_tier(row)
        shortlisted.append(enriched)

    shortlisted.sort(
        key=lambda item: (
            TIER_PRIORITY.get(item["risk_tier"], -1),
            STATUS_PRIORITY.get(item["top_status"], -1),
            float(item["review_priority"] or 0.0),
            float(item["top_confidence"] or 0.0),
        ),
        reverse=True,
    )
    shortlisted = shortlisted[: args.top_n]

    for index, row in enumerate(shortlisted, start=1):
        row["rank"] = str(index)

    csv_rows = []
    for row in shortlisted:
        csv_rows.append(
            {
                "rank": row["rank"],
                "risk_tier": row["risk_tier"],
                "image_name": row["image_name"],
                "group": row["group"],
                "top_status": row["top_status"],
                "top_confidence": row["top_confidence"],
                "review_priority": row["review_priority"],
                "top_text": row["top_text"],
                "top_reason": row["top_reason"],
                "top_readable_amount": row.get("top_readable_amount", ""),
                "top_downgraded": row.get("top_downgraded", ""),
                "needs_review": row["needs_review"],
                "preview_path": row["preview_path"],
            }
        )

    write_csv(csv_rows, output_dir / "high_risk_samples.csv")
    write_markdown(csv_rows, output_dir / "high_risk_samples.md", summary_csv)


if __name__ == "__main__":
    main()
