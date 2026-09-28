import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime
from importlib import metadata
from pathlib import Path
from statistics import mean, median


BOOL_FIELDS = [
    "phone_object_detected",
    "raw_phone_call_detected",
    "phone_call_detected",
    "raw_drinking_detected",
    "drinking_detected",
    "raw_zone_alert",
    "zone_alert",
]

PACKAGE_NAMES = [
    "opencv-python",
    "mediapipe",
    "ultralytics",
    "numpy",
]


def parse_bool(value):
    return str(value).strip().lower() == "true"


def parse_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_timestamp(value):
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return None


def normalize_status(value):
    return str(value).replace("Driver Behaviour Status:", "").strip() or "Unknown"


def split_detected_objects(value):
    text = str(value).strip()
    if not text or text.lower() == "none":
        return []

    return [item.strip() for item in text.split(",") if item.strip()]


def count_alert_segments(rows, field):
    segments = 0
    previous = False
    previous_frame = None

    for row in rows:
        frame_number = parse_int(row.get("frame_number"))
        active = parse_bool(row.get(field))
        new_session = previous_frame is not None and frame_number <= previous_frame

        if new_session:
            previous = False

        if active and not previous:
            segments += 1

        previous = active
        previous_frame = frame_number

    return segments


def max_streak(rows, field):
    longest = 0
    current = 0
    previous_frame = None

    for row in rows:
        frame_number = parse_int(row.get("frame_number"))
        new_session = previous_frame is not None and frame_number <= previous_frame

        if new_session:
            current = 0

        if parse_bool(row.get(field)):
            current += 1
            longest = max(longest, current)
        else:
            current = 0

        previous_frame = frame_number

    return longest


def count_sessions(rows):
    sessions = 0
    previous_frame = None

    for row in rows:
        frame_number = parse_int(row.get("frame_number"))

        if previous_frame is None or frame_number <= previous_frame:
            sessions += 1

        previous_frame = frame_number

    return sessions


def classify_screenshot(path):
    name = path.name.lower()
    labels = []

    for label in ("eyes_closed", "phone_call", "drinking", "zone"):
        if label in name:
            labels.append(label)

    return "+".join(labels) if labels else "manual_or_other"


def latest_screenshots_by_category(screenshot_dir):
    latest = {}

    if not screenshot_dir.exists():
        return latest

    for path in sorted(screenshot_dir.glob("*.png"), key=lambda item: item.stat().st_mtime):
        category = classify_screenshot(path)

        if category == "manual_or_other":
            continue

        for label in category.split("+"):
            latest[label] = path

    return latest


def fit_image_to_box(image, target_width, target_height, cv2, np):
    canvas = np.full((target_height, target_width, 3), 245, dtype=np.uint8)

    if image is None:
        return canvas

    source_height, source_width = image.shape[:2]

    if source_height <= 0 or source_width <= 0:
        return canvas

    scale = min(target_width / source_width, target_height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    x_offset = (target_width - resized_width) // 2
    y_offset = (target_height - resized_height) // 2
    canvas[y_offset:y_offset + resized_height, x_offset:x_offset + resized_width] = resized

    return canvas


def build_contact_sheet(screenshot_dir, output_path):
    try:
        import cv2
        import numpy as np
    except ImportError:
        return []

    latest = latest_screenshots_by_category(screenshot_dir)
    categories = [
        ("Safe-zone alert", "zone"),
        ("Eyes possibly closed", "eyes_closed"),
        ("Phone-call alert", "phone_call"),
        ("Drinking alert", "drinking"),
    ]
    selected = []

    tile_width = 560
    tile_height = 340
    label_height = 42
    tiles = []

    for title, key in categories:
        path = latest.get(key)

        if path is None:
            continue

        image = cv2.imread(str(path))
        tile = fit_image_to_box(image, tile_width, tile_height, cv2, np)
        labeled_tile = np.full((tile_height + label_height, tile_width, 3), 255, dtype=np.uint8)
        labeled_tile[:tile_height, :] = tile
        cv2.rectangle(labeled_tile, (0, tile_height), (tile_width - 1, tile_height + label_height - 1), (32, 36, 45), -1)
        cv2.putText(
            labeled_tile,
            title,
            (14, tile_height + 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        tiles.append(labeled_tile)
        selected.append(path)

    if not tiles:
        return []

    while len(tiles) < 4:
        blank = np.full((tile_height + label_height, tile_width, 3), 245, dtype=np.uint8)
        cv2.putText(
            blank,
            "No screenshot available",
            (50, (tile_height + label_height) // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (70, 70, 70),
            2,
            cv2.LINE_AA,
        )
        tiles.append(blank)

    top_row = np.hstack(tiles[:2])
    bottom_row = np.hstack(tiles[2:4])
    sheet = np.vstack([top_row, bottom_row])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)

    return selected


def package_versions():
    versions = {}

    for package_name in PACKAGE_NAMES:
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            versions[package_name] = "not installed"

    return versions


def analyze_event_log(log_path, screenshot_dir, contact_sheet_path):
    with log_path.open("r", encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))

    if not rows:
        raise ValueError(f"No rows found in {log_path}")

    timestamps = [parse_timestamp(row.get("timestamp")) for row in rows]
    timestamps = [value for value in timestamps if value is not None]
    risk_scores = [parse_int(row.get("risk_score")) for row in rows]

    status_counts = Counter(normalize_status(row.get("final_status")) for row in rows)
    eye_status_counts = Counter(row.get("eye_status", "Unknown") for row in rows)
    hand_status_counts = Counter(row.get("hand_status", "Unknown") for row in rows)
    object_counts = Counter()

    for row in rows:
        object_counts.update(split_detected_objects(row.get("detected_objects")))

    bool_frame_counts = {
        field: sum(1 for row in rows if parse_bool(row.get(field)))
        for field in BOOL_FIELDS
    }
    bool_segment_counts = {
        field: count_alert_segments(rows, field)
        for field in BOOL_FIELDS
    }
    bool_max_streaks = {
        field: max_streak(rows, field)
        for field in BOOL_FIELDS
    }

    screenshot_counts = Counter()
    screenshot_count_total = 0

    if screenshot_dir.exists():
        for path in screenshot_dir.glob("*.png"):
            screenshot_count_total += 1
            screenshot_counts[classify_screenshot(path)] += 1

    selected_contact_sheet_paths = build_contact_sheet(screenshot_dir, contact_sheet_path)

    metrics = {
        "event_log_path": str(log_path),
        "rows": len(rows),
        "sessions_estimated": count_sessions(rows),
        "first_timestamp": timestamps[0].isoformat(sep=" ") if timestamps else "unknown",
        "last_timestamp": timestamps[-1].isoformat(sep=" ") if timestamps else "unknown",
        "risk_min": min(risk_scores),
        "risk_max": max(risk_scores),
        "risk_mean": round(mean(risk_scores), 2),
        "risk_median": round(median(risk_scores), 2),
        "screenshot_total": screenshot_count_total,
        "contact_sheet_path": str(contact_sheet_path),
        "contact_sheet_images": len(selected_contact_sheet_paths),
    }

    for status, count in sorted(status_counts.items()):
        metrics[f"status_frames_{status.lower().replace(' ', '_')}"] = count

    for status, count in sorted(eye_status_counts.items()):
        metrics[f"eye_status_frames_{status.lower().replace(' ', '_').replace('.', '')}"] = count

    for status, count in sorted(hand_status_counts.items()):
        metrics[f"hand_status_frames_{status.lower().replace(' ', '_')}"] = count

    for field, count in bool_frame_counts.items():
        metrics[f"{field}_frames"] = count
        metrics[f"{field}_segments"] = bool_segment_counts[field]
        metrics[f"{field}_max_streak"] = bool_max_streaks[field]

    for object_name, count in sorted(object_counts.items()):
        metrics[f"object_frames_{object_name.replace(' ', '_')}"] = count

    for category, count in sorted(screenshot_counts.items()):
        metrics[f"screenshots_{category}"] = count

    return metrics, status_counts, eye_status_counts, hand_status_counts, object_counts, screenshot_counts, selected_contact_sheet_paths


def write_metrics_csv(metrics, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["metric", "value"])

        for key in sorted(metrics):
            writer.writerow([key, metrics[key]])


def markdown_counter_table(counter, first_column):
    lines = [
        f"| {first_column} | Count |",
        "| --- | ---: |",
    ]

    for key, count in sorted(counter.items()):
        lines.append(f"| {key} | {count} |")

    return "\n".join(lines)


def write_summary_markdown(
    metrics,
    status_counts,
    eye_status_counts,
    hand_status_counts,
    object_counts,
    screenshot_counts,
    selected_contact_sheet_paths,
    output_path,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    versions = package_versions()

    lines = [
        "# Preliminary Prototype Evaluation Summary",
        "",
        "This summary was generated from the real CabInspector event log and screenshot folder. It is an operational prototype evaluation, not a ground-truth accuracy benchmark, because the current repository does not contain manually labelled test clips.",
        "",
        "## Run Evidence",
        "",
        f"- Event log: `{metrics['event_log_path']}`",
        f"- Rows analysed: {metrics['rows']}",
        f"- Estimated live-run sessions: {metrics['sessions_estimated']}",
        f"- Timestamp range: {metrics['first_timestamp']} to {metrics['last_timestamp']}",
        f"- Screenshots found: {metrics['screenshot_total']}",
        f"- Contact sheet: `{metrics['contact_sheet_path']}`",
        "",
        "## Environment",
        "",
    ]

    for package_name, version in versions.items():
        lines.append(f"- {package_name}: {version}")

    lines.extend([
        "",
        "## Risk Score",
        "",
        f"- Minimum risk score: {metrics['risk_min']}",
        f"- Maximum risk score: {metrics['risk_max']}",
        f"- Mean risk score: {metrics['risk_mean']}",
        f"- Median risk score: {metrics['risk_median']}",
        "",
        "## Final Status Counts",
        "",
        markdown_counter_table(status_counts, "Final status"),
        "",
        "## Eye Status Counts",
        "",
        markdown_counter_table(eye_status_counts, "Eye status"),
        "",
        "## Hand Status Counts",
        "",
        markdown_counter_table(hand_status_counts, "Hand status"),
        "",
        "## Confirmed Alert Frame Counts",
        "",
        "| Signal | Frames | Segments | Longest streak |",
        "| --- | ---: | ---: | ---: |",
    ])

    for field in BOOL_FIELDS:
        lines.append(
            f"| {field} | {metrics.get(field + '_frames', 0)} | "
            f"{metrics.get(field + '_segments', 0)} | "
            f"{metrics.get(field + '_max_streak', 0)} |"
        )

    lines.extend([
        "",
        "## Detected Object Frame Counts",
        "",
        markdown_counter_table(object_counts, "Detected object") if object_counts else "No object detections were logged.",
        "",
        "## Screenshot Counts",
        "",
        markdown_counter_table(screenshot_counts, "Screenshot category") if screenshot_counts else "No screenshots were found.",
        "",
        "## Contact Sheet Source Images",
        "",
    ])

    if selected_contact_sheet_paths:
        for path in selected_contact_sheet_paths:
            lines.append(f"- `{path}`")
    else:
        lines.append("- No screenshots were available for the contact sheet.")

    lines.extend([
        "",
        "## Evaluation Limits",
        "",
        "- The log shows that the implemented pipeline can produce face/eye, hand, safe-zone, object, action, screenshot and risk-score outputs during live runs.",
        "- These counts do not measure precision, recall or false-positive rate because no manually labelled ground-truth clips are present in the repository.",
        "- The next evaluation step should use a small labelled set of recorded clips for normal driving, phone use, drinking, eyes closed and safe-zone violations.",
        "",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Evaluate CabInspector event-log evidence.")
    parser.add_argument("--log", default="outputs/events_log.csv", type=Path)
    parser.add_argument("--screenshots", default="outputs/screenshots", type=Path)
    parser.add_argument("--summary", default="docs/evaluation/preliminary_event_log_summary.md", type=Path)
    parser.add_argument("--metrics", default="docs/evaluation/preliminary_event_log_metrics.csv", type=Path)
    parser.add_argument("--contact-sheet", default="docs/assets/prototype_contact_sheet.png", type=Path)
    args = parser.parse_args()

    if not args.log.exists():
        raise FileNotFoundError(f"Event log not found: {args.log}")

    result = analyze_event_log(args.log, args.screenshots, args.contact_sheet)
    (
        metrics,
        status_counts,
        eye_status_counts,
        hand_status_counts,
        object_counts,
        screenshot_counts,
        selected_contact_sheet_paths,
    ) = result

    write_metrics_csv(metrics, args.metrics)
    write_summary_markdown(
        metrics,
        status_counts,
        eye_status_counts,
        hand_status_counts,
        object_counts,
        screenshot_counts,
        selected_contact_sheet_paths,
        args.summary,
    )

    print(f"Rows analysed: {metrics['rows']}")
    print(f"Estimated sessions: {metrics['sessions_estimated']}")
    print(f"Risk score max: {metrics['risk_max']}")
    print(f"Summary written: {args.summary}")
    print(f"Metrics written: {args.metrics}")
    print(f"Contact sheet written: {args.contact_sheet}")


if __name__ == "__main__":
    main()
