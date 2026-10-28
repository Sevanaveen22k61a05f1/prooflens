
"""
ProofLens - Mismatch Threshold Evaluation

Evaluates image-caption similarity scores across multiple thresholds.
Does not modify the detector's configured threshold automatically.

Required dataset columns:
image_path,caption,label

Labels must be:
match
mismatch
"""

import csv
from pathlib import Path

from mismatch_detector import MismatchDetector


BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "mismatch_eval.csv"
OUTPUT_CSV = BASE_DIR / "mismatch_eval_results.csv"
SWEEP_CSV = BASE_DIR / "mismatch_threshold_sweep.csv"


def calculate_metrics(actual, predicted):
    """Calculate classification metrics; mismatch is the positive class."""

    tp = sum(
        a == "mismatch" and p == "mismatch"
        for a, p in zip(actual, predicted)
    )
    tn = sum(
        a == "match" and p == "match"
        for a, p in zip(actual, predicted)
    )
    fp = sum(
        a == "match" and p == "mismatch"
        for a, p in zip(actual, predicted)
    )
    fn = sum(
        a == "mismatch" and p == "match"
        for a, p in zip(actual, predicted)
    )

    total = len(actual)

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    balanced_accuracy = (recall + specificity) / 2

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def predict_label(score, threshold):
    """Apply the existing lower-score-means-mismatch rule."""

    return "mismatch" if score < threshold else "match"


def load_dataset():
    """Load and validate the labelled CSV."""

    if not INPUT_CSV.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_CSV}"
        )

    with INPUT_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        required = {"image_path", "caption", "label"}

        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                "CSV must contain: image_path,caption,label"
            )

        rows = list(reader)

    if not rows:
        raise ValueError("Dataset contains no examples.")

    for row_number, row in enumerate(rows, start=2):
        row["label"] = row["label"].strip().lower()
        row["image_path"] = row["image_path"].strip()
        row["caption"] = row["caption"].strip()

        if row["label"] not in {"match", "mismatch"}:
            raise ValueError(
                f"Row {row_number}: label must be "
                "'match' or 'mismatch'."
            )

        if not row["image_path"] or not row["caption"]:
            raise ValueError(
                f"Row {row_number}: image path and caption "
                "must not be empty."
            )

    return rows


def build_threshold_candidates(scores, current_threshold):
    """
    Generate thresholds that test every distinct score partition.

    Predictions change when a threshold crosses an observed score.
    The current configured threshold is included for comparison.
    """

    unique_scores = sorted(set(scores))

    if not unique_scores:
        return []

    epsilon = 0.000001

    candidates = [
        unique_scores[0] - epsilon,
        unique_scores[-1] + epsilon,
        current_threshold,
    ]

    for index in range(len(unique_scores) - 1):
        midpoint = (
            unique_scores[index] + unique_scores[index + 1]
        ) / 2

        candidates.append(midpoint)

    return sorted(set(candidates))


def main():
    rows = load_dataset()

    print(f"Dataset examples: {len(rows)}")
    print("Loading CLIP once for all evaluations...")

    detector = MismatchDetector()
    evaluation_rows = []
    detailed_results = []

    for index, row in enumerate(rows, start=1):
        image_path = Path(row["image_path"])

        if not image_path.is_absolute():
            image_path = BASE_DIR / image_path

        caption = row["caption"]
        actual_label = row["label"]

        print(
            f"\n[{index}/{len(rows)}] "
            f"{image_path.name}"
        )
        print(f"Caption: {caption}")
        print(f"Actual label: {actual_label}")

        try:
            result = detector.analyze_image(
                image_path=image_path,
                caption=caption,
            )

            score = float(result["score"])

            current_prediction = predict_label(
                score,
                detector.threshold,
            )

            correct = current_prediction == actual_label

            evaluation_rows.append({
                "image_path": str(image_path),
                "caption": caption,
                "actual_label": actual_label,
                "score": score,
            })

            detailed_results.append({
                "image_path": str(image_path),
                "caption": caption,
                "actual_label": actual_label,
                "predicted_label_at_current_threshold": (
                    current_prediction
                ),
                "cosine_similarity": score,
                "current_threshold": detector.threshold,
                "correct_at_current_threshold": correct,
                "error": "",
            })

            print(f"Similarity: {score:.4f}")
            print(f"Current prediction: {current_prediction}")
            print(f"Correct: {correct}")

        except Exception as exc:
            print(f"ERROR: {type(exc).__name__}: {exc}")

            detailed_results.append({
                "image_path": str(image_path),
                "caption": caption,
                "actual_label": actual_label,
                "predicted_label_at_current_threshold": "",
                "cosine_similarity": "",
                "current_threshold": detector.threshold,
                "correct_at_current_threshold": "",
                "error": f"{type(exc).__name__}: {exc}",
            })

    detail_columns = [
        "image_path",
        "caption",
        "actual_label",
        "predicted_label_at_current_threshold",
        "cosine_similarity",
        "current_threshold",
        "correct_at_current_threshold",
        "error",
    ]

    with OUTPUT_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=detail_columns,
        )
        writer.writeheader()
        writer.writerows(detailed_results)

    print(f"\nDetailed results saved to: {OUTPUT_CSV}")

    if not evaluation_rows:
        print("No examples were evaluated successfully.")
        return

    actual = [
        row["actual_label"]
        for row in evaluation_rows
    ]

    scores = [
        row["score"]
        for row in evaluation_rows
    ]

    if len(set(actual)) < 2:
        print(
            "WARNING: The successfully evaluated dataset contains "
            "only one class. Add both match and mismatch examples."
        )

    print("\n===== CURRENT THRESHOLD =====")

    current_predictions = [
        predict_label(score, detector.threshold)
        for score in scores
    ]

    current_metrics = calculate_metrics(
        actual,
        current_predictions,
    )

    print(f"Threshold:         {detector.threshold:.6f}")
    print(f"Accuracy:          {current_metrics['accuracy']:.3f}")
    print(f"Precision:         {current_metrics['precision']:.3f}")
    print(f"Recall:            {current_metrics['recall']:.3f}")
    print(
        f"Balanced accuracy: "
        f"{current_metrics['balanced_accuracy']:.3f}"
    )
    print(f"F1 score:          {current_metrics['f1']:.3f}")

    candidates = build_threshold_candidates(
        scores,
        detector.threshold,
    )

    sweep_results = []

    for threshold in candidates:
        predictions = [
            predict_label(score, threshold)
            for score in scores
        ]

        metrics = calculate_metrics(
            actual,
            predictions,
        )

        sweep_results.append({
            "threshold": threshold,
            **metrics,
        })

    sweep_columns = [
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "balanced_accuracy",
        "f1",
        "tp",
        "tn",
        "fp",
        "fn",
    ]

    with SWEEP_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=sweep_columns,
        )
        writer.writeheader()
        writer.writerows(sweep_results)

    print(f"\nThreshold sweep saved to: {SWEEP_CSV}")

    # Prefer balanced accuracy, then F1, when ranking candidates.
    best = max(
        sweep_results,
        key=lambda item: (
            item["balanced_accuracy"],
            item["f1"],
            item["accuracy"],
        ),
    )

    print("\n===== BEST THRESHOLD ON THIS DATASET =====")
    print(f"Candidate threshold: {best['threshold']:.6f}")
    print(f"Accuracy:            {best['accuracy']:.3f}")
    print(f"Precision:           {best['precision']:.3f}")
    print(f"Recall:              {best['recall']:.3f}")
    print(f"Specificity:          {best['specificity']:.3f}")
    print(
        f"Balanced accuracy:   "
        f"{best['balanced_accuracy']:.3f}"
    )
    print(f"F1 score:             {best['f1']:.3f}")
    print(
        f"TP={best['tp']}, TN={best['tn']}, "
        f"FP={best['fp']}, FN={best['fn']}"
    )

    print(
        "\nIMPORTANT: This candidate is optimized only for your "
        "current evaluation examples. It has NOT been applied "
        "to mismatch_detector.py and is not a validated threshold."
    )
    print(
        "Your current dataset has very few examples, all using the "
        "same image. Add diverse images and independently label "
        "their captions before selecting a production threshold."
    )


if __name__ == "__main__":
    main()
