"""
ProofLens Multimodal Analyzer

OCR extraction and CLIP-based image/video caption comparison.

Uses OCRTextExtractor.extract_from_image(path), the actual method
available in this project's ocr_extractor.py.

OCR failure or absence of visible text does not prevent the
user from supplying a caption and running mismatch analysis.
"""

from pathlib import Path
from typing import Any
import tempfile

import cv2

from mismatch_detector import MismatchDetector
from ocr_extractor import OCRTextExtractor


def empty_ocr_result(
    source: str,
    status: str,
    message: str = "",
) -> dict[str, Any]:
    """Create a consistent OCR result when no text is available."""

    result = {
        "module": "ocr",
        "status": status,
        "source": source,
        "has_text": False,
        "extracted_text": "",
        "lines": [],
        "needs_user_confirmation": True,
    }

    if message:
        result["observation"] = message

    return result


def get_ocr_text(ocr_result: dict[str, Any]) -> str:
    """Get combined extracted text from the OCR result."""

    extracted = ocr_result.get("extracted_text", "")

    if isinstance(extracted, str) and extracted.strip():
        return extracted.strip()

    parts = []
    lines = ocr_result.get("lines", [])

    if isinstance(lines, (list, tuple)):
        for line in lines:
            if not isinstance(line, dict):
                continue

            text = line.get("text", "")

            if isinstance(text, str) and text.strip():
                parts.append(text.strip())

    return "\n".join(parts)


def normalize_ocr_result(
    result: Any,
    source: str,
) -> dict[str, Any]:
    """Normalize the dictionary returned by the existing OCR extractor."""

    if not isinstance(result, dict):
        return empty_ocr_result(
            source,
            "error",
            "OCR returned an unexpected result. Caption analysis can continue.",
        )

    normalized = dict(result)
    normalized.setdefault("module", "ocr")
    normalized.setdefault("source", source)

    lines = normalized.get("lines", [])

    if not isinstance(lines, (list, tuple)):
        lines = []

    normalized["lines"] = list(lines)

    extracted_text = get_ocr_text(normalized)

    normalized["extracted_text"] = extracted_text
    normalized["has_text"] = bool(extracted_text.strip())

    if "status" not in normalized:
        normalized["status"] = (
            "text_detected"
            if extracted_text.strip()
            else "no_text_detected"
        )

    normalized.setdefault("needs_user_confirmation", True)

    return normalized


def extract_image_ocr(
    extractor: OCRTextExtractor | None,
    image_path: Path,
) -> dict[str, Any]:
    """Extract OCR from an image using the project's real OCR method."""

    if extractor is None:
        return empty_ocr_result(
            image_path.name,
            "unavailable",
            "OCR could not initialize. Enter a caption manually.",
        )

    try:
        # This is the confirmed method in ocr_extractor.py.
        result = extractor.extract_from_image(str(image_path))

        return normalize_ocr_result(
            result,
            image_path.name,
        )

    except Exception as exc:
        return empty_ocr_result(
            image_path.name,
            "error",
            (
                f"OCR failed: {type(exc).__name__}: {exc}. "
                "Caption-based mismatch analysis can continue."
            ),
        )


def extract_video_ocr(
    extractor: OCRTextExtractor | None,
    video_path: Path,
    num_frames: int,
) -> dict[str, Any]:
    """Extract visible text from evenly sampled video frames."""

    if extractor is None:
        return empty_ocr_result(
            video_path.name,
            "unavailable",
            "OCR could not initialize. Enter a caption manually.",
        )

    capture = cv2.VideoCapture(str(video_path))

    all_lines = []
    text_parts = []
    seen_text = set()
    frames_scanned = 0
    frames_with_text = 0
    errors = []

    try:
        if not capture.isOpened():
            return empty_ocr_result(
                video_path.name,
                "error",
                (
                    "Could not open the video for OCR. "
                    "Caption-based analysis will still be attempted."
                ),
            )

        total_frames = int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        fps = float(capture.get(cv2.CAP_PROP_FPS))

        if total_frames <= 0:
            return empty_ocr_result(
                video_path.name,
                "no_frames",
                (
                    "No readable video frames were available for OCR. "
                    "Caption-based analysis will still be attempted."
                ),
            )

        sample_count = min(num_frames, total_frames)

        if sample_count == 1:
            frame_indices = [0]
        else:
            frame_indices = [
                round(
                    index * (total_frames - 1)
                    / (sample_count - 1)
                )
                for index in range(sample_count)
            ]

        for frame_index in frame_indices:
            capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_index,
            )

            success, frame = capture.read()

            if not success or frame is None:
                continue

            frames_scanned += 1

            timestamp = (
                frame_index / fps if fps > 0 else None
            )

            temporary_path = None

            try:
                # The existing OCR wrapper accepts an image file path.
                with tempfile.NamedTemporaryFile(
                    suffix=".jpg",
                    prefix="prooflens_frame_",
                    delete=False,
                ) as temp_file:
                    temporary_path = Path(temp_file.name)

                if not cv2.imwrite(str(temporary_path), frame):
                    raise OSError(
                        "Could not save a temporary frame for OCR."
                    )

                frame_result = extractor.extract_from_image(
                    str(temporary_path)
                )

                frame_result = normalize_ocr_result(
                    frame_result,
                    f"{video_path.name}, frame {frame_index}",
                )

                frame_text = get_ocr_text(frame_result)

                if frame_text:
                    frames_with_text += 1

                    for text_line in frame_text.splitlines():
                        clean_line = text_line.strip()

                        if not clean_line:
                            continue

                        key = " ".join(clean_line.lower().split())

                        if key not in seen_text:
                            text_parts.append(clean_line)
                            seen_text.add(key)

                lines = frame_result.get("lines", [])

                for original_line in lines:
                    if not isinstance(original_line, dict):
                        continue

                    line = dict(original_line)
                    line["frame_index"] = frame_index

                    if timestamp is not None:
                        line["timestamp_seconds"] = round(
                            timestamp,
                            2,
                        )

                    all_lines.append(line)

            except Exception as exc:
                errors.append(
                    f"Frame {frame_index}: "
                    f"{type(exc).__name__}: {exc}"
                )

            finally:
                if temporary_path is not None:
                    try:
                        temporary_path.unlink(missing_ok=True)
                    except OSError:
                        pass

    finally:
        capture.release()

    combined_text = "\n".join(text_parts)

    if combined_text:
        status = "text_detected"
    elif frames_scanned:
        status = "no_text_detected"
    else:
        status = "no_frames_read"

    result = {
        "module": "ocr",
        "status": status,
        "source": video_path.name,
        "has_text": bool(combined_text),
        "extracted_text": combined_text,
        "lines": all_lines,
        "frames_scanned_for_ocr": frames_scanned,
        "frames_with_text": frames_with_text,
        "needs_user_confirmation": True,
        "observation": (
            f"OCR scanned {frames_scanned} sampled frame(s); "
            f"text was found in {frames_with_text} frame(s)."
        ),
    }

    if errors:
        result["ocr_frame_errors"] = errors

    return result


def ask_caption(
    ocr_result: dict[str, Any],
    media_label: str,
) -> str:
    """Use OCR text by default when present; otherwise ask for a caption."""

    extracted_text = get_ocr_text(ocr_result)

    print("\n========== CAPTION INPUT ==========")

    if extracted_text:
        print("Text extracted by OCR:")
        print(extracted_text)

        print(
            "\nPress Enter to compare using the extracted text, "
            "or enter an independent caption."
        )

        entered_caption = input("Caption: ").strip()

        caption = (
            entered_caption
            if entered_caption
            else extracted_text
        )

    else:
        print("No readable on-screen text was detected.")
        print(
            "Mismatch analysis can still run using a caption you provide."
        )

        caption = input(
            f"Enter a caption describing the {media_label}: "
        ).strip()

    return caption


def print_report(
    ocr_result: dict[str, Any] | None,
    mismatch_result: dict[str, Any],
) -> None:
    """Print the combined OCR and mismatch results."""

    print("\n========== PROOFLENS REPORT ==========")

    if ocr_result is not None:
        print("\n--- OCR ---")

        for key, value in ocr_result.items():
            print(f"{key}: {value}")

    print("\n--- MISMATCH ANALYSIS ---")

    for key, value in mismatch_result.items():
        print(f"{key}: {value}")

    print("\nReminder: a mismatch signal does not prove misinformation.")
    print("======================================")


def main() -> None:
    """Run the interactive ProofLens analyzer."""

    print("\n===== PROOFLENS MULTIMODAL ANALYZER =====")
    print("1. Image + caption")
    print("2. Video + caption")

    media_choice = input(
        "Choose media type (1 or 2): "
    ).strip()

    if media_choice not in {"1", "2"}:
        print("Input error: choose 1 or 2.")
        return

    path_text = input(
        "Enter the full path to your image or video: "
    ).strip().strip('"')

    if not path_text:
        print("Input error: media path is required.")
        return

    path = Path(path_text)

    if not path.is_file():
        print(f"Input error: file not found: {path}")
        return

    try:
        # OCR initialization failure must not prevent manual-caption analysis.
        try:
            print("\nInitializing OCR extractor...")
            extractor = OCRTextExtractor()

        except Exception as exc:
            extractor = None

            print(
                f"Warning: OCR initialization failed: "
                f"{type(exc).__name__}: {exc}"
            )
            print(
                "You can still provide a caption and run mismatch analysis."
            )

        num_frames = 5

        if media_choice == "1":
            ocr_result = extract_image_ocr(
                extractor,
                path,
            )

            extracted_text = get_ocr_text(ocr_result)

            print("\n===== TEXT EXTRACTED BY OCR =====")
            print(
                extracted_text
                if extracted_text
                else "No text detected. Caption input will be requested."
            )
            print("================================")

            caption = ask_caption(
                ocr_result,
                "image",
            )

        else:
            frame_text = input(
                "Number of sampled frames (1-30, default 5): "
            ).strip()

            try:
                num_frames = int(frame_text) if frame_text else 5

            except ValueError:
                print(
                    "Input error: frame count must be a whole number."
                )
                return

            if not 1 <= num_frames <= 30:
                print(
                    "Input error: frame count must be between 1 and 30."
                )
                return

            print("\nExtracting text from sampled video frames...")

            ocr_result = extract_video_ocr(
                extractor,
                path,
                num_frames,
            )

            extracted_text = get_ocr_text(ocr_result)

            print("\n===== TEXT EXTRACTED FROM VIDEO =====")
            print(
                extracted_text
                if extracted_text
                else "No text detected. Caption input will be requested."
            )
            print("======================================")

            caption = ask_caption(
                ocr_result,
                "video",
            )

        # Continue even if OCR detected no text. A caption is now required.
        if not caption.strip():
            print(
                "Input error: enter a non-empty caption to run mismatch analysis."
            )
            return

        print("\nInitializing mismatch detector...")
        detector = MismatchDetector()

        if media_choice == "1":
            mismatch_result = detector.analyze_image(
                image_path=path,
                caption=caption,
                ocr_result=ocr_result,
            )

        else:
            mismatch_result = detector.analyze_video(
                video_path=path,
                caption=caption,
                num_frames=num_frames,
            )

            # Include the video OCR results in the final report.
            mismatch_result["ocr_text_scan"] = {
                "status": ocr_result.get("status", "unknown"),
                "has_text": ocr_result.get("has_text", False),
                "frames_scanned_for_ocr": ocr_result.get(
                    "frames_scanned_for_ocr",
                    0,
                ),
                "frames_with_text": ocr_result.get(
                    "frames_with_text",
                    0,
                ),
                "extracted_text": get_ocr_text(ocr_result),
                "note": (
                    "OCR was performed on sampled video frames. "
                    "Video mismatch detection compares sampled frames "
                    "against the selected caption. This version does not "
                    "mask OCR regions from video frames."
                ),
            }

        print_report(
            ocr_result,
            mismatch_result,
        )

    except (
        ValueError,
        FileNotFoundError,
        TypeError,
        AttributeError,
    ) as exc:
        print(
            f"\nInput or integration error: "
            f"{type(exc).__name__}: {exc}"
        )

    except Exception as exc:
        print(
            f"\nUnexpected error: {type(exc).__name__}: {exc}"
        )
        print(
            "Copy the complete terminal output and send it for troubleshooting."
        )


if __name__ == "__main__":
    main()