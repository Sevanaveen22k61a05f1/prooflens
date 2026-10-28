
"""
ProofLens OCR module.

Extracts visible text, confidence scores, and bounding boxes
from uploaded images using RapidOCR.
"""

from pathlib import Path
from typing import Any

from rapidocr import RapidOCR


class OCRTextExtractor:
    """Reusable OCR engine for image text extraction."""

    def __init__(self) -> None:
        print("Initializing RapidOCR...")
        self.engine = RapidOCR()
        print("OCR extractor ready.")

    def extract_from_image(
        self,
        image_path: str | Path,
    ) -> dict[str, Any]:
        """Extract visible text, confidence, and bounding boxes."""

        path = Path(image_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Image file not found: {path}"
            )

        result = self.engine(str(path))

        if result is None:
            text_lines = []
            confidence_scores = []
            detected_boxes = []
        else:
            raw_texts = getattr(result, "txts", None)
            raw_scores = getattr(result, "scores", None)
            raw_boxes = getattr(result, "boxes", None)

            # Avoid Boolean evaluation of NumPy arrays.
            text_lines = (
                list(raw_texts)
                if raw_texts is not None
                else []
            )

            confidence_scores = (
                list(raw_scores)
                if raw_scores is not None
                else []
            )

            detected_boxes = (
                list(raw_boxes)
                if raw_boxes is not None
                else []
            )

        lines = []

        for index, text in enumerate(text_lines):
            cleaned_text = str(text).strip()

            if not cleaned_text:
                continue

            confidence = None

            if index < len(confidence_scores):
                try:
                    confidence = float(
                        confidence_scores[index]
                    )
                except (TypeError, ValueError):
                    confidence = None

            # Convert each text region's four corner points
            # into ordinary Python lists of numeric coordinates.
            box = None

            if index < len(detected_boxes):
                try:
                    points = detected_boxes[index]

                    converted_box = [
                        [
                            round(float(point[0]), 2),
                            round(float(point[1]), 2),
                        ]
                        for point in points
                    ]

                    if len(converted_box) >= 4:
                        box = converted_box

                except (
                    TypeError,
                    ValueError,
                    IndexError,
                ):
                    box = None

            lines.append({
                "text": cleaned_text,
                "ocr_confidence": (
                    round(confidence, 4)
                    if confidence is not None
                    else None
                ),
                "bounding_box": box,
            })

        extracted_text = "\n".join(
            item["text"] for item in lines
        )

        if extracted_text:
            status = "text_detected"
            observation = (
                f"OCR extracted {len(lines)} text line(s). "
                "Review the extracted text before analysis."
            )
        else:
            status = "no_text_detected"
            observation = (
                "OCR did not return readable text. "
                "The user should enter a caption manually."
            )

        return {
            "module": "ocr",
            "status": status,
            "source": path.name,
            "has_text": bool(extracted_text),
            "extracted_text": extracted_text,
            "lines": lines,
            "needs_user_confirmation": bool(extracted_text),
            "observation": observation,
            "limitations": [
                "OCR can misread blurry, small, stylized, or obscured text.",
                "OCR cannot determine whether extracted statements are true.",
                "Bounding boxes can overlap or have imperfect boundaries.",
                "No detected text does not prove that an image contains no text.",
            ],
        }


def main() -> None:
    """Run a manual OCR test from the terminal."""

    print("\n===== PROOFLENS OCR TEXT EXTRACTOR =====")

    image_path = input(
        "Enter the full path to an image containing text: "
    ).strip().strip('"')

    if not image_path:
        print("Input error: image path is required.")
        return

    try:
        extractor = OCRTextExtractor()
        result = extractor.extract_from_image(image_path)

        print("\n===== OCR RESULT =====")

        for key, value in result.items():
            print(f"{key}: {value}")

        if result["has_text"]:
            print("\nExtracted text:")
            print(result["extracted_text"])

    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Input or OCR error: {exc}")

    except Exception as exc:
        print(
            f"Unexpected error: {type(exc).__name__}: {exc}"
        )
        print(
            "Send the complete terminal output for troubleshooting."
        )


if __name__ == "__main__":
    main()
