"""
ProofLens AI - Multimodal Mismatch Detection

Features:
1. Image-caption comparison using pretrained CLIP.
2. Video-caption comparison using sampled video frames.
3. OCR-guided text masking for image analysis.

Scores are raw cosine similarities, NOT probabilities.
A mismatch signal does not prove that content is misinformation.
"""

from pathlib import Path
from typing import Any

import cv2
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, UnidentifiedImageError
from transformers import CLIPModel, CLIPProcessor


MODEL_ID = "openai/clip-vit-base-patch32"

# Provisional thresholds. These are not independently calibrated.
DEFAULT_MISMATCH_THRESHOLD = 0.27
DEFAULT_VIDEO_THRESHOLD = 0.33


class MismatchDetector:
    """CLIP-based image-caption and video-caption detector."""

    def __init__(
        self,
        threshold: float = DEFAULT_MISMATCH_THRESHOLD,
        video_threshold: float = DEFAULT_VIDEO_THRESHOLD,
    ) -> None:
        if not -1.0 <= threshold <= 1.0:
            raise ValueError(
                "Image threshold must be between -1 and 1."
            )

        if not -1.0 <= video_threshold <= 1.0:
            raise ValueError(
                "Video threshold must be between -1 and 1."
            )

        self.threshold = threshold
        self.video_threshold = video_threshold
        self.device = torch.device("cpu")

        print("Loading CLIP processor...")
        self.processor = CLIPProcessor.from_pretrained(MODEL_ID)

        print("Loading CLIP model...")
        self.model = CLIPModel.from_pretrained(MODEL_ID)
        self.model.to(self.device)
        self.model.eval()

        print("CLIP mismatch detector ready.")
        print(f"Device: {self.device}")
        print(f"Image threshold: {self.threshold}")
        print(f"Video threshold: {self.video_threshold}")

    # =========================================================
    # 1. CORE CLIP IMAGE-CAPTION COMPARISON
    # =========================================================

    def analyze_pil_image(
        self,
        image: Image.Image,
        caption: str,
    ) -> dict[str, Any]:
        """Calculate cosine similarity between an image and caption."""

        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL Image.")

        if not isinstance(caption, str) or not caption.strip():
            raise ValueError("Caption must not be empty.")

        image = image.convert("RGB")
        caption = caption.strip()

        inputs = self.processor(
            images=image,
            text=[caption],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = self.model(**inputs)

            image_embedding = F.normalize(
                outputs.image_embeds,
                p=2,
                dim=-1,
            )

            text_embedding = F.normalize(
                outputs.text_embeds,
                p=2,
                dim=-1,
            )

            similarity = F.cosine_similarity(
                image_embedding,
                text_embedding,
                dim=-1,
            ).item()

        decision = (
            "possible_mismatch"
            if similarity < self.threshold
            else "no_mismatch_signal"
        )

        if decision == "possible_mismatch":
            observation = (
                "Image-caption similarity is below the provisional "
                "threshold. Review the content for a possible mismatch."
            )
        else:
            observation = (
                "Similarity is at or above the provisional threshold. "
                "This does not prove that the caption is accurate."
            )

        return {
            "module": "mismatch",
            "status": "ok",
            "evidence_type": "model_prediction",
            "decision": decision,
            "score": round(float(similarity), 4),
            "score_type": "raw_cosine_similarity",
            "threshold": self.threshold,
            "threshold_is_calibrated": False,
            "observation": observation,
            "method": (
                "CLIP ViT-B/32 image-text cosine similarity"
            ),
            "limitations": [
                "The threshold is provisional and uncalibrated.",
                "The score is not a probability of misinformation.",
                "CLIP can struggle with negation and fine details.",
                "Similarity does not establish factual truth.",
            ],
        }

    # =========================================================
    # 2. OCR-BASED TEXT MASKING
    # =========================================================

    @staticmethod
    def mask_ocr_regions(
        image: Image.Image,
        ocr_result: dict[str, Any],
    ) -> tuple[Image.Image, dict[str, Any]]:
        """
        Mask detected text regions using neutral gray.

        OCR lines are expected to have the following structure:

        {
            "lines": [
                {
                    "text": "example",
                    "bounding_box": [
                        [x1, y1],
                        [x2, y2],
                        [x3, y3],
                        [x4, y4]
                    ]
                }
            ]
        }
        """

        if not isinstance(ocr_result, dict):
            raise TypeError("ocr_result must be a dictionary.")

        masked_image = image.convert("RGB").copy()
        width, height = masked_image.size
        draw = ImageDraw.Draw(masked_image)

        lines = ocr_result.get("lines", [])

        if not isinstance(lines, (list, tuple)):
            lines = []

        boxes = []

        for line in lines:
            if not isinstance(line, dict):
                continue

            points = line.get("bounding_box")

            if not isinstance(points, (list, tuple)):
                continue

            if len(points) < 4:
                continue

            try:
                xs = [float(point[0]) for point in points]
                ys = [float(point[1]) for point in points]
            except (TypeError, ValueError, IndexError):
                continue

            x1 = max(0, int(min(xs)))
            y1 = max(0, int(min(ys)))
            x2 = min(width, int(max(xs)) + 1)
            y2 = min(height, int(max(ys)) + 1)

            # Expand the regions slightly to cover text edges.
            padding_x = max(3, int((x2 - x1) * 0.04))
            padding_y = max(3, int((y2 - y1) * 0.12))

            x1 = max(0, x1 - padding_x)
            y1 = max(0, y1 - padding_y)
            x2 = min(width, x2 + padding_x)
            y2 = min(height, y2 + padding_y)

            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])

        # Merge overlapping regions to prevent redundant masks.
        merged_boxes = []

        for new_box in boxes:
            current = new_box[:]
            index = 0

            while index < len(merged_boxes):
                other = merged_boxes[index]

                overlaps = not (
                    current[2] < other[0]
                    or current[0] > other[2]
                    or current[3] < other[1]
                    or current[1] > other[3]
                )

                if overlaps:
                    current = [
                        min(current[0], other[0]),
                        min(current[1], other[1]),
                        max(current[2], other[2]),
                        max(current[3], other[3]),
                    ]

                    merged_boxes.pop(index)

                    # Restart because the merged box may overlap
                    # another previously separate region.
                    index = 0
                else:
                    index += 1

            merged_boxes.append(current)

        # Mask the final regions.
        for x1, y1, x2, y2 in merged_boxes:
            draw.rectangle(
                (x1, y1, x2, y2),
                fill=(127, 127, 127),
            )

        metadata = {
            "ocr_boxes_received": len(lines),
            "valid_boxes_used": len(boxes),
            "merged_mask_regions": len(merged_boxes),
            "masked_regions": merged_boxes,
            "mask_fill_rgb": [127, 127, 127],
        }

        return masked_image, metadata

    # =========================================================
    # 3. COMPLETE IMAGE ANALYSIS
    # =========================================================

    def analyze_image(
        self,
        image_path: str | Path,
        caption: str,
        ocr_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Analyze an image and optionally run OCR-guided masking.

        When the caption matches the extracted OCR text, use the
        masked-image score as the primary decision score. This helps
        reduce the influence of text that appears directly in the image.

        For an independent user-provided caption, retain the original
        image score as the primary score and report masking separately.
        """

        path = Path(image_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Image file not found: {path}"
            )

        if not isinstance(caption, str) or not caption.strip():
            raise ValueError("Caption must not be empty.")

        try:
            with Image.open(path) as source:
                image = source.convert("RGB")

        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(
                f"Could not read image file: {path.name}"
            ) from exc

        # Always preserve the original image analysis.
        result = self.analyze_pil_image(image, caption)

        result["media_type"] = "image"
        result["source"] = path.name

        # Without OCR, the original score is the primary score.
        if ocr_result is None:
            result["caption_source"] = "user_provided"
            result["decision_basis"] = "original_image"

            result["text_mask_experiment"] = {
                "status": "skipped",
                "reason": "No OCR result was provided.",
            }

            return result

        # Collect OCR text from the dictionary.
        ocr_text = ocr_result.get("extracted_text", "")

        if not isinstance(ocr_text, str) or not ocr_text.strip():
            lines = ocr_result.get("lines", [])
            extracted_parts = []

            if isinstance(lines, (list, tuple)):
                for line in lines:
                    if not isinstance(line, dict):
                        continue

                    line_text = line.get("text", "")

                    if (
                        isinstance(line_text, str)
                        and line_text.strip()
                    ):
                        extracted_parts.append(line_text.strip())

            ocr_text = "\n".join(extracted_parts)

        # Normalize whitespace and capitalization before comparing.
        def normalize_text(value: str) -> str:
            return " ".join(value.casefold().split())

        caption_is_ocr_text = (
            bool(ocr_text.strip())
            and normalize_text(caption) == normalize_text(ocr_text)
        )

        result["caption_source"] = (
            "ocr_extracted_text"
            if caption_is_ocr_text
            else "user_provided_or_corrected"
        )

        # Create a masked copy of the image.
        masked_image, metadata = self.mask_ocr_regions(
            image,
            ocr_result,
        )

        # If no valid bounding boxes were found, do not invent
        # a masked score or override the original result.
        if metadata["valid_boxes_used"] == 0:
            result["text_mask_experiment"] = {
                "status": "skipped",
                "reason": "OCR supplied no usable bounding boxes.",
                **metadata,
            }

            result["decision_basis"] = "original_image"
            return result

        # Compare the masked image against the same caption.
        masked_result = self.analyze_pil_image(
            masked_image,
            caption,
        )

        original_score = result["score"]
        original_decision = result["decision"]

        masked_score = masked_result["score"]
        masked_decision = masked_result["decision"]

        # Retain both scores regardless of which decision is used.
        result["text_mask_experiment"] = {
            "status": "completed",
            "original_score": original_score,
            "original_decision": original_decision,
            "masked_score": masked_score,
            "masked_decision": masked_decision,
            "score_difference_masked_minus_original": round(
                masked_score - original_score,
                4,
            ),
            **metadata,
            "interpretation": (
                "Masking provides an additional visual comparison. "
                "It may remove useful image content. The score change "
                "does not prove whether a caption is true or false."
            ),
        }

        if caption_is_ocr_text:
            # The caption came directly from text printed on the image.
            # Do not let that same printed text validate itself.
            result["unmasked_score"] = original_score
            result["unmasked_decision"] = original_decision

            result["score"] = masked_score

            result["score_type"] = (
                "raw_cosine_similarity_on_ocr_masked_image"
            )

            result["decision"] = masked_decision

            result["decision_basis"] = (
                "ocr_masked_image_vs_extracted_text"
            )

            result["observation"] = (
                "The caption was extracted from the image. "
                "The primary decision uses the image with OCR text "
                "regions masked to reduce matching driven by the "
                "printed text itself. Review both original and masked "
                "scores. The threshold is provisional."
            )

            result["limitations"].append(
                "The threshold has not been independently calibrated "
                "for OCR-masked images."
            )

        else:
            # Independent or corrected caption:
            # retain the original-image score as the primary result.
            result["decision_basis"] = (
                "original_image_vs_user_caption"
            )

        return result

    # =========================================================
    # 4. VIDEO ANALYSIS
    # =========================================================

    def analyze_video(
        self,
        video_path: str | Path,
        caption: str,
        num_frames: int = 5,
    ) -> dict[str, Any]:
        """Compare a caption against evenly sampled video frames."""

        path = Path(video_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Video file not found: {path}"
            )

        if not isinstance(caption, str) or not caption.strip():
            raise ValueError("Caption must not be empty.")

        if (
            not isinstance(num_frames, int)
            or isinstance(num_frames, bool)
        ):
            raise ValueError("num_frames must be an integer.")

        if not 1 <= num_frames <= 30:
            raise ValueError(
                "num_frames must be between 1 and 30."
            )

        capture = cv2.VideoCapture(str(path))

        try:
            if not capture.isOpened():
                raise ValueError(
                    f"Could not open video: {path.name}. "
                    "Check the video file and codec."
                )

            total_frames = int(
                capture.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            fps = float(
                capture.get(cv2.CAP_PROP_FPS)
            )

            if total_frames <= 0:
                raise ValueError(
                    "The video has no readable frames."
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

            frame_results = []

            for frame_index in frame_indices:
                capture.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    frame_index,
                )

                success, frame = capture.read()

                # Skip failed frame reads, and report how many
                # frames actually succeeded in the final result.
                if not success or frame is None:
                    continue

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                pil_frame = Image.fromarray(rgb_frame)

                frame_result = self.analyze_pil_image(
                    pil_frame,
                    caption,
                )

                score = float(frame_result["score"])

                frame_decision = (
                    "possible_mismatch"
                    if score < self.video_threshold
                    else "no_mismatch_signal"
                )

                timestamp = (
                    frame_index / fps
                    if fps > 0
                    else None
                )

                frame_results.append({
                    "frame_index": frame_index,
                    "timestamp_seconds": (
                        round(timestamp, 2)
                        if timestamp is not None
                        else None
                    ),
                    "score": round(score, 4),
                    "decision": frame_decision,
                })

        finally:
            capture.release()

        if not frame_results:
            raise ValueError(
                "No video frames could be analyzed successfully."
            )

        scores = [
            item["score"]
            for item in frame_results
        ]

        analyzed_count = len(frame_results)

        mismatch_count = sum(
            score < self.video_threshold
            for score in scores
        )

        mismatch_fraction = (
            mismatch_count / analyzed_count
        )

        mean_similarity = sum(scores) / analyzed_count

        # Use the mean score as the primary video decision rule.
        possible_mismatch = (
            mean_similarity < self.video_threshold
        )

        return {
            "module": "mismatch",
            "status": "ok",
            "evidence_type": "model_prediction",
            "media_type": "video",
            "source": path.name,
            "decision": (
                "possible_mismatch"
                if possible_mismatch
                else "no_mismatch_signal"
            ),
            "score": round(mean_similarity, 4),
            "score_type": (
                "mean_raw_cosine_similarity_across_frames"
            ),
            "threshold": self.video_threshold,
            "image_threshold": self.threshold,
            "threshold_is_calibrated": False,
            "decision_rule": (
                "possible_mismatch when mean similarity across "
                "analyzed frames is below the provisional video threshold"
            ),
            "requested_frame_count": num_frames,
            "sampled_frame_count": analyzed_count,
            "total_video_frames": total_frames,
            "mismatch_frame_count": mismatch_count,
            "mismatch_frame_fraction": round(
                mismatch_fraction,
                4,
            ),
            "frame_results": frame_results,
            "observation": (
                f"Mean similarity: {mean_similarity:.4f}; "
                f"video threshold: {self.video_threshold:.2f}. "
                f"{mismatch_count} of {analyzed_count} analyzed "
                "frames fell below the threshold."
            ),
            "method": (
                "CLIP image-text similarity on evenly sampled "
                "video frames, aggregated by mean similarity"
            ),
            "limitations": [
                "Only successfully decoded sampled frames are analyzed.",
                "Video audio is not analyzed.",
                "Temporal event sequencing is not analyzed.",
                "The video threshold is provisional and uncalibrated.",
                "Important scenes may occur between sampled frames.",
                "A mismatch signal does not prove misinformation.",
            ],
        }


# =============================================================
# 5. STANDALONE COMMAND-LINE TEST
# =============================================================

def main() -> None:
    """Run a standalone mismatch-detector test."""

    print("\n===== PROOFLENS MISMATCH DETECTOR =====")
    print("1. Analyze an image")
    print("2. Analyze a video")

    media_choice = input(
        "Choose media type (1 or 2): "
    ).strip()

    if media_choice not in {"1", "2"}:
        print("Input error: choose 1 or 2.")
        return

    media_text = input(
        "Enter the full path to your media file: "
    ).strip().strip('"')

    if not media_text:
        print("Input error: media path is required.")
        return

    media_path = Path(media_text)

    if not media_path.is_file():
        print(
            f"Input error: file not found: {media_path}"
        )
        return

    caption = input(
        "Enter the caption to compare against the media: "
    ).strip()

    if not caption:
        print("Input error: caption is required.")
        return

    try:
        detector = MismatchDetector()

        if media_choice == "1":
            result = detector.analyze_image(
                image_path=media_path,
                caption=caption,
            )

        else:
            frame_text = input(
                "Number of sampled frames (1-30, default 5): "
            ).strip()

            try:
                num_frames = (
                    int(frame_text)
                    if frame_text
                    else 5
                )
            except ValueError:
                print(
                    "Input error: frame count must be a whole number."
                )
                return

            result = detector.analyze_video(
                video_path=media_path,
                caption=caption,
                num_frames=num_frames,
            )

        print("\n===== PROOFLENS ANALYSIS REPORT =====")

        for key, value in result.items():
            print(f"{key}: {value}")

    except (
        ValueError,
        FileNotFoundError,
        TypeError,
    ) as exc:
        print(
            f"\nInput or analysis error: "
            f"{type(exc).__name__}: {exc}"
        )

    except Exception as exc:
        print(
            f"\nUnexpected error: "
            f"{type(exc).__name__}: {exc}"
        )
        print(
            "Send the complete terminal error for troubleshooting."
        )


if __name__ == "__main__":
    main()