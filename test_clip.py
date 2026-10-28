from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from transformers import CLIPModel, CLIPProcessor


MODEL_ID = "openai/clip-vit-base-patch32"


def main():
    # Ask for an image on your computer.
    image_path = Path(input("Enter the full image file path: ").strip().strip('"'))

    if not image_path.is_file():
        print("ERROR: Image file not found.")
        return

    try:
        image = Image.open(image_path).convert("RGB")
    except (OSError, UnidentifiedImageError) as error:
        print(f"ERROR: Could not open image: {error}")
        return

    matching_caption = input(
        "Enter a caption that correctly describes the image: "
    ).strip()

    mismatching_caption = input(
        "Enter a clearly unrelated caption: "
    ).strip()

    if not matching_caption or not mismatching_caption:
        print("ERROR: Both captions must contain text.")
        return

    print("\nLoading the cached CLIP model...")

    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    model = CLIPModel.from_pretrained(MODEL_ID)
    model.eval()

    captions = [matching_caption, mismatching_caption]

    inputs = processor(
        text=captions,
        images=image,
        return_tensors="pt",
        padding=True,
    )

    print("Comparing image and captions...")

    with torch.inference_mode():
        outputs = model(**inputs)
        scores = outputs.logits_per_image.softmax(dim=1)[0]

    print("\n===== CLIP TEST RESULTS =====")
    print(f"Image: {image_path.name}")
    print(f"Matching caption: {matching_caption}")
    print(f"Relative score: {scores[0].item():.4f}")
    print()
    print(f"Unrelated caption: {mismatching_caption}")
    print(f"Relative score: {scores[1].item():.4f}")

    if scores[0] > scores[1]:
        print("\nResult: CLIP prefers the matching caption.")
    elif scores[1] > scores[0]:
        print("\nResult: CLIP prefers the unrelated caption.")
    else:
        print("\nResult: Both captions received equal scores.")

    print(
        "\nNOTE: These scores compare only the two supplied captions. "
        "They are not calibrated probabilities and do not prove whether "
        "a post is misinformation."
    )


if __name__ == "__main__":
    main()