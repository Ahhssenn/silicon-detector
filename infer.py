import argparse
import os
from model import SiliconClassifier


def main():
    parser = argparse.ArgumentParser(description="Run silicon / non-silicon classification on a single image.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--model", default="silicon_model.pth", help="Path to the trained model (.pth).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold for silicon decision.")

    args = parser.parse_args()

    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Image file not found: {args.image}")

    clf = SiliconClassifier(model_path=args.model, threshold=args.threshold)

    label, confidence = clf.predict(args.image)
    is_silicon = clf.is_silicon(args.image)

    print(f"Image:      {args.image}")
    print(f"Model:      {args.model}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.3f}")
    print(f"Is silicon: {is_silicon}")


if __name__ == "__main__":
    main()
