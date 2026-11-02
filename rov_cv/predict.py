import argparse
import json

from .inference import VisionPipeline


def main():
    parser = argparse.ArgumentParser(description="Single-image underwater damage inference")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--detection", help="Detection checkpoint")
    parser.add_argument("--segmentation", help="Crack segmentation checkpoint")
    parser.add_argument("--enhancement", help="Optional enhancement checkpoint")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--score-threshold", type=float)
    parser.add_argument("--mask-threshold", type=float)
    args = parser.parse_args()
    pipeline = VisionPipeline(args.detection, args.segmentation, args.enhancement, args.device)
    result = pipeline.predict(args.image, args.output, args.score_threshold, args.mask_threshold)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
