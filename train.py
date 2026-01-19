from ultralytics import YOLO

import argparse
import datetime as _dt
import os


def _default_run_name(prefix: str = "my_exp") -> str:
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLOv8 model using Ultralytics")
    parser.add_argument(
        "--data",
        default="data.yaml",
        help="Dataset YAML (default: data.yaml)",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Base model/weights to start from (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs (default: 5)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size (default: 640)",
    )
    parser.add_argument(
        "--project",
        default="models",
        help="Output project directory (default: models)",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Run name under --project (default: auto timestamp)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for training, e.g. '0', 'cpu'. Default lets Ultralytics decide.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve data yaml fallback to ultralytics' coco8.yaml if user didn't provide one
    data_yaml = args.data
    if data_yaml == "data.yaml" and not os.path.exists("data.yaml"):
        data_yaml = "coco8.yaml"

    run_name = args.name or _default_run_name("my_exp")

    model = YOLO(args.model)

    train_kwargs = dict(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        save=True,
        project=args.project,
        name=run_name,
    )
    if args.device:
        train_kwargs["device"] = args.device

    model.train(**train_kwargs)

    print(f"--- 訓練完成！權重已儲存於 {args.project}/{run_name}/weights/ ---")


if __name__ == "__main__":
    main()
