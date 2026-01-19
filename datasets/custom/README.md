# Custom YOLO dataset (place your data here)

This folder is a template for your own dataset in **Ultralytics YOLO** format.

## Expected structure

```
datasets/custom/
  images/
    train/
    val/
  labels/
    train/
    val/
```

## Label format (YOLO)

For each image file:
- `images/train/abc.jpg`
- there should be a matching label file: `labels/train/abc.txt`

Each line in `*.txt` is one object:

```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are **normalized** to `[0, 1]` relative to the image width/height.

Example (one box of class 0):

```
0 0.50 0.50 0.20 0.10
```

## Update your classes

Edit the repo root `data.yaml`:
- change `names:` to your real class names
- keep indices aligned with the `<class_id>` used in label files

## Tip

Avoid committing large datasets into Git unless you’re using Git LFS.
