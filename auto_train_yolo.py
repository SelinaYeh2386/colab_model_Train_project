from ultralytics import YOLO

# 1. 載入模型（使用 COCO 上預訓練的 yolov8n）
model = YOLO('yolov8n.pt')

# 2. 開始訓練你自己的資料集
model.train(
    data='dataset.yaml',   
    epochs=100,              # 訓練週期數
    imgsz=640,               # 訓練圖片尺寸
    batch=8,                 # 每批的圖片數，依照你的顯卡調整
    name='yolov8-ui-custom'  # 存模型的資料夾名稱
)
