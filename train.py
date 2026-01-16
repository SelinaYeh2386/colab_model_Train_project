from ultralytics import YOLO
import os

def main():
    # 1. 載入預訓練模型 (YOLOv8 奈米版，速度最快)
    model = YOLO('yolov8n.pt') 

    # 2. 開始訓練
    # 我們直接使用官方的 'coco8.yaml'，它會自動下載圖片與標註檔
    # coco8 包含 80 種常見物品（人、車、狗等）的標註
    results = model.train(
        data='coco8.yaml', 
        epochs=5,          # 先跑 5 輪測試流程
        imgsz=640,         # 圖片縮放大小
        save=True,          # 自動儲存權重 (.pt 檔)
        project='models',   # 儲存到我們建立的 models 資料夾
        name='my_exp'       # 這次實驗的名字
    )
    
    print("--- 訓練完成！權重已儲存於 models/my_exp/weights/ ---")

if __name__ == "__main__":
    main()
