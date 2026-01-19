from ultralytics import YOLO
import os

def main():
    # 1. 載入預訓練模型 (YOLOv8 奈米版，速度最快)
    model = YOLO('yolov8n.pt') 

    # 2. 選擇資料集設定
    # - 若專案根目錄有 data.yaml，就用它（較適合在 Colab / 本機使用本 repo 的 datasets/）
    # - 否則使用 ultralytics 內建的 coco8.yaml（會自動下載）
    data_yaml = 'data.yaml' if os.path.exists('data.yaml') else 'coco8.yaml'

    # 3. 開始訓練
    results = model.train(
        data=data_yaml,
        epochs=5,          # 先跑 5 輪測試流程
        imgsz=640,         # 圖片縮放大小
        save=True,          # 自動儲存權重 (.pt 檔)
        project='models',   # 儲存到我們建立的 models 資料夾
        name='my_exp'       # 這次實驗的名字
    )
    
    print("--- 訓練完成！權重已儲存於 models/my_exp/weights/ ---")

if __name__ == "__main__":
    main()
