from ultralytics import YOLO

def main():
    # 1. 載入預訓練模型 (省去從零開始訓練的時間)
    model = YOLO('yolov8n.pt') 

    # 2. 開始訓練
    # data='data.yaml' 裡會寫好你的 train/val 路徑與 .txt 類別
    model.train(
        data='data.yaml', 
        epochs=100, 
        imgsz=640,
        save=True,           # 自動儲存權重
        resume=True          # 如果中斷了，下次執行會自動從上次的地方開始
    )

if __name__ == "__main__":
    main()
