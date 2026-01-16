import torch
import torch.nn as nn
import torch.optim as optim
import os

# 1. 模型定義
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.network(x)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 設定儲存路徑 (建議指向 Google Drive 掛載後的路徑，例如 /content/drive/MyDrive/models)
    # 這裡先設在專案內的 models/
    checkpoint_dir = 'models'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print("--- 開始訓練 ---")
    
    # 模擬訓練 10 個 Epoch
    for epoch in range(1, 11):
        # --- 這裡原本會放訓練程式碼 (訓練過程略過) ---
        
        # 2. 自動儲存邏輯
        checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pth')
        
        # 儲存內容：除了權重(state_dict)，通常還會存當前的 epoch 和 optimizer 狀態
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        
        print(f"Epoch {epoch} 完成，權重已自動儲存至: {checkpoint_path}")

if __name__ == "__main__":
    main()
