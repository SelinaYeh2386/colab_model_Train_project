import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# 1. 模型定義 (針對 28x28 的手寫數字圖片)
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10) # 輸出 0~9 共 10 類
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 準備現有的 MNIST 資料集
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    # 下載到我們建立的 data 資料夾
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    model = MyModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    checkpoint_dir = 'models'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"--- 開始訓練 (使用設備: {device}) ---")
    
    for epoch in range(1, 6): # 跑 5 輪示範
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # 3. 自動儲存權重
        checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pth')
        torch.save({'model_state_dict': model.state_dict()}, checkpoint_path)
        print(f"Epoch {epoch} 完成! 平均 Loss: {running_loss/len(train_loader):.4f}, 存檔成功。")

if __name__ == "__main__":
    main()
