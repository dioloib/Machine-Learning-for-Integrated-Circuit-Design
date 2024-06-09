import os
import pickle
import torch
from torch_geometric.data import Data, DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from tqdm import tqdm
import matplotlib.pyplot as plt

# 数据加载和批处理函数
# 加载数据
with open('train_data_tensor_99.pkl', 'rb') as f:
    train_data_eval = pickle.load(f)

with open('updated_train_data_tensor.pkl', 'rb') as f:
    train_data_reward = pickle.load(f)

# 数据集准备
class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        x = item['x'].float()  # 转换为 Float 类型
        edge_index = item['edge_index']
        y = item['y'].float()  # 确保 y 也为 Float 类型
        return Data(x=x, edge_index=edge_index, y=y)

dataset_eval = GraphDataset(train_data_eval)
dataset_reward = GraphDataset(train_data_reward)

def collate_fn(batch):
    return batch

loader_eval = DataLoader(dataset_eval, batch_size=32, shuffle=True, collate_fn=collate_fn)
loader_reward = DataLoader(dataset_reward, batch_size=32, shuffle=True, collate_fn=collate_fn)

# 模型定义
class GNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = torch.mean(x, dim=0)  # 聚合节点特征
        x = self.fc(x)
        return x

# 模型训练和评估
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in tqdm(loader, desc="Training"):
        optimizer.zero_grad()
        data = data.to(device)
        output = model(data.x, data.edge_index)
        loss = criterion(output, data.y.view(-1, 1))  # 确保目标张量与输出张量的尺寸匹配
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in tqdm(loader, desc="Evaluating"):
            data = data.to(device)
            output = model(data.x, data.edge_index)
            loss = criterion(output, data.y.view(-1, 1))  # 确保目标张量与输出张量的尺寸匹配
            total_loss += loss.item()
    return total_loss / len(loader)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dim = train_data_eval[0]['x'].shape[1]
print(input_dim)
hidden_dim = 64
output_dim = 1

model_eval = GNNModel(input_dim, hidden_dim, output_dim).to(device)
model_reward = GNNModel(input_dim, hidden_dim, output_dim).to(device)

optimizer_eval = torch.optim.Adam(model_eval.parameters(), lr=0.01)
optimizer_reward = torch.optim.Adam(model_reward.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

num_epochs = 100
eval_losses = []
reward_losses = []

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_loss_eval = train(model_eval, loader_eval, optimizer_eval, criterion, device)
    val_loss_eval = evaluate(model_eval, loader_eval, criterion, device)
    train_loss_reward = train(model_reward, loader_reward, optimizer_reward, criterion, device)
    val_loss_reward = evaluate(model_reward, loader_reward, criterion, device)
    print(f'Eval Loss: {val_loss_eval:.4f}, Reward Loss: {val_loss_reward:.4f}')
    
    eval_losses.append(val_loss_eval)
    reward_losses.append(val_loss_reward)

torch.save(model_eval, 'model_eval.pth')
torch.save(model_eval.state_dict(), 'model_eval_params.pth')
torch.save(model_reward, 'model_reward.pth')
torch.save(model_reward.state_dict(), 'model_reward_params.pth')


# 绘制损失值随 epoch 变化的折线图
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), eval_losses, label='Eval Loss')
plt.plot(range(1, num_epochs + 1), reward_losses, label='Reward Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epochs')
plt.legend()
plt.show()
