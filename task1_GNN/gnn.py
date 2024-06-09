import pickle
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data, DataLoader
import matplotlib.pyplot as plt

# 定义原始GNN
class GNN(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GNN, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # 添加自环边
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # 线性转换 (Step 1)
        x = self.lin(x)

        # 通过所有节点对 (Step 2-5)
        return self.propagate(edge_index, x=x)

    def message(self, x_j, edge_index, size):
        # 将邻接节点的特征聚合 (Step 3)
        return x_j

    def update(self, aggr_out):
        # 返回聚合后的节点特征 (Step 5)
        return aggr_out

# 定义GNN+MLP网络
class GNN_MLP(torch.nn.Module):
    def __init__(self):
        super(GNN_MLP, self).__init__()
        self.conv1 = GNN(2, 16)  # 输入特征维度为2，输出特征维度为16
        self.conv2 = GNN(16, 32)  # 输出特征维度为32
        self.fc1 = torch.nn.Linear(32, 16)
        self.fc2 = torch.nn.Linear(16, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = torch_geometric.nn.global_mean_pool(x, data.batch)  # 全局池化层
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

# 加载数据
filename = 'train_data_tensor_99.pkl'

with open(filename, 'rb') as file:
    data_list = pickle.load(file)

# 将数据转换为torch_geometric的数据格式
pyg_data_list = []
for item in data_list:
    x = item['x'].float()  # 转换为float类型
    edge_index = item['edge_index'].long()  # 确保edge_index是long类型
    y = item['y'].view(-1, 1).float()  # 确保y是二维张量并转换为float类型
    pyg_data_list.append(Data(x=x, edge_index=edge_index, y=y))

# 创建数据加载器
train_loader = DataLoader(pyg_data_list, batch_size=32, shuffle=True)

# 实例化模型和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNN_MLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练模型
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

# 训练多个epoch并记录损失值
num_epochs = 100
losses = []
for epoch in range(num_epochs):
    loss = train()
    losses.append(loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss}")

# 绘制损失值曲线
plt.figure()
plt.plot(range(1, num_epochs + 1), losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs. Epochs')
plt.legend()
plt.show()
