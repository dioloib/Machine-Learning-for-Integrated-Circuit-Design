import torch
import networkx as nx
import pickle
from tqdm import tqdm

filename = 'train_data_tensor.pkl'
with open(filename, 'rb') as f:
    all_data = pickle.load(f)

dropout_rate = 0

# 打印all_data的长度
print(len(all_data))
all_features = []
data_last = all_data[-1]
features_last = []
labels = []

for data in tqdm(all_data):
    try:
        if dropout_rate > 0:
            if torch.rand(1).item() < dropout_rate:
                continue
        edge_index = data['edge_index']
        numNodes = data['x'].shape[0]
        label = data['y']

        eq1=torch.equal(data['x'], data_last['x'])
        eq2=torch.equal(data['edge_index'], data_last['edge_index'])

        if eq1 and eq2:
            continue
        data_last = data
        
        # print(data)

        # 将 edge_index 转换为边列表，用于 networkx
        edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))

        # 使用 networkx 构建有向图
        G = nx.DiGraph()
        G.add_edges_from(edges)

        # 节点个数
        node_count = len(G.nodes())

        # 边个数
        edge_count = len(G.edges())
        
        # print(node_count)
        # print(numNodes)
        # print(edge_count)
        # print(edge_index.shape)

        # 计算平均入度和出度
        average_indegree = sum(degree for _, degree in G.in_degree()) / node_count
        average_outdegree = sum(degree for _, degree in G.out_degree()) / node_count

        # 节点层级最小最大值之差 (拓扑排序)
        topological_order = list(nx.topological_sort(G))
        layer_min = min(topological_order)
        layer_max = max(topological_order)
        layer_difference = layer_max - layer_min

        # 图的半径 (最长最短路径)
        # G_undirected = G.to_undirected()
        # radius = nx.radius(G_undirected)
        radius = 0

        # 关键路径长度
        critical_path_length = nx.dag_longest_path_length(G)

        # 组合为特征向量
        features = [
            node_count,
            edge_count,
            average_indegree,
            average_outdegree,
            layer_difference,
            radius,
            critical_path_length
        ]

        if features_last == features:
            continue
        features_last = features

        # 打印结果
        # print(f"Node Count: {node_count}")
        # print(f"Edge Count: {edge_count}")
        # print(f"Average Indegree: {average_indegree / numNodes}")
        # print(f"Average Outdegree: {average_outdegree / numNodes}")
        # print(f"Layer Difference: {layer_difference}")
        # print(f"Graph Radius: {radius}")
        # print(f"Critical Path Length: {critical_path_length} \n")

        all_features.append(features)
        labels.append(label)
    except Exception as e:
        continue

# 将特征向量转换为张量
all_features = torch.tensor(all_features, dtype=torch.float)
labels = torch.tensor(labels, dtype=torch.float)
print(all_features.shape)
# 保存特征张量
torch.save(all_features, 'all_features.pt')
torch.save(labels, 'labels.pt')