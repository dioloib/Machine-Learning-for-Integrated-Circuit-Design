import torch
import subprocess
from pyparsing import Word, alphanums, Suppress, alphas
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import re
import os

# 定义GNN模型
class GNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(input_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = torch.mean(x, dim=0)  # 聚合节点特征
        x = self.fc(x)
        return x

# 加载 AIG 文件并提取特征
def load_aig_features(aig_file_path):
    command = f"yosys -p 'read_aiger {aig_file_path}; show -format dot -prefix output'"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Yosys failed: {result.stderr}")
    
    # 解析生成的 dot 文件
    dot_file_path = 'output.dot'
    with open(dot_file_path, 'r') as file:
        lines = file.readlines()

    nodes = []
    edges = []

    node_parser = Word(alphanums + ":._") + Suppress("[label=") + Word(alphanums + alphas + "\"") + Suppress("];")
    edge_parser = Word(alphanums + ":._") + Suppress("->") + Word(alphanums + ":._")

    for line in lines:
        if '->' in line:
            parsed_edge = edge_parser.parseString(line)
            source_str = parsed_edge[0].split(":")[0]
            target_str = parsed_edge[1].split(":")[0]
            
            if source_str.startswith('n'):
                source = int(source_str[1:])  # 去掉 'n' 后转换为整数
            else:
                source = int(source_str.replace('c', ''))

            if target_str.startswith('n'):
                target = int(target_str[1:])  # 去掉 'n' 后转换为整数
            else:
                target = int(target_str.replace('c', ''))

            edges.append([source, target])
        elif '[label=' in line:
            parsed_node = node_parser.parseString(line)
            node_id_str = parsed_node[0].split(":")[0]
            
            if node_id_str.startswith('n'):
                node_id = int(node_id_str[1:])  # 去掉 'n' 后转换为整数
            else:
                node_id = int(node_id_str.replace('c', ''))

            label = parsed_node[1].replace('"', '')
            nodes.append([node_id, label])

    # 检查提取的节点和边
    print(f"Nodes: {nodes}")
    print(f"Edges: {edges}")

    # 转换为张量
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    node_features = [int(node[1]) if node[1].isdigit() else 0 for node in nodes]
    x = torch.tensor(node_features, dtype=torch.float).view(-1, 1)

    # 检查转换后的张量
    print(f"Node features tensor: {x}")
    print(f"Edge index tensor: {edge_index}")

    return x, edge_index

# 应用综合操作
def apply_synthesis_op(aig_file_path, op):
    output_aig = 'output.aig'
    command = f"yosys -p 'read_aiger {aig_file_path}; {op}; write_aiger {output_aig}'"
    subprocess.run(command, shell=True)
    return output_aig

# 获取AIG文件的评估值
def evaluate_aig(aig_file, lib_file, log_file):
    command = f"./yosys-abc -c 'read {aig_file}; read_lib {lib_file}; map; topo; stime' > {log_file}"
    subprocess.run(command, shell=True)
    with open(log_file, 'r') as f:
        area_info = re.findall(r'[a-zA-Z0-9\.\+]+', f.readlines()[-1])
    adp_val = float(area_info[-9]) * float(area_info[-4])
    return adp_val

# 优化AIG文件并输出最优动作序列
def optimize_aig(model_eval, model_reward, initial_aig_path, synthesis_ops, device):
    current_aig = initial_aig_path
    best_sequence = []
    max_steps = 10  # 最大综合步骤数
    lib_file = './lib/7nm/7nm.lib'
    log_file = 'alu4.log'

    for step in range(max_steps):
        childs = []
        for child in range(len(synthesis_ops)):
            child_file = f'alu4_{child}.aig'
            abc_run_cmd = (f"./yosys-abc -c 'read {current_aig}; {synthesis_ops[child]}; "
                           f"read_lib {lib_file}; write {child_file}; print_stats' > {log_file}")
            os.system(abc_run_cmd)
            if not os.path.exists(child_file):
                print(f"Error: {child_file} not found")
                continue
            childs.append(child_file)

        if not childs:
            print("No valid AIG files generated, stopping optimization.")
            break

        child_scores = []
        for child_file in childs:
            x, edge_index = load_aig_features(child_file)
            data = Data(x=x, edge_index=edge_index).to(device)
            with torch.no_grad():
                reward_value = model_reward(data.x, data.edge_index)
            child_scores.append(reward_value.item())

        action = torch.argmax(torch.tensor(child_scores)).item()
        best_sequence.append(synthesis_ops[action])
        current_aig = childs[action]

    return best_sequence

# 加载训练好的模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dim = 2
hidden_dim = 64
output_dim = 1

model_eval = GNNModel(input_dim, hidden_dim, output_dim).to(device)
model_reward = GNNModel(input_dim, hidden_dim, output_dim).to(device)

model_eval.load_state_dict(torch.load("model_eval_params.pth"))
model_reward.load_state_dict(torch.load("model_reward_params.pth"))

# 设置模型为评估模式
model_eval.eval()
model_reward.eval()

# 定义综合操作序列
synthesis_ops = ["refactor", "refactor -z", "rewrite", "rewrite -z", "resub", "resub -z", "balance"]

# 优化 AIG 文件并输出最优动作序列
initial_aig_path = "alu4.aig"
best_sequence = optimize_aig(model_eval, model_reward, initial_aig_path, synthesis_ops, device)
print("最优动作序列:", best_sequence)

