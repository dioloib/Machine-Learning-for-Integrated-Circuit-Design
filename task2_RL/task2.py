# %%
import os
import pickle
import torch
import numpy as np
from torch_geometric.data import Data
import re
import abc_py
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from tqdm import tqdm, trange
import threading
from rich.progress import track
import subprocess
import sys

# %%
file_path = "project/task2/project_data2/adder_0.pkl"

# %%
data = pickle.load(open(file_path, "rb"))
data

# %%
train_name = []
test_name = []
for f in os.scandir('./project/InitialAIG/train'):
    train_name.append(f.name.split('.')[0])
for f in os.scandir('./project/InitialAIG/test'):
    test_name.append(f.name.split('.')[0])
print(train_name, test_name, sep='\n')
print(len(train_name), len(test_name))

# %%
num_node_features = 2
file_path = './project/task2/project_data2/'
i = 0
train_data_tensor = []
test_data_tensor = []


# %%
def generate(k, block_size):
    num_node_features = 2
    file_path = './project/task2/project_data2/'
    print("thread " + str(k) + " begins")
    train_data_tensor = []
    test_data_tensor = []
    # train = -1
    # if circ_name in train_name:
    #     train = 1
    # elif circ_name in test_name:
    #     train = 0
    # else:
    #     raise Exception("file not found") 
    dir_path, dir_name, files = next(os.walk(file_path))
    # print(files, type(files))
    beg = k * block_size
    print("thread", k, beg, block_size, len(files))
    if beg + block_size > len(files):
        block_size = len(files) - beg
    for i in trange(block_size):
        # print(f.name)
        # i += 1
        f = files[beg + i]
        # if f.split('_')[0] != circ_name:
        #     continue
        with open(file_path + f, "rb") as f:
            circ = pickle.load(f)
        states = circ['input']
        targets = circ['target']
        for state, target in zip(states, targets):
            # # state = circ['input'][-1]
            # target = circ['target'][-1]
            # for i in range(len(states)):
                # state = states[i]
                # target = targets[i]
            # print(state, target)
            circuitName, actions = state.split('_')
            logFile = ''
            nextState = ''
            # if circuitName in train_name:
                # print("train")
                # train = 1
            # if train == 1:
            logFile = './task2_aig/train/log/' + state + '.log'
            nextState = './task2_aig/train/aig/' + state + '.aig'
            circuitPath = './project/InitialAIG/train/' + circuitName + '.aig'
            # elif circuitName in test_name:
                # print("test")
                # train = 0
            # elif train == 0:
            #     logFile = './task2_aig/test/log/' + state + '.log'
            #     nextState = './task2_aig/test/aig/' + state + '.aig'
            #     circuitPath = './project/InitialAIG/test/' + circuitName + '.aig'
            # else:
            #     # print("None")
            #     raise Exception("file not found")
            # print(train)
            libFile = './project/lib/7nm/7nm.lib'
            synthesisOpToPosDic = {
                0: 'refactor',
                1: 'refactor -z',
                2: 'rewrite',
                3: 'rewrite -z',
                4: 'resub',
                5: 'resub -z',
                6: 'balance'
            }
            actionCmd = ''
            # print(actions)
            for action in actions:
                actionCmd += (synthesisOpToPosDic[int(action)] + ';')
            # print(actionCmd)
            abcRunCmd = "./oss-cad-suite-linux/bin/yosys-abc -c \"read " + circuitPath + ";" + actionCmd + "read_lib " + libFile + "; write " + nextState + "; print_stats\" > " + logFile
            # print(abcRunCmd)
            # os.system(abcRunCmd)
            if not os.path.exists(nextState):
                try:
                    p = subprocess.Popen(abcRunCmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
                    p.wait()
                except Exception as e:
                    print("yoys error: ", e)
                    continue
            try:
                _abc = abc_py.AbcInterface()
                _abc.start()
                _abc.read(nextState)
            except Exception as e:
                print("abc error: ", e)
                continue
            data = {}
            numNodes = _abc.numNodes()
            data['node_type'] = np.zeros(numNodes, dtype=int)
            data['num_inverted_predecessors'] = np.zeros(numNodes, dtype=int)
            edge_src_index = []
            edge_target_index = []
            for nodeIdx in range(numNodes):
                aigNode = _abc.aigNode(nodeIdx)
                nodeType = aigNode.nodeType()
                data['num_inverted_predecessors'][nodeIdx] = 0
                if nodeType == 0 or nodeType == 2:
                    data['node_type'][nodeIdx] = 0
                elif nodeType == 1:
                    data['node_type'][nodeIdx] = 1
                else:
                    data['node_type'][nodeIdx] = 2
                    if nodeType == 4:
                        data['num_inverted_predecessors'][nodeIdx] = 1
                    if nodeType == 5:
                        data['num_inverted_predecessors'][nodeIdx] = 2
                if (aigNode.hasFanin0()):
                    fanin = aigNode.fanin0()
                    edge_src_index.append(nodeIdx)
                    edge_target_index.append(fanin)
                if (aigNode.hasFanin1()):
                    fanin = aigNode.fanin1()
                    edge_src_index.append(nodeIdx)
                    edge_target_index.append(fanin)
            data['edge_index'] = torch.tensor([edge_src_index, edge_target_index], dtype=torch.long)
            data['node_type'] = torch.tensor(data['node_type'], dtype=torch.long)
            data['num_inverted_predecessors'] = torch.tensor(data['num_inverted_predecessors'], dtype=torch.long)
            data['nodes'] = numNodes
            # print(numNodes)
            nodeAttribute = torch.concat((data['node_type'], data['num_inverted_predecessors'])).reshape(-1, num_node_features)
            # print(nodeAttribute.shape)
            # os._exit()
            data_tensor = Data(x=nodeAttribute, edge_index=data['edge_index'], y=torch.tensor(target))
            # if train == 1:
            train_data_tensor.append(data_tensor.to_dict())
            # else:
                # test_data_tensor.append(data_tensor.to_dict())
    # if train == 1:
    print(len(train_data_tensor))
    pickle.dump(train_data_tensor, open('train_data_tensor_' + str(k) + '.pkl', 'wb'))
    # else:
        # print(len(test_data_tensor))
        # pickle.dump(test_data_tensor, open('test_data_tensor_' + circ_name + '.pkl', 'wb'))
    print("thread " + str(k) + "ends")

# %%
# generate(train_name[0])

# generate(2, 4000)
# print(sys.argv[0], sys.argv[1])
# print(type(sys.argv[1]))
# generate(int(sys.argv[1]), 4000)
for i in range(5):
    if os.path.exists('train_data_tensor_' + str(i + 16) + '.pkl'):
        continue
    generate(i + 16, 4000)

# thread_pool = []
# for i in range(17):
#     thread = threading.Thread(target=generate, args=(i + 3, 4000))
#     thread_pool.append(thread)
#     thread.start()

# for i in range(17):
#     thread_pool[i].join()