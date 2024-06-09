import re
import os
import sys

libFile = './lib/7nm/7nm.lib'
logFile = './log/mylog.log'

def evaluate(aigFile):
    logFile = './log/eval.log'
    libFile = './lib/7nm/7nm.lib'
    abcRunCmd = "./yosys-abc -c \"" + "read " + aigFile + "; read_lib " + libFile + "; " + \
            "map; topo; stime \" > " + logFile
    os.system(abcRunCmd)
    with open(logFile) as f:
        areaInformation = re.findall('[a-zA-Z0-9.]+', f.readlines()[-1])
    eval = float(areaInformation[-9]) * float(areaInformation[-4])
    return eval

def Evaluation(children):
    scores = []
    for child in children:
        scores.append(evaluate(child))
    return scores

synthesisOpToPosDict = {
    0: "refactor",
    1: "refactor -z",
    2: "rewrite",
    3: "rewrite -z",
    4: "resub",
    5: "resub -z",
    6: "balance"
}

def greedy(AIG, AIGname):
    seq = []
    aig = AIG
    bestScore = 0
    for step in range(10):
        children = []
        for child in range(7):
            childseq = seq.copy()
            childseq.append(child)
            seqStr = seq2str(childseq)
            childFile = './aig/'+AIGname+'_{}.aig'.format(seqStr)
            abcRunCmd = "./yosys-abc -c \"read {}; ".format(aig) + \
                        synthesisOpToPosDict[child] + "; " + \
                        "read_lib {}; write ".format(libFile) + \
                        childFile + "; print_stats \" > " + logFile
            os.system(abcRunCmd)
            children.append(childFile)

        childScores = Evaluation(children)
        action = childScores.index(max(childScores))
        bestScore = childScores[action]
        aig = children[action]
        seq.append(action)
    return bestScore, seq

def seq2str(seq):
    seqStr = ''
    for action in seq:
        seqStr += str(action)
    return seqStr

def DFS(aigFile, AIGname, seq): # 传入父节点的AIG文件和当前操作序列
    seqStr = seq2str(seq)
    
    if len(seqStr) > 0:
        # 生成当前节点的AIG文件
        myFile = './aig/'+AIGname+'_{}.aig'.format(seqStr)
        abcRunCmd = "./yosys-abc -c \"read {}; ".format(aigFile) + \
                    synthesisOpToPosDict[seq[-1]] + "; " + \
                    "read_lib {}; write ".format(libFile) + \
                    myFile + "; print_stats \" > " + logFile
        os.system(abcRunCmd)
    else:
        myFile = aigFile

    if len(seq) == 10:
        if seqStr[-1] == '0' and seqStr[-2] == '0':
            print('Current AIG:', myFile)
        # 计算当前节点的评估值
        return evaluate(myFile)

    # 递归调用DFS
    children = []
    for child in range(7):
        childSeq = seq.copy()
        childSeq.append(child)
        children.append(DFS(myFile, childSeq))
    
    # 返回最大评估值
    return max(children)

def beamSearch(aigFile, AIGname, seq, beamWidth): # 传入当前节点的AIG文件和当前操作序列
    # 将seq转换为字符串
    seqStr = ''
    for action in seq:
        seqStr += str(action)

    if len(seq) == 10:
        # if seqStr[-1] == '0' and seqStr[-2] == '0':
        #     print('Current AIG:', aigFile)
        # 计算当前节点的评估值
        return evaluate(aigFile), seq

    childrenScore = []
    childrenSeq = []
    childrenFile = []
    for child in range(7):
        childSeq = seq.copy()
        childSeq.append(child)
        childrenSeq.append(childSeq)
        # 生成子节点的AIG文件
        childFile = './aig/'+AIGname+'_{}.aig'.format(seq2str(childSeq))
        abcRunCmd = "./yosys-abc -c \"read {}; ".format(aigFile) + \
                    synthesisOpToPosDict[child] + "; " + \
                    "read_lib {}; write ".format(libFile) + \
                    childFile + "; print_stats \" > " + logFile
        os.system(abcRunCmd)
        childrenFile.append(childFile)
        # 计算子节点的评估值
        childScore = evaluate(childFile)
        childrenScore.append(childScore)
    
    # 对评估值进行排序
    childrenWithScore = list(zip(childrenScore, childrenSeq, childrenFile))
    childrenWithScore.sort(key=lambda x: x[0], reverse=True)
    # 取前beamWidth个子节点
    bestChildrenWithScore = childrenWithScore[:beamWidth]
    bestChildrenSeq = []
    bestChildrenFile = []
    for i in range(beamWidth):
        bestChildrenSeq.append(bestChildrenWithScore[i][1])
        bestChildrenFile.append(bestChildrenWithScore[i][2])
    
    # print('seq:', seq)
    # print('Best Children Seq:', bestChildrenSeq)

    # 递归调用beamSearch
    children = []
    childrenSeq = []
    for i in range(beamWidth):
        bestChild, bestSeq = beamSearch(bestChildrenFile[i], AIGname, bestChildrenSeq[i], beamWidth)
        children.append(bestChild)
        childrenSeq.append(bestSeq)

    # 返回最大评估值
    rt_index = children.index(max(children))

    # print('Current AIG:', aigFile)
    # for i in range(beamWidth):
    #     print('Score:', children[i], 'Seq:', seq2str(childrenSeq[i]))
    # print('Best Child:{}\n\n'.format(rt_index))

    return children[rt_index], childrenSeq[rt_index]

AIG = './InitialAIG/test/alu4.aig'
AIGname = AIG.split('/')[-1].split('.')[0]

seq = []
beamWidth = 2
score, seq = beamSearch(AIG, AIGname, seq, beamWidth)
# score, seq = greedy(AIG, AIGname)

print('Best Score:', score)
print('Best Sequence:', seq2str(seq))

# 保存最佳序列
filename = AIGname + '.txt'
text = 'Best Score: ' + str(score) + '\n' + 'Best Sequence: ' + seq2str(seq)
with open(filename, 'w') as f:
    f.write(text)

# beam-2

# alu4.aig
# Best Score: 493824.6411
# Best Sequence: 3000034142

# apex1.aig
# Best Score: 949760.721
# Best Sequence: 4444444444

# apex2.aig
# Best Score: 67138.8216
# Best Sequence: 4444444444

# b9.aig
# Best Score: 9030.67
# Best Sequence: 6464444444

# Greedy

# alu4.aig
# Best Score: 443107.1095
# Best Sequence: 3000000000

# apex1.aig
# Best Score: 949760.721
# Best Sequence: 4444444444

# apex2.aig
# Best Score: 60309.5328
# Best Sequence: 1110000000