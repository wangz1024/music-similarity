"""
    计算特征向量间的相似度(使用余弦相似度)
"""

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

base_path = '/home/wangz/Desktop/音乐推荐系统/vector'


# 从文件中返回一个向量集合
def genVectorListFromFile(txtpath):
    list = []
    with open(txtpath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            vec = line.split(',')[1]
            list.append([float(x) for x in vec.replace('\n', '').split('-')])
    return list


# 返回相似度大矩阵, 数值越大表明越相似
def cosSimMatrix(vector_list):
    return cosine_similarity(vector_list)


# 从相似度大矩阵中找出与之最相似的样本
def findSimSample(num, cos_sim_matrix, sim_num):
    index_list = []
    # 将自身相似度置为0
    cos_sim_matrix[num - 1][num - 1] = 0
    for sn in range(sim_num):
        index = np.argmax(cos_sim_matrix[num - 1], axis=0)
        cos_sim_matrix[num - 1][index] = 0
        index_list.append(index + 1)
    return index_list


# test
# vec_list = genVectorListFromFile(os.path.join(base_path, 'ChaCha.txt'))
# matrix = cosSimMatrix(vec_list)
# print(matrix)
# print('=========================================')
# print(matrix[0])
# print('=========================================')
# print(findSimSample(50, matrix, 3))


files = os.listdir(base_path)
for f in files:
    # 在每个分类下进行相似度推荐
    print(f)
    vector_list = genVectorListFromFile(os.path.join(base_path, f))
    matrix = cosSimMatrix(vector_list)
    with open(os.path.join(base_path, f), 'r+') as f:
        lines = f.readlines()
        f.seek(0)
        for index, line in enumerate(lines):
            newline = line.replace('\n', '') + ',' + '-'.join([str(x) for x in findSimSample(index + 1, matrix, 3)])
            f.write(newline)
            f.write('\n\r')
