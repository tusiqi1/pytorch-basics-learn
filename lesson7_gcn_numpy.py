"""
# 图卷积神经网络numpy实现，主要参考：
# https://mp.weixin.qq.com/s/sg9O761F0KHAmCPOfMW_kQ
# https://towardsdatascience.com/how-to-do-deep-learning-on-graphs-with-graph-convolutional-networks-7d2250723780
import numpy as np
A = np.matrix([
    [0, 1, 0, 0], # 出度，表示当前点向哪些点发出射线
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [1, 0, 1, 0]],
    dtype=float
) # 邻接矩阵, 出度矩阵

X = np.matrix([
            [i, -i]
            for i in range(A.shape[0])
        ], dtype=float) # 对每个节点定义一个特征矩阵

print(A * X) # 因为计算特征的时候是对特征X左乘A，此时是从出度的角度来对X进行求和的，也可以从入度B的角度求和。
# 自环改进
I = np.matrix(np.eye(A.shape[0]))
A_hat = A + I
print(A_hat * X) # 增加自身的特征后，查看后续的节点特征是多少

# 归一化改进
D = np.array(np.sum(A, axis=1).T)[0] # axis=1表示对A按行相加，获得某一个节点的出度。 度矩阵也可以根据自己的实际情况进行设计
D = np.matrix(np.diag(D)) # 获得出度矩阵
print("矩阵D是：",D)
print(D**-1 * A) # 对出度矩阵A左乘出度D矩阵的逆，表示对出度矩阵A的每一行进行归一化

print(D**-1 * A * X) # 重要注：若使用B * D**-1 *X 针对该有向图，则表示将某个点(末尾点)的特征信息用指向该点的邻居节点(开始点)的特征信息表示，各个点的特征信息加权求和加在指向的末尾点上，权重设计依据开始点的输出射线条数的平均值决定(即：1/(开始点共有几条输出射线))。

# 自环+归一化。对A_hat进行归一化后：
D_hat = np.array(np.sum(A_hat, axis=1).T)[0]
D_hat = np.matrix(np.diag(D_hat))

# 设置权重矩阵
W = np.matrix([
             [1, -1],
             [-1, 1]
         ])

print((D_hat**-1) * A_hat * X * W)


def relu(M):
    [a,b] = M.shape
    for i in range(a):
        for j in range(b):
            if M[i,j] < 0:
                M[i,j] = 0
    return M
print(relu(D_hat**-1 * A_hat * X * W))
"""


