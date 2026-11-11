import torch

def channel_difference_loss(A):
    # 假设A是一个形状为[b, m, w, h]的张量，其中m>1
    b, m, w, h = A.shape
    reshape_a  = A.reshape(b, m, -1)  # [b, m, w*h]

    # 确保通道数量大于1，否则无法计算相邻通道差值
    assert m > 1, "通道数量必须大于1才能计算相邻通道的差值"

    # 初始化一个张量来存储每对相邻通道的差值的绝对值
    differences = torch.zeros_like(reshape_a[:, :-1, :])  # 不包括最后一个通道

    # 计算相邻通道的差值并取绝对值
    for i in range(m - 1):  # 遍历直到倒数第二个通道
        differences[:, i, :] = torch.abs(reshape_a[:, i, :] - reshape_a[:, i + 1, :])

    print(differences)
    # 沿着通道维度求和得到每个样本的总差值损失
    # [b, m, w*h] → [b, w*h] → [b, 1]
    loss_per_sample = differences.sum(dim=1).mean(dim=1) # 这里已经没有通道维度了，所以求和即可

    print(loss_per_sample)
    # 对所有样本求求和得到最终损失
    final_loss = loss_per_sample.mean()

    return final_loss

a = torch.tensor([
    [
        [[1.0, 1], [1, 1]],
        [[5, 6], [7, 8]],
        [[9, 10], [11, 12]]
    ],
    [
        [[13, 14], [15, 16]],
        [[17, 18], [19, 20.0]],
        [[21, 22], [23, 24]]
    ]
])

l = channel_difference_loss(a)
print(l)

a = float('inf')
b = 2.0
c = max(a, b)
print(c)