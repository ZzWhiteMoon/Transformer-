import torch
import torch.nn as nn
import math

class PositionEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len = 5000):
        """
        位置编码

        参数：
            d_model: 嵌入维度
            dropout: 位置编码后应用的 Dropout 概率
            max_len: 序列的最大长度，适应不同长度的序列
        """
        super(PositionEmbedding,self).__init__()
        self.dropout = nn.Dropout(p=dropout) 

        # 位置编码矩阵 shape (max_len, d_model)
        pe = torch.zeros(max_len,d_model)
        # torch.arange(): 创建一个一维张量，从0 ~ max_len-1, shape 是(max_len, )
        # unsqueeze(1): 在第一维上插入一个大小为1的维度, shape 变成(max_len, 1) -> 变成列向量
        position = torch.arange(0, max_len).unsqueeze(1) # 位置索引 (max_len, 1)

        # 每个维度对应的频率
        # 一维张量，步长是2
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(1000.0) / d_model)
        )

        # 
        pe[:,0::2] = torch.sin(position * div_term) # 偶数维度
        pe[:,1::2] = torch.cos(position * div_term) # 奇数维度

        # 增加一个维度，方便后续和输入相加，shape (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # 位置编码注册为模型的缓冲区，不作为参数更新
        self.register_buffer('pe', pe) 
    
    def forward(self, x):
        """
        前向传播函数。

        参数:
            x: 输入序列的嵌入向量，形状为 (batch_size, seq_len, d_model)。

        返回:
            加入位置编码和 Dropout 后的嵌入向量，形状为 (batch_size, seq_len, d_model)。
        """
        # 取出与输入序列长度相同的部分位置编码，并与输入相加
        x = x + self.pe[:, :x.size(1), :]
        
        # 应用 dropout
        return self.dropout(x)