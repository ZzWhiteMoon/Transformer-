import torch
import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        位置前馈网络。
        
        参数:
            d_model: 输入和输出向量的维度
            d_ff: FFN 隐藏层的维度，或者说中间层
            dropout: 随机失活率（Dropout），即随机屏蔽部分神经元的输出，用于防止过拟合
        
        （实际上论文并没有确切地提到在这个模块使用 dropout，所以注释）
        """
        super(PositionwiseFeedForward,self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff) # 第一个线性层
        self.w_2 = nn.Linear(d_ff, d_model) # 2nd
        # self.dropout = nn.Dropout(dropout) # Dropout

    def forward(self, x):
        return self.w_2(self.w_1(x).relu())