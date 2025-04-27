import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        """
        多头注意力机制
        把输入x拆成N份，每份拥有独立的Wq Wk Wv
        按注意力机制公式计算：(softmax(QK)/sqrt(Dk))V
        
        参数：
            embed_size: 输入序列的嵌入维度
            num_heads: 注意力头的数量
        """
        super(MultiHeadAttention, self).__init__()
        # assert是检查（断言），如果不满足条件就抛出错误并打印后面的信息
        assert embed_size % num_heads == 0, "embed_size 必须能被 num_heads 整除"

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads # 每个头的维度

        # 每个头都要有单独的Wq Wk Wv
        # ModuleList是一个特殊容器，用于保存任意数量的子模块，让Torch能识别他们
        # nn.Linear是一个线性仿射变换，将维度从embed_size投射到head_dim
        # 所以W_q中存储的是一系列仿射变换的参数
        self.W_q = nn.ModuleList([nn.Linear(embed_size, self.head_dim) for _ in range(num_heads)])
        self.W_k = nn.ModuleList([nn.Linear(embed_size, self.head_dim) for _ in range(num_heads)])
        self.W_v = nn.ModuleList([nn.Linear(embed_size, self.head_dim) for _ in range(num_heads)])

        # 输出拼接
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, q, k, v, mask=None):
        """
        前向传播
        参数：
            q: (batch_size, seq_len_q, embed_size)
            k
            v
        """
        batch_size = q.shape[0]
        multi_head_outputs = []

        for i in range(self.num_heads):
            # (...)表示PyTorch中调用一个nn.Module，相当于调用它的forward方法，完成一次前向计算
            # q的形状是(batch_size, seq_len_q, embed_size) -> Q = q (W_i)^T + b_i
            Q = self.W_q[i](q)  # Q shape(batch_size, seq_len_q, head_dim)
            K = self.W_k[i](k)  
            V = self.W_v[i](v)

            # 计算
            scaled_attention, _ = scaled_dot_product_attention(Q,K,V,mask)
            multi_head_outputs.append(scaled_attention)

        # 拼接
        concat_out = torch.cat(multi_head_outputs,dim=-1) # (batch_size, seq_len_q, embed_size)

        # 输出线性层
        out = self.fc_out(concat_out)

        return out
    
def scaled_dot_product_attention(Q,K,V,mask=None):
    """
    Attention的计算

    参数：
        Q, K, V     (batch_size, seq_len_q, embed_size)
        mask: 掩码矩阵, 屏蔽不关注的位置
    """

    embed_size = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(embed_size)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # dim =-1就是对最后一个维度进行操作，具体例子见下
    attention_weights = F.softmax(scroes, dim=-1)

    output = torch.matmul(attention_weights, V)

    return output, attention_weights

"""
[
  [  [a, b, c],    # batch 0, seq 0
     [d, e, f] ],  # batch 0, seq 1

  [  [g, h, i],    # batch 1, seq 0
     [j, k, l] ]   # batch 1, seq 1
]

直观上看，batch是第一个维度：[[],[],[]]
里面的[]是第二个维度，代表一个个序列
[]中的值就是第三个维度，也就是softmax要操作的维度，就是具体的数值了
"""