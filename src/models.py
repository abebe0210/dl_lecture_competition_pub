import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128,
        dropout_rate: float = 0.3 # ドロップアウト率を追加
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim, dropout_rate=dropout_rate),
            ConvBlock(hid_dim, hid_dim, dropout_rate=dropout_rate),
            ConvBlock(hid_dim, hid_dim, dropout_rate=dropout_rate),
        )
        
        self.attention = SelfAttention(hid_dim)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.blocks(X)
        X = self.attention(X)

        return self.head(X)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        # self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size) # , padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))
        X = self.dropout(X)  # 活性化関数の後にドロップアウトを追加

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))
        X = self.dropout(X)  # 活性化関数の後にドロップアウトを追加

        # X = self.conv2(X)
        # X = F.glu(X, dim=-2)

        return X

class SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Conv1d(dim, dim, 1)
        self.key = nn.Conv1d(dim, dim, 1)
        self.value = nn.Conv1d(dim, dim, 1)
        
    def forward(self, x):
        q, k, v = self.query(x), self.key(x), self.value(x)
        attn = F.softmax(torch.bmm(q.transpose(1, 2), k), dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2))
        return out + x