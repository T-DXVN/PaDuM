import torch
from torch import nn
from mamba_ssm import Mamba

class MyMamba(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch,d_state,d_model):
        super(MyMamba, self).__init__()

        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.dim = d_model
        self.d_state = d_state
        # patch 
        self.patch_num = (seq_len - patch_len) // stride + 1
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            self.patch_num += 1

        # Patch Embedding
        self.fc1 = nn.Linear(patch_len, self.dim)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.patch_num)

        # Residual projection
        self.fc2 = nn.Linear(self.dim, self.dim)

        # Mamba 
        self.mamba1 = Mamba(
            d_model=self.dim,
            d_state=self.d_state,
            d_conv=2,
            expand=1
        )
        self.mamba2 = Mamba(
            d_model=self.dim,
            d_state=self.d_state,
            d_conv=2,
            expand=1
        )

        # Flatten + MLP Head
        self.flatten1 = nn.Flatten(start_dim=1)
        self.fc3 = nn.Linear(self.patch_num * self.dim, pred_len * 2)
        self.gelu4 = nn.GELU()
        self.fc4 = nn.Linear(pred_len * 2, pred_len)

    def forward(self, x):
        # x: [Batch, Input, Channel]
        x = x.permute(0, 2, 1)  # → [B, C, T]
        B, C, T = x.shape
        x = x.reshape(B * C, T)  # → [B*C, T]

        if self.padding_patch == 'end':
            x = self.padding_patch_layer(x)

        # Patching
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # → [B*C, patch_num, patch_len]

        # Patch Embedding
        x = self.fc1(x)       # [B*C, patch_num, dim]
        x = self.gelu1(x)
        x = self.bn1(x)

        # Residual
        res = self.fc2(x)     # [B*C, patch_num, dim]

        #  Mamba 
        x = self.mamba1(x)    # [B*C, patch_num, dim]
        x = x + res           
        x = self.mamba2(x)    # [B*C, patch_num, dim]

        # Flatten
        x = self.flatten1(x)  # [B*C, patch_num*dim]
        x = self.fc3(x)       # [B*C, pred_len*2]
        x = self.gelu4(x)
        x = self.fc4(x)       # [B*C, pred_len]

        # Reshape to [B, pred_len, C]
        x = x.view(B, C, self.pred_len).permute(0, 2, 1)
        return x
