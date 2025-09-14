import torch
from torch import nn

class MyCNN(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch, trend):
        super(MyCNN, self).__init__()

        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.trend = trend
        self.dim = patch_len * patch_len

        # patch number
        self.patch_num = (seq_len - patch_len) // stride + 1
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            self.patch_num += 1

        # Patch Embedding
        self.fc1 = nn.Linear(patch_len, self.dim)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.patch_num)

        # CNN Depthwise 2D
        self.depthwise = nn.Conv2d(
            in_channels=self.patch_num,
            out_channels=self.patch_num,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.patch_num  # Depthwise
        )
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm2d(self.patch_num)

        # Residual projection
        self.fc2 = nn.Linear(self.dim, self.dim)

        # CNN Pointwise 2D
        self.conv2 = nn.Conv2d(
            in_channels=self.patch_num,
            out_channels=self.patch_num // trend,
            kernel_size=1
        )
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm2d(self.patch_num // trend)

        # Flatten + MLP Head
        self.flatten1 = nn.Flatten(start_dim=1)
        self.fc3 = nn.Linear((self.patch_num // trend) * self.patch_len * self.patch_len, pred_len * 2)
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
        x = self.fc1(x)       # → [B*C, patch_num, dim]
        x = self.gelu1(x)
        x = self.bn1(x)

        # Residual
        res = self.fc2(x)     # → [B*C, patch_num, dim]

        # Reshape to 2D: [B*C, patch_num, patch_len, patch_len]
        x = x.view(B * C, self.patch_num, self.patch_len, self.patch_len)
        res = res.view(B * C, self.patch_num, self.patch_len, self.patch_len)

        # CNN Depthwise
        x = self.depthwise(x)  # → [B*C, patch_num, patch_len, patch_len]
        x = self.gelu2(x)
        x = self.bn2(x)

        # Add residual
        x = x + res  # [B*C, patch_num, patch_len, patch_len]

        # CNN Pointwise
        x = self.conv2(x)     # → [B*C, patch_num//trend, patch_len, patch_len]
        x = self.gelu3(x)
        x = self.bn3(x)

        # Flatten
        x = self.flatten1(x)  # → [B*C, flatten_dim]
        x = self.fc3(x)       # → [B*C, pred_len*2]
        x = self.gelu4(x)
        x = self.fc4(x)       # → [B*C, pred_len]

        # Reshape to [B, pred_len, C]
        x = x.view(B, C, self.pred_len).permute(0, 2, 1)
        return x

