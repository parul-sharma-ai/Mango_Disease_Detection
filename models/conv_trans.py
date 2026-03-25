import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ----------------------------
# ConvTransformer Block
# ----------------------------
class ConvTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4., dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        x_flat = rearrange(x, 'b c h w -> b (h w) c')  # flatten dynamically

        # Attention
        x_attn, _ = self.attn(x_flat, x_flat, x_flat)

        # MLP + residuals
        x_out = x_flat + x_attn
        x_out = x_out + self.mlp(self.norm(x_out))

        # reshape back
        x_out = rearrange(x_out, 'b (h w) c -> b c h w', h=H, w=W)
        return x_out

# ----------------------------
# Conv-Transformer Backbone
# ----------------------------
class ConvTransformer(nn.Module):
    def __init__(self, in_channels=3, num_classes=5):
        super().__init__()
        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.AdaptiveAvgPool2d((14, 14))  # flexible size, 14x14 output

        # ConvTransformer blocks
        self.trans_block1 = ConvTransformerBlock(dim=128, num_heads=8)
        self.trans_block2 = ConvTransformerBlock(dim=128, num_heads=8)

        # Classification head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # Conv layers
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        # ConvTransformer blocks
        x = self.trans_block1(x)
        x = self.trans_block2(x)

        # Classification
        x = self.global_pool(x)  # [B, C, 1, 1]
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        return x
