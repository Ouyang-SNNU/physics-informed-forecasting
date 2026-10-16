"""
v1 网络架构: ClimateEncoderCNN + TCN head (suggest.md 2.1.1)

相比 v0 (Dynamics_Net.py) 的改进:
  - CLIMATE_WINDOW: 35 → 56 天
  - ClimateEncoderCNN: 输出维度保持 4, fc 输入自适应 seq_len
  - TransNet: MLP head → TCN (因果膨胀卷积, 残差连接)
    - 因果 padding: β[t] 仅依赖 ≤t 的气候特征
    - 膨胀率: [1, 2, 4, 8, 16] — 感受野 ~63 天
    - 残差连接: 稳定梯度, 无需 BPTT
    - 最终 Sigmoid: β ∈ (0, 1)

类:
  ClimateEncoderCNN   — 56 天气象窗口 → 1D CNN → 4 维气候编码
  TCNBlock             — 因果膨胀卷积残差块
  TransNet     — encoder + TCN head → β
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ====================== 全局窗口参数 ======================
CLIMATE_WINDOW = 56  # v1: 35 → 56 天
# ==========================================================


class ClimateEncoderCNN(nn.Module):
    """
    56 天气象窗口 → 1D CNN → 4 维气候特征.

    输入尺寸适配 seq_len=56:
      Conv1d(2, 16, k5) → ReLU → MaxPool1d(2)   # (16, 28)
      Conv1d(16, 32, k5) → ReLU → MaxPool1d(2)  # (32, 14)
      Conv1d(32, 8, k3) → ReLU                   # (8, 14)
      Flatten → fc(112, 4)

    seq_len=35 时: fc(64, 4); seq_len=56 时: fc(112, 4).
    """
    def __init__(self, in_channels=2, seq_len=56, out_dim=4):
        super().__init__()
        self.seq_len = seq_len
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),  # seq_len → seq_len//2

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),  # seq_len//2 → seq_len//4

            nn.Conv1d(32, 8, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # 经过 2 次 MaxPool1d(2): seq_len → seq_len // 4, 8 通道
        fc_input_dim = 8 * (seq_len // 4)
        self.fc = nn.Linear(fc_input_dim, out_dim)

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, 2, -1)
        x = self.conv(x)
        x = x.view(B, -1)
        return self.fc(x)


class TCNBlock(nn.Module):
    """
    因果膨胀卷积残差块.

    输入/输出形状相同: (B, C, T).
    每层卷积独立左侧 padding, 确保因果属性: 输出[t] 仅依赖输入[:t+1].
    残差连接确保梯度无障碍流动 (无需 BPTT 截断).
    """
    def __init__(self, ch, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(ch, ch, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(ch, ch, kernel_size, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, T) — 每层独立因果左填充
        # Conv1
        out = F.pad(x, (self.padding, 0))
        out = self.conv1(out)
        out = torch.relu(out)
        out = self.dropout(out)
        # Conv2
        out = F.pad(out, (self.padding, 0))
        out = self.conv2(out)
        out = torch.relu(out)
        out = self.dropout(out)
        return out + x  # 残差连接


class TransNet(nn.Module):
    """
    v1 直接 β 预测 (encoder + TCN head).

    流程:
      1. ClimateEncoderCNN: (*, 112) → (*, 4)         气候编码
      2. 拼接季节特征:      (*, 4) + (*, 3) → (*, 7)  每日特征
      3. Input projection:  Conv1d(7, hidden)          通道映射
      4. TCN blocks × 5:    因果膨胀卷积 [1,2,4,8,16]  时序建模
      5. Head:              Linear(hidden, 1) → Sigmoid  β ∈ (0,1)

    因果感受野: (kernel-1) * sum(dilations) = 2 * 31 = 62 天
    """
    def __init__(self, hidden_dim=32, kernel_size=3,
                 dilations=(1, 2, 4, 8, 16), dropout=0.1):
        super().__init__()
        self.seq_len = CLIMATE_WINDOW
        self.hidden_dim = hidden_dim

        # 气候编码器 (输出 4 维)
        self.encoder = ClimateEncoderCNN(
            in_channels=2, seq_len=CLIMATE_WINDOW, out_dim=4)

        # 输入投影: 7 (4 climate + 3 seasonal) → hidden_dim
        self.input_proj = nn.Conv1d(7, hidden_dim, kernel_size=1)

        # TCN 残差块
        self.tcn_blocks = nn.ModuleList()
        for dil in dilations:
            self.tcn_blocks.append(
                TCNBlock(hidden_dim, kernel_size, dil, dropout))

        # 输出头
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_clim, x_seasonal):
        """
        x_clim:     (*, CLIMATE_WINDOW*2)  e.g. (L, 112)
        x_seasonal: (*, 3)
        返回:        (*,)                  β 预测值, 经 Sigmoid ∈ (0,1)
        """
        lead_shape = x_clim.shape[:-1]  # e.g. (L,)

        # 展平前导维度
        x_clim_flat = x_clim.reshape(-1, self.seq_len * 2)
        x_seas_flat = x_seasonal.reshape(-1, 3)

        # 气候编码
        clim_encoded = self.encoder(x_clim_flat)           # (-1, 4)

        # 拼接 → 每日特征向量
        combined = torch.cat([clim_encoded, x_seas_flat], dim=-1)  # (-1, 7)

        # 转为 TCN 需要的 (B, C, T) 格式
        tcn_input = combined.unsqueeze(0).transpose(1, 2)  # (1, 7, T)

        # 输入投影
        x = self.input_proj(tcn_input)                     # (1, hidden, T)

        # TCN 残差块
        for block in self.tcn_blocks:
            x = block(x)                                    # (1, hidden, T)

        # 转回 (T, hidden) → head → (T, 1)
        x = x.squeeze(0).transpose(0, 1)                   # (T, hidden)
        beta_flat = self.head(x)                            # (T, 1)

        return beta_flat.reshape(*lead_shape)               # (*,)
