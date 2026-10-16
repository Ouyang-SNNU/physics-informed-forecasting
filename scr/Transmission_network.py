import torch
import torch.nn as nn
import torch.nn.functional as F

CLIMATE_WINDOW = 56


class ClimateEncoderCNN(nn.Module):
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
        # MaxPool1d(2): seq_len → seq_len // 4, 8
        fc_input_dim = 8 * (seq_len // 4)
        self.fc = nn.Linear(fc_input_dim, out_dim)

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, 2, -1)
        x = self.conv(x)
        x = x.view(B, -1)
        return self.fc(x)


class TCNBlock(nn.Module):
    def __init__(self, ch, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(ch, ch, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(ch, ch, kernel_size, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, T)
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
        return out + x


class TransNet(nn.Module):
    def __init__(self, hidden_dim=32, kernel_size=3,
                 dilations=(1, 2, 4, 8, 16), dropout=0.1):
        super().__init__()
        self.seq_len = CLIMATE_WINDOW
        self.hidden_dim = hidden_dim

        # encoder (output 4 dim)
        self.encoder = ClimateEncoderCNN(
            in_channels=2, seq_len=CLIMATE_WINDOW, out_dim=4)

        # input projection: 7 (4 climate + 3 seasonal) → hidden_dim
        self.input_proj = nn.Conv1d(7, hidden_dim, kernel_size=1)

        # TCN blocks
        self.tcn_blocks = nn.ModuleList()
        for dil in dilations:
            self.tcn_blocks.append(
                TCNBlock(hidden_dim, kernel_size, dil, dropout))

        # output
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_clim, x_seasonal):
        lead_shape = x_clim.shape[:-1]  # e.g. (L,)

        x_clim_flat = x_clim.reshape(-1, self.seq_len * 2)
        x_seas_flat = x_seasonal.reshape(-1, 3)

        clim_encoded = self.encoder(x_clim_flat)

        combined = torch.cat([clim_encoded, x_seas_flat], dim=-1)

        # (B, C, T)
        tcn_input = combined.unsqueeze(0).transpose(1, 2)

        x = self.input_proj(tcn_input)

        for block in self.tcn_blocks:
            x = block(x)

        x = x.squeeze(0).transpose(0, 1)
        beta_flat = self.head(x)

        return beta_flat.reshape(*lead_shape)
