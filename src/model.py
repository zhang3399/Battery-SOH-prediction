import torch.nn as nn
import torch

class BatteryLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# class BatteryLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, dropout, output_size):
#         super().__init__()
#         self.lstm = nn.LSTM(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             dropout=dropout,
#             batch_first=True
#         )
#         self.attention = nn.Sequential(
#             nn.Linear(hidden_size, 16),
#             nn.Tanh(),
#             nn.Linear(16, 1),
#             nn.Softmax(dim=1)
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_size, 64),
#             nn.LayerNorm(64),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(64, output_size)
#         )
#
#     def forward(self, x):
#         lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden]
#         attn_weights = self.attention(lstm_out)  # [batch, seq_len, 1]
#         context = torch.sum(attn_weights * lstm_out, dim=1)  # [batch, hidden]
#         return self.fc(context)