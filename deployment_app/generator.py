import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=256, embed_dim=128, vocab_size=64, features=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.text_fc = nn.Linear(embed_dim, z_dim)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim*2, features*8, 4, 1, 0),  # 1x1 → 4x4
            nn.BatchNorm2d(features*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(features*8, features*4, 4, 2, 1),  # 4x4 → 8x8
            nn.BatchNorm2d(features*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(features*4, features*2, 4, 2, 1),  # 8x8 → 16x16
            nn.BatchNorm2d(features*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(features*2, features, 4, 2, 1),    # 16x16 → 32x32
            nn.BatchNorm2d(features),
            nn.ReLU(True),

            nn.ConvTranspose2d(features, 3, 4, 2, 1),             # 32x32 → 64x64
            nn.Tanh(),
        )

    def forward(self, z, text):
        text_emb = self.embed(text).mean(dim=1)
        text_feat = self.text_fc(text_emb)
        combined = torch.cat([z, text_feat], dim=1).unsqueeze(2).unsqueeze(3)
        return self.net(combined)
