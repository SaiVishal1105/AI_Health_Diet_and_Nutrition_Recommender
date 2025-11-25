import torch
import torch.nn as nn

class RecipeRanker(nn.Module):
    def __init__(self, user_feat_dim, recipe_feat_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(user_feat_dim + recipe_feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 1),
            nn.Sigmoid()
        )
    def forward(self, user_x, recipe_x):
        x = torch.cat([user_x, recipe_x], dim=1)
        return self.net(x).squeeze(1)