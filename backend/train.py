"""
Train a ranking model that scores how suitable a recipe is for a user.
This version FIXES:
- low label variance
- synthetic user randomness
- missing suitability logic
- use of sigmoid + MSE collapse
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from model import RecipeRanker
from data_processing import load_and_process
import torch.optim as optim


# -----------------------------
# Dataset
# -----------------------------
class PairDataset(Dataset):
    def __init__(self, df, n_samples=6000):
        self.df = df
        self.n = n_samples
        self.prepare_samples()

    def prepare_samples(self):
        """Generate realistic user → recipe scores."""
        samples = []

        for _ in range(self.n):
            # ---- USER ----
            age = np.random.randint(18, 70)
            height = np.random.randint(150, 195)
            weight = np.random.randint(45, 110)
            bmi = weight / ((height / 100) ** 2)

            activity = np.random.choice([1.2, 1.375, 1.55, 1.725])
            goal = np.random.choice(['loss', 'gain', 'muscle'])
            deficiency = np.random.choice(['none', 'iron', 'vitd', 'protein'])
            chronic = np.random.choice(['none', 'diabetes', 'hypertension'])
            cuisine_pref = np.random.choice(self.df['cuisine'].unique())

            user = {
                "age": age,
                "bmi": bmi,
                "activity": activity,
                "goal": goal,
                "deficiency": deficiency,
                "chronic": chronic,
                "cuisine_pref": cuisine_pref,
            }

            # ---- RECIPE ----
            ridx = np.random.randint(0, len(self.df))
            recipe = self.df.iloc[ridx]

            # ---- SCORING (IMPROVED) ----
            score = 0.0

            # 1. Cuisine preference
            if recipe["cuisine"] == cuisine_pref:
                score += 0.35

            # 2. Goal logic
            if goal == "muscle":
                score += min(recipe["protein_g"] / 40, 0.35)  # high protein

            if goal == "loss":
                score += (1 - min(recipe["calories"] / 800, 1)) * 0.30  # low calories

            # 3. Deficiency logic
            if deficiency == "iron":
                score += min(recipe["iron_mg"] / 8, 0.35)

            if deficiency == "protein":
                score += min(recipe["protein_g"] / 40, 0.35)

            # 4. Chronic condition logic
            if chronic == "diabetes":
                score += (1 - min(recipe["sugar_g"] / 10, 1)) * 0.30

            if chronic == "hypertension":
                score += (1 - min(recipe["sodium_mg"] / 600, 1)) * 0.30

            # clamp 0–1
            label = float(max(0.0, min(score, 1.0)))

            samples.append((user, ridx, label))

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        user, ridx, label = self.samples[idx]
        recipe = self.df.iloc[ridx]

        # -----------------------------
        # Build USER vector
        # -----------------------------
        user_vec = [
            user["age"] / 100.0,
            user["bmi"] / 50.0,
            user["activity"] / 2.0,
        ]

        goals = ["loss", "gain", "muscle"]
        defs_ = ["none", "iron", "vitd", "protein"]
        chronics = ["none", "diabetes", "hypertension"]

        user_vec += [1.0 if user["goal"] == g else 0.0 for g in goals]
        user_vec += [1.0 if user["deficiency"] == d else 0.0 for d in defs_]
        user_vec += [1.0 if user["chronic"] == c else 0.0 for c in chronics]

        user_x = torch.tensor(user_vec, dtype=torch.float32)

        # -----------------------------
        # Build RECIPE vector
        # -----------------------------
        recipe_feats = recipe[
            [c for c in self.df.columns if c.startswith("std_")]
        ].values.astype("float32")

        recipe_x = torch.tensor(recipe_feats, dtype=torch.float32)

        return user_x, recipe_x, torch.tensor(label, dtype=torch.float32)


# -----------------------------
# Training Loop
# -----------------------------
def train():
    df, enc, scaler, num_cols = load_and_process()

    ds = PairDataset(df, n_samples=6000)
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    user_dim = 3 + 3 + 4 + 3
    recipe_dim = len(df.columns[df.columns.str.startswith("std_")])

    model = RecipeRanker(user_dim, recipe_dim)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    # IMPORTANT: using logits + BCE prevents output collapse
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(8):
        total = 0.0
        for ux, rx, y in dl:
            pred = model(ux, rx)  # raw logits
            loss = loss_fn(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        print(f"Epoch {epoch} loss={total / len(dl):.4f}")

    torch.save(model.state_dict(), "model.pt")
    print("Saved model.pt")


if __name__ == "__main__":
    train()
