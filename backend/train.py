"""Train a simple ranking model that scores recipes for a (user, recipe) pair.
This script constructs heuristic labels based on suitability and nutrient matching.
"""
import torch, random, numpy as np
from torch.utils.data import Dataset, DataLoader
from model import RecipeRanker
from data_processing import load_and_process
import torch.optim as optim
import pandas as pd

class PairDataset(Dataset):
    def __init__(self, df, num_cols, enc, scaler, n_samples=5000):
        self.df = df
        self.num_cols = num_cols
        self.n = n_samples
        self.enc = enc
        self.scaler = scaler
        self.prepare_samples()

    def prepare_samples(self):
        # create synthetic user features and label pairs
        samples = []
        for _ in range(self.n):
            # random user
            age = float(np.random.randint(18,70))
            height_cm = float(np.random.randint(150,195))
            weight_kg = float(np.random.randint(45,110))
            activity = float(np.random.choice([1.2,1.375,1.55,1.725])) # sedentary->very active
            goal = np.random.choice(['loss','gain','muscle'])
            deficiency = np.random.choice(['iron','none','vitd','protein'])
            chronic = np.random.choice(['diabetes','hypertension','none'])
            cuisine_pref = np.random.choice(self.df['cuisine'].unique())

            user = {'age':age,'height_cm':height_cm,'weight_kg':weight_kg,'activity':activity,'goal':goal,
                    'deficiency':deficiency,'chronic':chronic,'cuisine_pref':cuisine_pref}
            # pick a recipe at random
            ridx = np.random.randint(0, len(self.df))
            recipe = self.df.iloc[ridx]
            # label heuristics: increase score if recipe matches cuisine, meal time allowed, and suitability flags
            score = 0.1
            if recipe['cuisine'] == cuisine_pref: score += 0.3
            if goal == 'muscle' and recipe.get('suitable_for_muscle_gain',False): score += 0.25
            if goal == 'loss' and recipe.get('suitable_for_weight_loss',False): score += 0.25
            if deficiency == 'iron' and recipe.get('suitable_for_iron_deficiency',False): score += 0.25
            if chronic == 'diabetes' and recipe.get('suitable_for_diabetes',False): score += 0.25
            # add small nutrient-based boosts: protein or iron
            if goal == 'muscle' and recipe['protein_g'] > 15: score += 0.1
            if deficiency == 'iron' and recipe['iron_mg'] > 3: score += 0.1
            # clamp
            label = min(1.0, score)
            samples.append((user, ridx, float(label)))
        self.samples = samples

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        user, ridx, label = self.samples[idx]
        recipe = self.df.iloc[ridx]
        # user vector: normalized age, bmi, activity one-hot goal/def/chronic, cuisine as index
        bmi = user['weight_kg'] / ((user['height_cm']/100)**2 + 1e-6)
        user_vec = [user['age']/100.0, bmi/50.0, user['activity']/2.0]
        # one-hot goal, deficiency, chronic (small fixed mapping)
        goals = ['loss','gain','muscle']
        defs_ = ['none','iron','vitd','protein']
        chs = ['none','diabetes','hypertension']
        for g in goals: user_vec.append(1.0 if user['goal']==g else 0.0)
        for d in defs_: user_vec.append(1.0 if user['deficiency']==d else 0.0)
        for c in chs: user_vec.append(1.0 if user['chronic']==c else 0.0)
        # recipe vector: scaled numeric features (std_...)
        recipe_feats = recipe[[c for c in self.df.columns if c.startswith('std_')]].values.astype('float32')
        user_x = torch.tensor(user_vec, dtype=torch.float32)
        recipe_x = torch.tensor(recipe_feats, dtype=torch.float32)
        return user_x, recipe_x, torch.tensor(label, dtype=torch.float32)

def train():
    df, enc, scaler, num_cols = load_and_process()
    ds = PairDataset(df, num_cols, enc, scaler, n_samples=4000)
    dl = DataLoader(ds, batch_size=64, shuffle=True)
    user_dim = 3 + 3 + 4 + 3  # age,bmi,activity + goals + defs + chronic
    recipe_dim = len(df.columns[df.columns.str.startswith('std_')])
    model = RecipeRanker(user_dim, recipe_dim)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    for epoch in range(6):
        total=0.0
        for ux, rx, y in dl:
            pred = model(ux, rx)
            loss = loss_fn(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        print(f"Epoch {epoch} loss={total/len(dl):.4f}")
    torch.save(model.state_dict(), 'model.pt')
    print('Saved model.pt')

if __name__ == '__main__':
    train()