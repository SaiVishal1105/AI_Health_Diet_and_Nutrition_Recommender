import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = r"C:\Users\admin\Downloads\diet_recommender_project\backend\healthy_recipes_augmented.xlsx"

def load_and_process():
    df = pd.read_excel(DATA_PATH)
    # Basic cleaning
    df = df.dropna(subset=['recipe_name','meal_type']).reset_index(drop=True)
    # Fill numeric columns if missing
    num_cols = ['calories','protein_g','carbs_g','fat_g','fiber_g','glycemic_index','iron_mg','calcium_mg','vitamin_d_mcg','selenium_mcg',
                'saturated_fat_g','sugar_g','sodium_mg','potassium_mg','vitamin_b12_mcg','vitamin_c_mg']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
        else:
            df[c] = 0.0
    # Binarize suitability columns if present
    bool_cols = [c for c in df.columns if c.startswith('suitable_for_') or c.startswith('recommended_for_')]
    for c in bool_cols:
        df[c] = df[c].astype(bool)
    # Encode cuisine and meal_type
    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat = df[['cuisine','meal_type']].fillna('Unknown')
    enc_arr = enc.fit_transform(cat)
    enc_cols = list(enc.get_feature_names_out(['cuisine','meal_type']))
    enc_df = pd.DataFrame(enc_arr, columns=enc_cols, index=df.index)
    df = pd.concat([df, enc_df], axis=1)
    # numeric scaler (for training/prediction)
    scaler = StandardScaler()
    numeric = df[num_cols].fillna(0.0)
    scaled = scaler.fit_transform(numeric)
    scaled_df = pd.DataFrame(scaled, columns=[f'std_{c}' for c in num_cols], index=df.index)
    df = pd.concat([df, scaled_df], axis=1)
    return df, enc, scaler, num_cols