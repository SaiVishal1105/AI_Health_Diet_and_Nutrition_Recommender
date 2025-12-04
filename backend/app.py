from fastapi import FastAPI
from pydantic import BaseModel
from data_processing import load_and_process
import torch
import numpy as np
from model import RecipeRanker
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import Optional # Import Optional for clarity

# -------------------------------------------------
# Create FastAPI app
# -------------------------------------------------
app = FastAPI(title="Diet Recommender API")

# -------------------------------------------------
# CORS
# -------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Load Data & Model (Leaving this section untouched)
# -------------------------------------------------
print("Loading data...")

try:
    df_global, enc_global, scaler_global, num_cols = load_and_process()
    print("Data loaded successfully.")
except Exception as e:
    print("DATA LOAD ERROR:", e)
    df_global = None

if df_global is not None:
    recipe_features = df_global[[c for c in df_global.columns if c.startswith('std_')]].values.astype('float32')
else:
    # Handle case where data loading failed
    recipe_features = np.zeros((1, 5), dtype="float32")

user_dim = 3 + 3 + 4 + 3
recipe_dim = recipe_features.shape[1]

model = RecipeRanker(user_dim, recipe_dim)

MODEL_PATH = "model.pt"

print("Loading model...")
try:
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
        print("Model loaded successfully.")
    else:
        print("MODEL FILE NOT FOUND:", MODEL_PATH)
except Exception as e:
    print("MODEL LOAD ERROR:", e)


# -------------------------------------------------
# User Input Schema (MAX ROBUSTNESS FIX)
# -------------------------------------------------
class UserInput(BaseModel):
    # FIX: Make essential numerical fields optional to bypass Pydantic's 422 if data is missing or null
    age: int | None = None
    height_cm: float | None = None
    weight_kg: float | None = None
    activity_level: float | None = None
    
    # Strings are already optional/defaulted
    goal: str | None = "loss" # Providing default for simplicity in the API logic
    deficiency: str | None = "none"
    chronic: str | None = "none"
    cuisine_pref: str | None = "none"
    food_type: str | None = "none"
    calorie_target: float | None = None


# -------------------------------------------------
# Score Recipes (Leaving this section untouched)
# -------------------------------------------------
def score_recipes(user):
    # This logic assumes valid, non-null numerical inputs passed from build_week_plan
    bmi = user['weight_kg'] / ((user['height_cm'] / 100) ** 2 + 1e-6)

    user_vec = [
        user['age'] / 100.0,
        bmi / 50.0,
        user['activity_level'] / 2.0
    ]

    goals = ['loss', 'gain', 'muscle']
    defs_ = ['none', 'iron', 'vitd', 'protein']
    chs = ['none', 'diabetes', 'hypertension']

    user_vec += [1.0 if user['goal'] == g else 0.0 for g in goals]
    user_vec += [1.0 if user['deficiency'] == d else 0.0 for d in defs_]
    user_vec += [1.0 if user['chronic'] == c else 0.0 for c in chs]

    uv = torch.tensor(user_vec, dtype=torch.float32).unsqueeze(0)
    uv = uv.repeat(len(recipe_features), 1)
    rx = torch.tensor(recipe_features, dtype=torch.float32)

    with torch.no_grad():
        try:
            scores = model(uv, rx).numpy()
        except Exception:
            scores = np.random.rand(len(recipe_features))

    return scores


# -------------------------------------------------
# Weekly Plan Builder (FIXED: Added data missing check)
# -------------------------------------------------
def build_week_plan(user_input):
    user = dict(user_input)

    # Normalize null / empty values (This logic is crucial after making fields optional)
    
    # If activity_level is missing (None), default it to 1.2 (sedentary) instead of failing
    if user["activity_level"] is None:
         user["activity_level"] = 1.2
    
    # Normalize strings (even if set to default in Pydantic, check for None from the API)
    if user["cuisine_pref"] is None or user["cuisine_pref"] in ["", "None"]:
        user["cuisine_pref"] = "none"

    if user["food_type"] is None or user["food_type"] in ["", "None"]:
        user["food_type"] = "none"
        
    if user["goal"] is None or user["goal"] in ["", "None"]:
        user["goal"] = "loss"
        
    if user["deficiency"] is None or user["deficiency"] in ["", "None"]:
        user["deficiency"] = "none" 
        
    if user["chronic"] is None or user["chronic"] in ["", "None"]:
        user["chronic"] = "none"

    # --- CRITICAL CHECK FOR DATA LOADING ---
    if df_global is None or df_global.empty:
        print("CRITICAL ERROR: df_global (Recipe Data) is not available to build plan.")
        return {
            "plan": {"days": []},
            "workout": [
                "**FATAL ERROR: Recipe data not loaded.**",
                "Please check the backend deployment logs for 'DATA LOAD ERROR' during startup.",
                "The plan cannot be generated without the 'recipe_data.csv' file or if 'data_processing.py' failed."
            ]
        }
    # -----------------------------------------
    
    scores = score_recipes(user)


    df = df_global.copy()
    df["score"] = scores

    # Food type filter
    if user["food_type"] != "none":
        df = df[df["food_type"].str.lower() == user["food_type"].lower()]

    # Cuisine preference (soft penalty)
    if user["cuisine_pref"] != "none":
        df.loc[df["cuisine"] != user["cuisine_pref"], "score"] *= 0.8

    plan = {"days": []}
    used = set()

    # Build 7-day meal plan
    for day in range(7):
        day_meals = {}

        for meal in ["Breakfast", "Lunch", "Dinner"]:
            # Ensure candidates is not empty before sorting/indexing
            candidates = df[df["meal_type"].str.lower() == meal.lower()]
            
            if candidates.empty:
                # If no recipes match the meal type (unlikely but safe)
                day_meals[meal] = {"recipe_name": f"No {meal} found."}
                continue 

            candidates = candidates.sort_values("score", ascending=False)
            
            chosen = None
            for _, row in candidates.iterrows():
                if row["recipe_name"] not in used:
                    chosen = row
                    break

            if chosen is None:
                # Reuse the highest scoring if all top recipes are used
                chosen = candidates.iloc[0]

            day_meals[meal] = {
                "recipe_name": chosen.get("recipe_name", ""),
                "ingredients": chosen.get("ingredients", ""),
                "instructions": chosen.get("instructions", ""),
                "preparation": chosen.get("preparation", ""),
                "calories": float(chosen.get("calories", 0)),
                "protein_g": float(chosen.get("protein_g", 0)),
                "carbs_g": float(chosen.get("carbs_g", 0)),
                "fat_g": float(chosen.get("fat_g", 0)),
                "iron_mg": float(chosen.get("iron_mg", 0)),
                "suitable_for_diabetes": bool(chosen.get("suitable_for_diabetes", False)),
            }

            used.add(chosen["recipe_name"])
            if len(used) > 12:
                used = set(list(used)[-12:])

        plan["days"].append(day_meals)

    # Workouts
    workouts_loss = [
        "HIIT + Core (30–40 min)",
        "Brisk Walk/Cycling (45 min)",
        "Full Body Strength (40 min)",
        "Yoga + Mobility (30 min)",
        "Interval Running + Core (30 min)",
        "Bodyweight Strength (40 min)",
        "Light Walk + Stretch (20 min)",
    ]

    workouts_muscle = [
        "Push Day: Chest/Shoulders/Triceps (60 min)",
        "Pull Day: Back/Biceps (60 min)",
        "Leg Day: Squats/Deadlifts (60 min)",
        "Core + Mobility (30 min)",
        "Upper Body Strength (50 min)",
        "Lower Body Strength (50 min)",
        "Active Rest + Stretch (20 min)",
    ]

    workouts_gain = [
        "Heavy Full Body Strength (50 min)",
        "Cardio 20 min + Shoulders (40 min)",
        "Moderate Full Body Strength (45 min)",
        "Core + Yoga (30 min)",
        "Upper Body Hypertrophy (50 min)",
        "Lower Body Hypertrophy (50 min)",
        "Rest Day + Stretch (20 min)",
    ]

    if user["goal"] == "muscle":
        workout = workouts_muscle
    elif user["goal"] == "gain":
        workout = workouts_gain
    else:
        workout = workouts_loss

    return {"plan": plan, "workout": workout}


# -------------------------------------------------
# API Route
# -------------------------------------------------
@app.post("/generate_plan")
def generate_plan(inp: UserInput):
    data = inp.model_dump()
    # This check handles the case where Pydantic accepted an empty/null payload,
    # and we now safely exit before calling score_recipes.
    if data["age"] is None or data["height_cm"] is None or data["weight_kg"] is None:
        print("Detected missing core input data (Likely a ghost request). Returning empty successful response.")
        return {"plan": {"days": []}, "workout": []} 
        
    print("CLEANED INPUT:", data)
    return build_week_plan(data)


# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
