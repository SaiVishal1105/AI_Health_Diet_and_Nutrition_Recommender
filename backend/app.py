# app.py (updated, robust, Render-friendly)
import os
import logging
from typing import Optional, Union, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import uvicorn

# Local imports (ensure these modules and data files are present in repo)
from data_processing import load_and_process
from model import RecipeRanker

# -----------------------
# Logging
# -----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("diet_recommender")

# -----------------------
# FastAPI app & CORS
# -----------------------
app = FastAPI(title="Diet Recommender API")

# For development keep "*", in production replace with your Vercel domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Resolve paths & globals
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df_global = None
recipe_features = np.zeros((1, 5), dtype="float32")
model = None

# -----------------------
# Load data
# -----------------------
try:
    logger.info("Loading data...")
    df_global, enc_global, scaler_global, num_cols = load_and_process()
    logger.info("Data loaded successfully.")
    recipe_features = df_global[[c for c in df_global.columns if c.startswith("std_")]].values.astype("float32")
except Exception as e:
    logger.exception("DATA LOAD ERROR: %s", e)
    df_global = None
    recipe_features = np.zeros((1, 5), dtype="float32")

# -----------------------
# Load model
# -----------------------
MODEL_PATH = os.path.join(BASE_DIR, "model.pt")
user_dim = 3 + 3 + 4 + 3
recipe_dim = recipe_features.shape[1] if recipe_features is not None else 5

try:
    logger.info("Initializing model architecture (no weights yet).")
    model = RecipeRanker(user_dim, recipe_dim)
    if os.path.exists(MODEL_PATH):
        logger.info("Loading model weights from %s", MODEL_PATH)
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
        logger.info("Model loaded successfully.")
    else:
        logger.warning("MODEL FILE NOT FOUND: %s", MODEL_PATH)
        model = None
except Exception as e:
    logger.exception("MODEL LOAD ERROR: %s", e)
    model = None

# -----------------------
# Pydantic input model (lenient defaults)
# Accept strings OR numbers (coercion done downstream)
# -----------------------
class UserInput(BaseModel):
    age: Optional[Union[int, float, str]] = 23
    height_cm: Optional[Union[int, float, str]] = 170
    weight_kg: Optional[Union[int, float, str]] = 70
    activity_level: Optional[Union[int, float, str]] = 1.55
    goal: Optional[str] = "loss"
    deficiency: Optional[str] = "none"
    chronic: Optional[str] = "none"
    cuisine_pref: Optional[str] = "none"
    food_type: Optional[str] = "none"
    calorie_target: Optional[Union[int, float, str]] = None

# -----------------------
# Helpers
# -----------------------
def coerce_float(v: Any, default: float = 0.0) -> float:
    """Try to coerce v (which might be str, int, float, None) -> float safely."""
    if v is None:
        return default
    if isinstance(v, (int, float, np.number)):
        return float(v)
    try:
        s = str(v).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default

def coerce_int(v: Any, default: int = 0) -> int:
    try:
        return int(coerce_float(v, default))
    except Exception:
        return default

def clean_user_input(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Return a sanitized dictionary with proper types and defaults."""
    u = {}
    u["age"] = coerce_float(raw.get("age", 23), 23)
    u["height_cm"] = coerce_float(raw.get("height_cm", 170), 170)
    u["weight_kg"] = coerce_float(raw.get("weight_kg", 70), 70)
    u["activity_level"] = coerce_float(raw.get("activity_level", 1.55), 1.55)

    # Strings: normalize
    def norm_str(k, default="none"):
        val = raw.get(k, default)
        if val is None:
            return default
        s = str(val).strip()
        if s == "":
            return default
        return s.lower()

    u["goal"] = norm_str("goal", "loss")
    # keep original casing for comparisons below (we lower-case later)
    u["deficiency"] = norm_str("deficiency", "none")
    u["chronic"] = norm_str("chronic", "none")
    u["cuisine_pref"] = norm_str("cuisine_pref", "none")
    u["food_type"] = norm_str("food_type", "none")

    u["calorie_target"] = None
    if raw.get("calorie_target") is not None and raw.get("calorie_target") != "":
        u["calorie_target"] = coerce_float(raw.get("calorie_target"), None)

    return u

# -----------------------
# Scoring and plan building (adapted from your original logic)
# -----------------------
def score_recipes(user: Dict[str, Any]) -> np.ndarray:
    if recipe_features is None or recipe_features.size == 0:
        return np.array([])

    bmi = user["weight_kg"] / ((user["height_cm"] / 100) ** 2 + 1e-6)

    user_vec = [
        user["age"] / 100.0,
        bmi / 50.0,
        user["activity_level"] / 2.0,
    ]

    goals = ["loss", "gain", "muscle"]
    defs_ = ["none", "iron", "vitd", "protein"]
    chs = ["none", "diabetes", "hypertension"]

    user_vec += [1.0 if user["goal"] == g else 0.0 for g in goals]
    user_vec += [1.0 if user["deficiency"] == d else 0.0 for d in defs_]
    user_vec += [1.0 if user["chronic"] == c else 0.0 for c in chs]

    uv = torch.tensor(user_vec, dtype=torch.float32).unsqueeze(0)
    uv = uv.repeat(len(recipe_features), 1)
    rx = torch.tensor(recipe_features, dtype=torch.float32)

    with torch.no_grad():
        if model is None:
            logger.warning("Model is not loaded; returning random scores as fallback.")
            return np.random.rand(len(recipe_features))
        try:
            scores = model(uv, rx).numpy().squeeze()
            # ensure 1D numpy array
            return np.asarray(scores).reshape(-1)
        except Exception:
            logger.exception("Model scoring failed; using random scores.")
            return np.random.rand(len(recipe_features))

def build_week_plan(user_input: Dict[str, Any]) -> Dict[str, Any]:
    if df_global is None:
        raise RuntimeError("Server data not loaded.")
    user = dict(user_input)

    # Normalize "none"-like values already done in clean_user_input
    scores = score_recipes(user)
    if scores.size == 0:
        raise RuntimeError("No recipe features available on server.")

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
            candidates = df[df["meal_type"].str.lower() == meal.lower()].sort_values("score", ascending=False)

            if candidates.empty:
                chosen = {}
            else:
                chosen = None
                for _, row in candidates.iterrows():
                    rn = row.get("recipe_name", "")
                    if rn not in used:
                        chosen = row
                        break
                if chosen is None:
                    chosen = candidates.iloc[0]

            # safe extraction (handles chosen being empty dict)
            def g(k, default=None):
                try:
                    return chosen.get(k, default)
                except Exception:
                    try:
                        return chosen[k]
                    except Exception:
                        return default

            day_meals[meal] = {
                "recipe_name": g("recipe_name", ""),
                "ingredients": g("ingredients", ""),
                "instructions": g("instructions", ""),
                "preparation": g("preparation", ""),
                "calories": float(g("calories", 0) or 0),
                "protein_g": float(g("protein_g", 0) or 0),
                "carbs_g": float(g("carbs_g", 0) or 0),
                "fat_g": float(g("fat_g", 0) or 0),
                "iron_mg": float(g("iron_mg", 0) or 0),
                "suitable_for_diabetes": bool(g("suitable_for_diabetes", False)),
            }

            rn = g("recipe_name", "")
            if rn:
                used.add(rn)
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

    if user.get("goal") == "muscle":
        workout = workouts_muscle
    elif user.get("goal") == "gain":
        workout = workouts_gain
    else:
        workout = workouts_loss

    return {"plan": plan, "workout": workout}

# -----------------------
# Endpoints
# -----------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "data_loaded": df_global is not None,
        "model_loaded": model is not None,
    }

@app.post("/generate_plan")
def generate_plan(inp: UserInput):
    # Backwards-compatible extraction for pydantic v1/v2 objects
    raw = None
    try:
        if hasattr(inp, "model_dump"):
            raw = inp.model_dump()
        elif hasattr(inp, "dict"):
            raw = inp.dict()
        else:
            raw = dict(inp)
    except Exception:
        # fallback: try to use object as mapping
        try:
            raw = dict(inp)
        except Exception:
            raw = {}

    user = clean_user_input(raw)
    logger.info("CLEANED INPUT: %s", user)

    if df_global is None:
        raise HTTPException(status_code=503, detail="Server data not available.")
    try:
        return build_week_plan(user)
    except RuntimeError as e:
        logger.exception("Error building plan: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error.")

# -----------------------
# Local run helper (Render uses start command)
# -----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
