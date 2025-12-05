import os
import logging
from typing import Optional, Union, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import uvicorn

# local modules
from data_processing import load_and_process
from model import RecipeRanker

# -----------------------
# Logging
# -----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("diet_recommender")

# -----------------------
# FastAPI App
# -----------------------
app = FastAPI(title="Diet Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Globals
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df_global = None
recipe_features = None
model = None

# FIXED: user_dim MUST MATCH MODEL INPUT
# 3 numeric + 3 goal + 4 deficiency + 3 chronic = 13
user_dim = 13

MODEL_PATH = os.path.join(BASE_DIR, "model.pt")

# -----------------------
# Helpers
# -----------------------
def coerce_float(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        return float(str(v).strip())
    except:
        return default

def norm_str(raw, key, default="none"):
    v = raw.get(key, default)
    if v is None:
        return default
    s = str(v).strip().lower()
    return s if s else default

def safe_assign_scores(df, scores):
    if scores.ndim != 1:
        scores = scores.reshape(-1)
    if len(scores) != len(df):
        raise ValueError("Length mismatch score vs df")
    df["score"] = scores

# -----------------------
# Pydantic Input
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
# Startup: Load Data + Model
# -----------------------
@app.on_event("startup")
def startup():
    global df_global, recipe_features, model

    # Load data
    try:
        logger.info("Loading data...")
        df_global, enc_global, scaler_global, num_cols = load_and_process()
        feat_cols = [c for c in df_global.columns if c.startswith("std_")]
        recipe_features = df_global[feat_cols].values.astype("float32")
        logger.info("Data loaded: %d recipes, %d features", len(df_global), recipe_features.shape[1])
    except Exception as e:
        logger.exception("Data load failed: %s", e)
        df_global = None
        recipe_features = None

    # Model
    try:
        recipe_dim = recipe_features.shape[1]
        model = RecipeRanker(user_dim, recipe_dim)
        if os.path.exists(MODEL_PATH):
            logger.info("Loading model from %s", MODEL_PATH)
            model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
            model.eval()
            logger.info("Model loaded successfully")
        else:
            logger.error("Model file missing: %s", MODEL_PATH)
            model = None
    except Exception as e:
        logger.exception("Model init failed: %s", e)
        model = None

# -----------------------
# Build Correct User Vector (13 dims)
# -----------------------
def make_user_vector(user: Dict[str, Any]) -> np.ndarray:
    age_norm = user["age"] / 100.0
    bmi = user["weight_kg"] / ((user["height_cm"] / 100.0) ** 2 + 1e-6)
    bmi_norm = bmi / 50.0
    act_norm = user["activity_level"] / 2.0

    goals = ["loss", "gain", "muscle"]
    defs_ = ["none", "iron", "vitd", "protein"]
    chs = ["none", "diabetes", "hypertension"]

    vec = [
        age_norm,
        bmi_norm,
        act_norm,
        *[1.0 if user["goal"] == g else 0.0 for g in goals],
        *[1.0 if user["deficiency"] == d else 0.0 for d in defs_],
        *[1.0 if user["chronic"] == c else 0.0 for c in chs],
    ]

    vec = np.asarray(vec, dtype="float32")
    if vec.shape[0] != user_dim:
        raise RuntimeError(f"User vector dim {vec.shape[0]} != expected {user_dim}")

    return vec

# -----------------------
# Scoring
# -----------------------
def score_recipes(user):
    global model, recipe_features

    uv = make_user_vector(user)
    n = len(recipe_features)

    try:
        uv_batch = torch.tensor(uv).float().unsqueeze(0).repeat(n, 1)
        rx = torch.tensor(recipe_features).float()
        with torch.no_grad():
            out = model(uv_batch, rx)
        scores = out.numpy().reshape(-1)
        return scores
    except Exception:
        logger.exception("Model scoring failed: using fallback.")

    # fallback dot product scoring
    uv_norm = uv / (np.linalg.norm(uv) + 1e-9)
    rx_norm = recipe_features / (np.linalg.norm(recipe_features, axis=1, keepdims=True) + 1e-9)
    scores = (rx_norm @ uv_norm)
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    return scores.astype("float32")

# -----------------------
# Weekly Plan
# -----------------------
def build_week_plan(user_in):
    global df_global

    user = dict(user_in)
    scores = score_recipes(user)

    df = df_global.copy()
    safe_assign_scores(df, scores)

    # Filters
    if user["food_type"] != "none":
        df = df[df["food_type"].str.lower() == user["food_type"]]

    if user["cuisine_pref"] != "none":
        df.loc[df["cuisine"] != user["cuisine_pref"], "score"] *= 0.8

    plan = {"days": []}
    used = set()

    for _ in range(7):
        day_meals = {}
        for meal in ["Breakfast", "Lunch", "Dinner"]:
            sub = df[df["meal_type"].str.lower() == meal.lower()].sort_values("score", ascending=False)
            chosen = None

            for _, row in sub.iterrows():
                rn = row["recipe_name"]
                if rn not in used:
                    chosen = row
                    break
            if chosen is None and not sub.empty:
                chosen = sub.iloc[0]

            if chosen is None:
                day_meals[meal] = {}
                continue

            rn = chosen["recipe_name"]
            used.add(rn)

            day_meals[meal] = {
                "recipe_name": chosen["recipe_name"],
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

        plan["days"].append(day_meals)

    workouts = {
        "loss": [
            "HIIT + Core (30–40 min)", "Brisk Walk (45 min)", "Full Body Strength (40 min)",
            "Yoga (30 min)", "Interval Running (30 min)", "Bodyweight Strength (40 min)", "Light Walk (20 min)"
        ],
        "muscle": [
            "Push Day (60 min)", "Pull Day (60 min)", "Leg Day (60 min)", "Core (30 min)",
            "Upper Strength (50 min)", "Lower Strength (50 min)", "Stretching (20 min)"
        ],
        "gain": [
            "Full Body Strength (50 min)", "Cardio + Shoulders (40 min)", "Strength (45 min)",
            "Yoga (30 min)", "Upper Hypertrophy (50 min)", "Lower Hypertrophy (50 min)", "Rest + Stretch (20 min)"
        ]
    }

    goal = user["goal"]
    return {"plan": plan, "workout": workouts.get(goal, workouts["loss"])}

# -----------------------
# Endpoints
# -----------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "data_loaded": df_global is not None,
        "model_loaded": model is not None,
        "recipes": len(df_global) if df_global is not None else 0,
        "recipe_dim": recipe_features.shape[1] if recipe_features is not None else 0,
    }

@app.post("/generate_plan")
def generate_plan(inp: UserInput):
    raw = inp.model_dump()

    user = {
        "age": coerce_float(raw["age"]),
        "height_cm": coerce_float(raw["height_cm"]),
        "weight_kg": coerce_float(raw["weight_kg"]),
        "activity_level": coerce_float(raw["activity_level"]),
        "goal": norm_str(raw, "goal", "loss"),
        "deficiency": norm_str(raw, "deficiency", "none"),
        "chronic": norm_str(raw, "chronic", "none"),
        "cuisine_pref": norm_str(raw, "cuisine_pref", "none"),
        "food_type": norm_str(raw, "food_type", "none"),
        "calorie_target": coerce_float(raw.get("calorie_target", None))
    }

    try:
        return build_week_plan(user)
    except Exception as e:
        logger.exception("Plan generation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------
# Local Run
# -----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
