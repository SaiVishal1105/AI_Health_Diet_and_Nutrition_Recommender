
# app.py (robust, Render-friendly, model+heuristic fallback)
import os
import logging
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Local imports (keep these in repo)
try:
    from data_processing import load_and_process
except Exception:
    load_and_process = None

from model import RecipeRanker

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("diet_recommender")

# App + CORS
app = FastAPI(title="Diet Recommender API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with your frontend domain for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths & globals
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_XLSX = os.path.join(BASE_DIR, "healthy_recipes_augmented.xlsx")
MODEL_PATH = os.path.join(BASE_DIR, "model.pt")

df_global: Optional[pd.DataFrame] = None
recipe_features: Optional[np.ndarray] = None
model: Optional[torch.nn.Module] = None

# Safe column names to use in output -- choose fallbacks if missing
DEFAULT_COLS = {
    "recipe_name": "recipe_name",
    "cuisine": "cuisine",
    "food_type": "food_type",
    "meal_type": "meal_type",
    "ingredients": "ingredients",
    # we have 'preparation' in dataset; use that for instructions fallback
    "instructions": "instructions",  # may not exist -> handled safely
    "preparation": "preparation",
    "calories": "calories",
    "protein_g": "protein_g",
    "carbs_g": "carbs_g",
    "fat_g": "fat_g",
    "iron_mg": "iron_mg",
}

# -------------------------
# Load dataset (try load_and_process() if present, else read xlsx)
# -------------------------
def load_dataset():
    global df_global, recipe_features
    try:
        if load_and_process is not None:
            logger.info("Using load_and_process() to load dataset.")
            df_global, *_ = load_and_process()
        else:
            raise RuntimeError("load_and_process not available")
    except Exception as e:
        logger.warning("load_and_process() failed or not available: %s", e)
        # Fallback: read the excel directly
        try:
            logger.info("Falling back to reading Excel at %s", DATA_XLSX)
            df_global = pd.read_excel(DATA_XLSX)
        except Exception as ee:
            logger.exception("Failed to read Excel dataset: %s", ee)
            df_global = None

    if df_global is not None:
        # ensure column names are lower-case for consistent checks
        df_global.columns = [c if not isinstance(c, str) else c for c in df_global.columns]
        # If processed 'std_' feature columns exist (created in load_and_process), use those;
        # otherwise try to derive a simple numeric feature matrix from nutrition columns.
        std_cols = [c for c in df_global.columns if isinstance(c, str) and c.startswith("std_")]
        if std_cols:
            recipe_features = df_global[std_cols].fillna(0).values.astype("float32")
            logger.info("Using std_ features (%d dims).", len(std_cols))
        else:
            # Derive features from available nutrition columns (deterministic fallback)
            fallback_cols = []
            for k in ("calories", "protein_g", "carbs_g", "fat_g", "iron_mg", "fiber_g"):
                if k in df_global.columns:
                    fallback_cols.append(k)
            if fallback_cols:
                recipe_features = df_global[fallback_cols].fillna(0).values.astype("float32")
                logger.info("Using nutrition-derived features: %s", fallback_cols)
            else:
                # last resort: use zeros
                recipe_features = np.zeros((len(df_global), 5), dtype="float32")
                logger.warning("No numeric recipe features found; using zeros.")
    else:
        recipe_features = np.zeros((1, 5), dtype="float32")


# -------------------------
# Load model safely
# -------------------------
def load_model():
    global model
    try:
        # user_dim consistent with scoring function below: 4 base + 3 goals + 4 defs + 3 chronic = 14
        user_dim = 4 + 3 + 4 + 3
        recipe_dim = recipe_features.shape[1] if recipe_features is not None else 5
        model = RecipeRanker(user_dim, recipe_dim)
        if os.path.exists(MODEL_PATH):
            logger.info("Loading model weights from %s", MODEL_PATH)
            model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
            model.eval()
            logger.info("Model loaded.")
        else:
            logger.warning("Model file not found at %s; model set to None", MODEL_PATH)
            model = None
    except Exception as e:
        logger.exception("Model init/load error: %s", e)
        model = None


# initialize
load_dataset()
load_model()

# -------------------------
# Pydantic input model (lenient)
# -------------------------
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


# -------------------------
# Helper utilities
# -------------------------
def coerce_float(v: Any, default: float) -> float:
    if v is None:
        return default
    try:
        return float(str(v).strip())
    except Exception:
        return default


def norm_str_val(raw: Dict[str, Any], k: str, default: str = "none") -> str:
    v = raw.get(k, default)
    if v is None:
        return default
    s = str(v).strip().lower()
    return s if s else default


def safe_row_get(row: pd.Series, col: str, default: Any = None):
    if row is None or col not in row.index:
        return default
    v = row.get(col, default)
    if pd.isna(v):
        return default
    return v


# -------------------------
# Heuristic fallback scorer (deterministic, depends on user input)
# -------------------------
def heuristic_score(user: Dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    # Use recipe nutrition to compute a sensible score per user
    # Higher is better
    n = len(df)
    scores = np.zeros(n, dtype=float)

    # Unpack user
    calorie_target = user.get("calorie_target") or 1800
    goal = user.get("goal", "loss")
    deficiency = user.get("deficiency", "none")
    chronic = user.get("chronic", "none")
    cuisine_pref = user.get("cuisine_pref", "none")
    food_type = user.get("food_type", "none")

    # Use columns if present
    calories = df.get("calories", pd.Series([0] * n)).fillna(0).astype(float).values
    protein = df.get("protein_g", pd.Series([0] * n)).fillna(0).astype(float).values
    carbs = df.get("carbs_g", pd.Series([0] * n)).fillna(0).astype(float).values
    fat = df.get("fat_g", pd.Series([0] * n)).fillna(0).astype(float).values
    iron = df.get("iron_mg", pd.Series([0] * n)).fillna(0).astype(float).values

    # base calorie closeness: smaller distance to target -> higher score
    cal_score = 1 - (np.abs(calories - calorie_target) / max(1.0, calorie_target))
    scores += cal_score

    # protein preference for muscle/gain
    if goal == "muscle":
        scores += protein * 0.02
    elif goal == "gain":
        scores += protein * 0.01
    else:  # loss
        # prefer lower calories and moderate protein
        scores += (-calories / 4000.0) + protein * 0.01

    # deficiency bonuses
    if deficiency == "iron":
        scores += iron * 0.05
    if deficiency == "protein":
        scores += protein * 0.03

    # chronic penalties
    if chronic == "diabetes":
        scores -= (carbs / 500.0)  # penalize high carb
    if chronic == "hypertension":
        if "sodium_mg" in df.columns:
            scores -= df["sodium_mg"].fillna(0).astype(float).values / 10000.0

    # cuisine & food type adjustments
    if cuisine_pref and cuisine_pref != "none" and "cuisine" in df.columns:
        scores += (df["cuisine"].str.lower() == cuisine_pref).astype(float) * 0.5
    if food_type and food_type != "none" and "food_type" in df.columns:
        scores += (df["food_type"].str.lower() == food_type).astype(float) * 0.4

    # normalize
    if np.std(scores) > 0:
        scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-9)

    return scores


# -------------------------
# model-based scoring with variance check
# -------------------------
def model_score_or_fallback(user: Dict[str, Any]) -> np.ndarray:
    global model, recipe_features, df_global

    if recipe_features is None or len(recipe_features) == 0:
        return np.zeros(len(df_global) if df_global is not None else 1)

    # Build user vector consistent with model
    bmi = user["weight_kg"] / ((user["height_cm"] / 100) ** 2 + 1e-6)
    base_user = [
        user["age"] / 100.0,
        bmi / 50.0,
        user["activity_level"] / 2.0,
        (user.get("calorie_target") or 1800) / 4000.0,
    ]

    goals = ["loss", "gain", "muscle"]
    defs_ = ["none", "iron", "vitd", "protein"]
    chs = ["none", "diabetes", "hypertension"]

    full_user_vec = base_user + \
        [1.0 if user.get("goal") == g else 0.0 for g in goals] + \
        [1.0 if user.get("deficiency") == d else 0.0 for d in defs_] + \
        [1.0 if user.get("chronic") == c else 0.0 for c in chs]

    try:
        if model is None:
            raise RuntimeError("model not loaded")

        uv = torch.tensor(full_user_vec, dtype=torch.float32).unsqueeze(0)
        uv = uv.repeat(len(recipe_features), 1)
        rx = torch.tensor(recipe_features, dtype=torch.float32)

        with torch.no_grad():
            raw_scores = model(uv, rx).cpu().numpy().reshape(-1)

        # if model output variance is too low, treat as unreliable
        if np.var(raw_scores) < 1e-8:
            logger.warning("Model output variance too low (%.6g) — using heuristic fallback", np.var(raw_scores))
            return heuristic_score(user, df_global)

        # normalize model scores to [0,1]
        s = raw_scores.copy()
        s = (s - s.min()) / (s.max() - s.min() + 1e-9)

        return s

    except Exception as e:
        logger.warning("Model scoring failed or unavailable: %s — using heuristic fallback", e)
        return heuristic_score(user, df_global)


# -------------------------
# Build week plan
# -------------------------
def build_week_plan(user_input: Dict[str, Any]) -> Dict[str, Any]:
    if df_global is None:
        raise RuntimeError("Dataset not loaded on server.")

    user = user_input.copy()
    # ensure normalized strings
    for k in ("goal", "deficiency", "chronic", "cuisine_pref", "food_type"):
        user[k] = str(user.get(k, "none") or "none").strip().lower()

    # scoring
    scores = model_score_or_fallback(user)
    # final tiny noise to break ties in sorting deterministically
    scores = scores + np.random.normal(0, 1e-6, size=scores.shape)

    df = df_global.copy()
    df["score"] = scores

    # apply food_type hard filter if specified
    if user.get("food_type") and user["food_type"] != "none" and "food_type" in df.columns:
        df = df[df["food_type"].str.lower() == user["food_type"]]

    # soft cuisine penalty handled in heuristic/model adjustments already

    plan = {"days": []}
    used = set()

    for day in range(7):
        day_meals = {}
        for meal in ("Breakfast", "Lunch", "Dinner"):
            candidates = df[df["meal_type"].str.lower() == meal.lower()].sort_values("score", ascending=False)
            chosen = None
            for _, row in candidates.iterrows():
                rn = safe_row_get(row, DEFAULT_COLS["recipe_name"], "")
                if rn and rn not in used:
                    chosen = row
                    break
            if chosen is None and not candidates.empty:
                chosen = candidates.iloc[0]

            # extract safe values
            recipe_name = safe_row_get(chosen, DEFAULT_COLS["recipe_name"], "") if chosen is not None else ""
            ingredients = safe_row_get(chosen, DEFAULT_COLS["ingredients"], "")
            # prefer 'instructions' if exists, else 'preparation'
            instructions = safe_row_get(chosen, "instructions", None)
            if instructions is None:
                instructions = safe_row_get(chosen, "preparation", "Instructions not provided.")
            preparation = safe_row_get(chosen, "preparation", "")
            calories = float(safe_row_get(chosen, "calories", 0) or 0)
            protein_g = float(safe_row_get(chosen, "protein_g", 0) or 0)
            carbs_g = float(safe_row_get(chosen, "carbs_g", 0) or 0)
            fat_g = float(safe_row_get(chosen, "fat_g", 0) or 0)
            iron_mg = float(safe_row_get(chosen, "iron_mg", 0) or 0)
            suitable_for_diabetes = bool(safe_row_get(chosen, "suitable_for_diabetes", False) or False)

            day_meals[meal] = {
                "recipe_name": recipe_name,
                "ingredients": ingredients,
                "instructions": instructions,
                "preparation": preparation,
                "calories": calories,
                "protein_g": protein_g,
                "carbs_g": carbs_g,
                "fat_g": fat_g,
                "iron_mg": iron_mg,
                "suitable_for_diabetes": suitable_for_diabetes,
            }

            if recipe_name:
                used.add(recipe_name)
                if len(used) > 20:
                    used = set(list(used)[-20:])

        plan["days"].append(day_meals)

    # workouts as before (simple mapping)
   workouts = {
        "loss": [
            "HIIT + Core (30–40 min)",
            "Brisk Walk/Cycling (45 min)",
            "Full Body Strength (40 min)",
            "Yoga + Mobility (30 min)",
            "Interval Running + Core (30 min)",
            "Bodyweight Strength (40 min)",
            "Light Walk + Stretch (20 min)",
        ],
        "muscle": [
            "Push Day: Chest/Shoulders/Triceps (60 min)",
            "Pull Day: Back/Biceps (60 min)",
            "Leg Day: Squats/Deadlifts (60 min)",
            "Core + Mobility (30 min)",
            "Upper Body Strength (50 min)",
            "Lower Body Strength (50 min)",
            "Active Rest + Stretch (20 min)",
        ],
        "gain": [
            "Heavy Full Body Strength (50 min)",
            "Moderate Full Body Strength (45 min)",
            "Core + Yoga (30 min)",
            "Upper Body Hypertrophy (50 min)",
            "Lower Body Hypertrophy (50 min)",
            "Light Cardio (20 min)",
            "Rest Day + Stretch (20 min)",
        ],
    }

    return {"plan": plan, "workout": workouts.get(user["goal"], workouts["loss"])}


# -------------------------
# Endpoints
# -------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "data_loaded": df_global is not None,
        "n_recipes": int(len(df_global)) if df_global is not None else 0,
        "model_loaded": model is not None,
    }


@app.post("/generate_plan")
def generate_plan(inp: UserInput):
    # support pydantic v1/v2
    try:
        raw = inp.model_dump() if hasattr(inp, "model_dump") else inp.dict()
    except Exception:
        raw = dict(inp) if isinstance(inp, dict) else {}

    # coerce numeric fields
    raw_parsed = {
        "age": coerce_float(raw.get("age"), 23),
        "height_cm": coerce_float(raw.get("height_cm"), 170),
        "weight_kg": coerce_float(raw.get("weight_kg"), 70),
        "activity_level": coerce_float(raw.get("activity_level"), 1.55),
        "goal": raw.get("goal", "loss"),
        "deficiency": raw.get("deficiency", "none"),
        "chronic": raw.get("chronic", "none"),
        "cuisine_pref": raw.get("cuisine_pref", "none"),
        "food_type": raw.get("food_type", "none"),
        "calorie_target": (coerce_float(raw.get("calorie_target"), None)
                           if raw.get("calorie_target") not in (None, "", "None") else None),
    }

    logger.info("CLEANED INPUT: %s", raw_parsed)

    if df_global is None:
        raise HTTPException(status_code=503, detail="Server dataset not loaded.")

    try:
        plan = build_week_plan(raw_parsed)
        return plan
    except Exception as e:
        logger.exception("Error building plan: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------
# Local run
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
