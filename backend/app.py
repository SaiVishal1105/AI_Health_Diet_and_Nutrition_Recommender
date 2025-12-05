# app.py
import os
import logging
from typing import Optional, Union, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import uvicorn

# local modules - keep these in repo
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Globals will be initialized on startup()
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df_global = None
recipe_features = None    # numpy array shape (n_recipes, n_features)
model = None
user_dim = 3 + 3 + 4 + 3  # keep consistent with your model
MODEL_PATH = os.path.join(BASE_DIR, "model.pt")

# -----------------------
# Helper utilities
# -----------------------
def coerce_float(v: Any, default: float = 0.0) -> float:
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

def norm_str_value(raw: Dict[str, Any], key: str, default: str = "none") -> str:
    v = raw.get(key, default)
    if v is None:
        return default
    s = str(v).strip()
    return s.lower() if s != "" else default

def safe_assign_scores(df, scores: np.ndarray):
    """Safely assign scores array to df as column 'score'. Prevent accidental broadcasting."""
    if scores is None:
        raise ValueError("scores is None")
    if scores.ndim != 1:
        scores = np.asarray(scores).reshape(-1)
    if len(scores) == len(df):
        df["score"] = scores
    elif len(scores) == 1:
        df["score"] = float(scores[0])
    else:
        # do not silently broadcast / truncate; raise to surface bug
        raise ValueError(f"Length mismatch between scores ({len(scores)}) and df ({len(df)})")

# -----------------------
# Pydantic model
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
# Startup: load data & model once
# -----------------------
@app.on_event("startup")
def startup():
    global df_global, recipe_features, model, user_dim
    # Load data
    try:
        logger.info("Startup: loading data...")
        df_global, enc_global, scaler_global, num_cols = load_and_process()
        # find standardized recipe features (same logic as you used locally)
        feat_cols = [c for c in df_global.columns if c.startswith("std_")]
        recipe_features = df_global[feat_cols].values.astype("float32")
        logger.info("Loaded data: %d rows, %d recipe features", len(df_global), recipe_features.shape[1])
    except Exception as e:
        logger.exception("DATA LOAD FAILED: %s", e)
        df_global = None
        recipe_features = None

    # Validate shapes
    if df_global is not None and recipe_features is not None:
        if len(recipe_features) != len(df_global):
            # This is a critical mismatch; surface it immediately.
            msg = f"recipe_features length {len(recipe_features)} != df rows {len(df_global)}"
            logger.error(msg)
            # Null out to prevent silent wrong behavior
            df_global = None
            recipe_features = None

    # Load model architecture & weights (map_location cpu for Render)
    try:
        logger.info("Startup: initializing model architecture")
        # compute recipe_dim from recipe_features if available
        recipe_dim = recipe_features.shape[1] if (recipe_features is not None) else 5
        model = RecipeRanker(user_dim, recipe_dim)
        if os.path.exists(MODEL_PATH):
            logger.info("Loading model weights from %s", MODEL_PATH)
            model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
            model.eval()
            logger.info("Model loaded")
        else:
            logger.warning("Model file not found at %s; model will be None for scoring fallback.", MODEL_PATH)
            model = None
    except Exception as e:
        logger.exception("Model init/load failed: %s", e)
        model = None

# -----------------------
# Scoring: deterministic with stable fallback
# -----------------------
def make_user_vector(user: Dict[str, Any]) -> np.ndarray:
    """Return user vector matching model input ordering as numpy array (float32)."""
    bmi = user["weight_kg"] / ((user["height_cm"] / 100) ** 2 + 1e-6)

    user_vec = [
        user["age"] / 100.0,
        bmi / 50.0,
        user["activity_level"] / 2.0
    ]

    goals = ["loss", "gain", "muscle"]
    defs_ = ["none", "iron", "vitd", "protein"]
    chs = ["none", "diabetes", "hypertension"]

    user_vec += [1.0 if user["goal"] == g else 0.0 for g in goals]
    user_vec += [1.0 if user["deficiency"] == d else 0.0 for d in defs_]
    user_vec += [1.0 if user["chronic"] == c else 0.0 for c in chs]

    return np.asarray(user_vec, dtype="float32")

def score_recipes(user: Dict[str, Any]) -> np.ndarray:
    """Return 1D numpy array of scores with length == number of recipes."""
    global model, recipe_features
    if recipe_features is None or recipe_features.size == 0:
        raise RuntimeError("Server recipe features not available.")

    uv = make_user_vector(user)  # shape (user_dim,)
    n = len(recipe_features)

    # Torch/model path (preferred)
    try:
        if model is not None:
            # build batch user matrix: (n, user_dim)
            uv_t = torch.tensor(uv, dtype=torch.float32).unsqueeze(0).repeat(n, 1)
            rx_t = torch.tensor(recipe_features, dtype=torch.float32)
            with torch.no_grad():
                out = model(uv_t, rx_t)
            # safe conversion
            scores = out.cpu().numpy().reshape(-1)
            if scores.shape[0] != n:
                raise ValueError("Model returned scores of wrong length.")
            return scores
    except Exception:
        logger.exception("Model scoring failed; falling back to deterministic heuristic.")

    # Deterministic fallback scoring (no randomness)
    # Approach: normalize user vector and recipe_features then compute dot-product (cosine-like)
    try:
        # ensure 2D
        rx = np.asarray(recipe_features, dtype="float32")
        # If recipe features and user vec dims mismatch, try to adapt: if rx.shape[1] > uv.size use projection
        if rx.shape[1] == uv.size:
            # cosine similarity
            uv_norm = uv / (np.linalg.norm(uv) + 1e-9)
            rx_norm = rx / (np.linalg.norm(rx, axis=1, keepdims=True) + 1e-9)
            scores = (rx_norm @ uv_norm).astype("float32").reshape(-1)
        else:
            # simple dot with broadcast trimming or padding
            minDim = min(rx.shape[1], uv.size)
            scores = (rx[:, :minDim] @ uv[:minDim]).astype("float32").reshape(-1)
        # scale into [0,1] for stability
        if np.all(np.isfinite(scores)):
            smin, smax = scores.min(), scores.max()
            if smax - smin > 1e-6:
                scores = (scores - smin) / (smax - smin)
            else:
                scores = np.zeros_like(scores)
        else:
            raise ValueError("Non-finite scores")
        return scores
    except Exception:
        logger.exception("Fallback deterministic scoring also failed.")
        # As a last resort return zeros (not random) so system is deterministic
        return np.zeros(n, dtype="float32")

# -----------------------
# Build week plan (same logic but defensive)
# -----------------------
def build_week_plan(user_input: Dict[str, Any]) -> Dict[str, Any]:
    if df_global is None:
        raise RuntimeError("Server data not loaded.")
    user = dict(user_input)

    scores = score_recipes(user)
    if scores is None or len(scores) == 0:
        raise RuntimeError("Scoring produced no outputs.")

    df = df_global.copy()
    # safe assign
    safe_assign_scores(df, scores)

    # Food type filter
    if user["food_type"] != "none":
        df = df[df["food_type"].str.lower() == user["food_type"].lower()]

    # Cuisine preference (soft penalty)
    if user["cuisine_pref"] != "none":
        df.loc[df["cuisine"] != user["cuisine_pref"], "score"] *= 0.8

    plan = {"days": []}
    used = set()

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

    # Workouts - same as previous
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
        "n_recipes": len(df_global) if df_global is not None else 0,
        "recipe_feature_dim": recipe_features.shape[1] if (recipe_features is not None) else 0,
    }

@app.post("/generate_plan")
def generate_plan(inp: UserInput):
    # pydantic compatibility extraction
    raw = {}
    try:
        if hasattr(inp, "model_dump"):
            raw = inp.model_dump()
        elif hasattr(inp, "dict"):
            raw = inp.dict()
        else:
            raw = dict(inp)
    except Exception:
        try:
            raw = dict(inp)
        except Exception:
            raw = {}

    user = {
        "age": coerce_float(raw.get("age", 23), 23),
        "height_cm": coerce_float(raw.get("height_cm", 170), 170),
        "weight_kg": coerce_float(raw.get("weight_kg", 70), 70),
        "activity_level": coerce_float(raw.get("activity_level", 1.55), 1.55),
        "goal": norm_str_value(raw, "goal", "loss"),
        "deficiency": norm_str_value(raw, "deficiency", "none"),
        "chronic": norm_str_value(raw, "chronic", "none"),
        "cuisine_pref": norm_str_value(raw, "cuisine_pref", "none"),
        "food_type": norm_str_value(raw, "food_type", "none"),
        "calorie_target": coerce_float(raw.get("calorie_target"), None) if raw.get("calorie_target") not in (None, "") else None,
    }

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
# Run locally (Render uses start command)
# -----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info")

