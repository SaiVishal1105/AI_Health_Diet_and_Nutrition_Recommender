from fastapi import Request, HTTPException

# -------------------------
# Helper: validate + normalize
# -------------------------
def validate_and_normalize_input(data: dict) -> dict:
    """
    Ensure all required fields exist and coerce them to proper types.
    Returns normalized dict or raises HTTPException(400) if required field missing/invalid.
    """
    # required fields and defaults
    expected = {
        "age": 23,
        "height_cm": 170.0,
        "weight_kg": 70.0,
        "activity_level": 1.55,
        "goal": "none",
        "deficiency": "none",
        "chronic": "none",
        "cuisine_pref": "none",
        "food_type": "none",
        "calorie_target": None,
    }

    normalized = {}

    # If body is not a dict, fail early
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # Coerce numeric fields safely
    def to_float(x, default):
        if x is None or x == "":
            return default
        try:
            return float(x)
        except Exception:
            return default

    def to_int(x, default):
        if x is None or x == "":
            return default
        try:
            return int(float(x))
        except Exception:
            return default

    # Age (int)
    normalized["age"] = to_int(data.get("age", expected["age"]), expected["age"])

    # Height/weight/activity (floats)
    normalized["height_cm"] = to_float(data.get("height_cm", expected["height_cm"]), expected["height_cm"])
    normalized["weight_kg"] = to_float(data.get("weight_kg", expected["weight_kg"]), expected["weight_kg"])
    normalized["activity_level"] = to_float(data.get("activity_level", expected["activity_level"]), expected["activity_level"])

    # Strings (with safe defaults)
    def clean_str_field(key):
        v = data.get(key, expected[key])
        if v is None:
            return expected[key]
        v = str(v).strip()
        if v == "" or v.lower() == "none":
            return expected[key]
        return v

    normalized["goal"] = clean_str_field("goal")
    normalized["deficiency"] = clean_str_field("deficiency")
    normalized["chronic"] = clean_str_field("chronic")
    normalized["cuisine_pref"] = clean_str_field("cuisine_pref")
    normalized["food_type"] = clean_str_field("food_type")

    # calorie_target (optional float)
    ct = data.get("calorie_target", expected["calorie_target"])
    if ct is None or ct == "":
        normalized["calorie_target"] = None
    else:
        try:
            normalized["calorie_target"] = float(ct)
        except Exception:
            normalized["calorie_target"] = None

    # Basic sanity checks (optional)
    if normalized["age"] <= 0 or normalized["height_cm"] <= 0 or normalized["weight_kg"] <= 0:
        raise HTTPException(status_code=400, detail="age/height/weight must be positive numbers")

    return normalized


# -------------------------
# New route: accept raw JSON + validate
# -------------------------
@app.post("/generate_plan")
async def generate_plan(request: Request):
    """
    Accepts any JSON payload, normalizes it, and then runs plan generation.
    This avoids intermittent 422s caused by strict Pydantic validation.
    """
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # Normalize & coerce types
    try:
        data = validate_and_normalize_input(payload)
    except HTTPException as e:
        # Bad request from client
        raise e
    except Exception as e:
        # Unexpected server-side error during validation
        raise HTTPException(status_code=500, detail=str(e))

    # For debugging: log the cleaned input (you already used print)
    print("CLEANED INPUT:", data)

    # Ensure df_global loaded
    if df_global is None:
        raise HTTPException(status_code=503, detail="Server data not available")

    # Build plan (wrap errors)
    try:
        return build_week_plan(data)
    except Exception as e:
        # Log exception and return 500
        print("PLAN BUILD ERROR:", e)
        raise HTTPException(status_code=500, detail="Failed to build plan")
