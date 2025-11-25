# AI Diet & Nutrition Recommender (Deep Learning)
This project implements a simple, extendable AI-backed weekly diet & workout recommender tailored to the provided recipe dataset.

## What's included
- `backend/` — FastAPI backend with data preprocessing, a simple PyTorch model, training script, and endpoints for inference.
- `frontend/` — Minimal React app (Vite-compatible) that posts user inputs and displays a generated 7-day diet + workout plan.
- The backend uses the dataset at: **/mnt/data/healthy_recipes_augmented.xlsx** (already uploaded).

## Quick Start (VS Code)
1. Open two terminals in VS Code.
2. Backend:
   - `cd backend`
   - Create a virtualenv: `python -m venv .venv && source .venv/bin/activate` (on Windows: `.\.venv\Scripts\activate`)
   - `pip install -r requirements.txt`
   - Train a quick model (optional, creates `model.pt`): `python train.py`
   - Run server: `uvicorn app:app --reload --port 8000`
3. Frontend:
   - `cd frontend`
   - `npm install`
   - `npm run dev` (or `npm start` if using CRA; this project uses a simple setup — see package.json)
4. Open the frontend in your browser (usually http://localhost:5173) and test.

## Notes / Next steps
- The provided model is intentionally simple (feed-forward) and trained using heuristic labels derived from user-goal/food suitability columns.
- For production: use a larger model, proper cross-validation, more features (user history), data augmentation, and secure deployment (HTTPS, authentication).

Dataset path used by backend: `/mnt/data/healthy_recipes_augmented.xlsx`\n