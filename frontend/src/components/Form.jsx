import React, { useState } from "react";

export default function Form({ onResult }) {
  const [state, setState] = useState({
    age: "23",
    height_cm: "170",
    weight_kg: "70",
    activity_level: "1.55",
    goal: "loss",
    deficiency: "none",
    chronic: "none",
    cuisine_pref: "none",
    food_type: "none",
  });

  const submit = async (e) => {
    e.preventDefault(); // 🚀 Prevent reload ALWAYS

    // 🔥 Important: FastAPI expects "none" (string), NOT null.
    const payload = {
      age: Number(state.age),
      height_cm: Number(state.height_cm),
      weight_kg: Number(state.weight_kg),
      activity_level: Number(state.activity_level),
      goal: state.goal.trim(),
      deficiency: state.deficiency.trim(),
      chronic: state.chronic.trim(),
      cuisine_pref: state.cuisine_pref.trim(),   // always string
      food_type: state.food_type.trim(),         // always string
      calorie_target: null,
    };

    console.log("FINAL PAYLOAD SENT:", payload);

    try {
      const resp = await fetch(
        "https://charm-care-health-ai-based-regimen.onrender.com/generate_plan",
        {
          method: "POST",
          headers: {
            "Accept": "application/json",
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
        }
      );

      const data = await resp.json();
      console.log("RESPONSE:", data);

      if (!resp.ok) {
        console.error("Backend error object:", data);
        throw new Error("Backend rejected request");
      }

      onResult(data);
    } catch (err) {
      console.error("❌ FETCH FAILED:", err);
      alert("⚠️ Failed to fetch plan. Check console.");
    }
  };

  return (
    <form onSubmit={submit}>
      <div className="row">
        <div className="col">
          <label>Age</label>
          <input
            type="number"
            value={state.age}
            onChange={(e) => setState({ ...state, age: e.target.value })}
          />
        </div>

        <div className="col">
          <label>Height (cm)</label>
          <input
            type="number"
            value={state.height_cm}
            onChange={(e) => setState({ ...state, height_cm: e.target.value })}
          />
        </div>

        <div className="col">
          <label>Weight (kg)</label>
          <input
            type="number"
            value={state.weight_kg}
            onChange={(e) => setState({ ...state, weight_kg: e.target.value })}
          />
        </div>
      </div>

      <label>Activity Level</label>
      <input
        type="number"
        step="0.1"
        value={state.activity_level}
        onChange={(e) =>
          setState({ ...state, activity_level: e.target.value })
        }
      />

      <div className="row">
        <div className="col">
          <label>Goal</label>
          <select
            value={state.goal}
            onChange={(e) => setState({ ...state, goal: e.target.value })}
          >
            <option value="loss">Weight Loss</option>
            <option value="gain">Weight Gain</option>
            <option value="muscle">Muscle Gain</option>
          </select>
        </div>

        <div className="col">
          <label>Deficiency</label>
          <select
            value={state.deficiency}
            onChange={(e) =>
              setState({ ...state, deficiency: e.target.value })
            }
          >
            <option value="none">None</option>
            <option value="iron">Iron</option>
            <option value="vitd">Vitamin D</option>
            <option value="protein">Protein</option>
          </select>
        </div>

        <div className="col">
          <label>Chronic</label>
          <select
            value={state.chronic}
            onChange={(e) =>
              setState({ ...state, chronic: e.target.value })
            }
          >
            <option value="none">None</option>
            <option value="diabetes">Diabetes</option>
            <option value="hypertension">Hypertension</option>
          </select>
        </div>
      </div>

      <label>Cuisine Preference</label>
      <input
        value={state.cuisine_pref}
        onChange={(e) =>
          setState({ ...state, cuisine_pref: e.target.value })
        }
      />

      <label>Food Type</label>
      <select
        value={state.food_type}
        onChange={(e) => setState({ ...state, food_type: e.target.value })}
      >
        <option value="none">No Preference</option>
        <option value="vegetarian">Vegetarian</option>
        <option value="vegan">Vegan</option>
        <option value="non-vegetarian">Non-Vegetarian</option>
      </select>

      <button type="submit">Generate 7-day Plan</button>
    </form>
  );
}
