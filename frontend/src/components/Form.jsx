import React, { useState } from 'react'

export default function Form({ onResult }) {
  const [state, setState] = useState({
    age: 23,
    height_cm: 170,
    weight_kg: 70,
    activity_level: 1.55,
    goal: 'loss',
    deficiency: 'none',
    chronic: 'none',
    cuisine_pref: '',
    food_type: 'none'   // added correctly
  })

  const submit = async (e) => {
    e.preventDefault()

    const resp = await fetch('http://localhost:8000/generate_plan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(state)
    })

    const data = await resp.json()
    onResult(data)
  }

  return (
    <form onSubmit={submit}>
      <div className="row">
        <div className="col">
          <label>Age</label>
          <input
            type="number"
            value={state.age}
            onChange={e => setState({ ...state, age: parseInt(e.target.value) })}
          />
        </div>

        <div className="col">
          <label>Height (cm)</label>
          <input
            type="number"
            value={state.height_cm}
            onChange={e => setState({ ...state, height_cm: parseFloat(e.target.value) })}
          />
        </div>

        <div className="col">
          <label>Weight (kg)</label>
          <input
            type="number"
            value={state.weight_kg}
            onChange={e => setState({ ...state, weight_kg: parseFloat(e.target.value) })}
          />
        </div>
      </div>

      <label>Activity Level</label>
      <input
        type="number"
        step="0.5"
        value={state.activity_level}
        onChange={e => setState({ ...state, activity_level: parseFloat(e.target.value) })}
      />

      <div className="row">
        <div className="col">
          <label>Goal</label>
          <select
            value={state.goal}
            onChange={e => setState({ ...state, goal: e.target.value })}
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
            onChange={e => setState({ ...state, deficiency: e.target.value })}
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
            onChange={e => setState({ ...state, chronic: e.target.value })}
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
        onChange={e => setState({ ...state, cuisine_pref: e.target.value })}
      />

      <label>Food Type</label>
      <select
        value={state.food_type}
        onChange={(e) => setState({ ...state, food_type: e.target.value })}
      >
        <option value="None">No Preference</option>
        <option value="Vegetarian">Vegetarian</option>
        <option value="Vegan">Vegan</option>
        <option value="Non-Vegetarian">Non-Vegetarian</option>
      </select>

      <button type="submit">Generate 7-day Plan</button>
    </form>
  )
}
