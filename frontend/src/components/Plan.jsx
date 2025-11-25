import React from 'react'

export default function Plan({ data }) {
  const { plan, workout } = data
  return (
    <div style={{ marginTop: 20 }}>
      <h2>Weekly Diet Plan</h2>
      {plan.days.map((d, i) => (
        <div key={i} style={{ marginBottom: 12 }}>
          <h3>Day {i + 1}</h3>
          {['Breakfast', 'Lunch', 'Dinner'].map(m => (
            <div className="meal" key={m}>
              <strong>{m}</strong>: {d[m].recipe_name} — {d[m].calories} kcal, P:{d[m].protein_g}g, C:{d[m].carbs_g}g, F:{d[m].fat_g}g
              <p><strong>Ingredients:</strong> {d[m].ingredients}</p>
              <p><strong>Preparation:</strong> {d[m].preparation}</p>
            </div>
          ))}
          <div style={{ fontStyle: 'italic' }}>Workout: {workout[i]}</div>
        </div>
      ))}
    </div>
  )
}
