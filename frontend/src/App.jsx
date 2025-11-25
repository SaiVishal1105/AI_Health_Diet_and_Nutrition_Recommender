import React, { useState } from 'react'
import Form from './components/Form'
import Plan from './components/Plan'

export default function App() {
  const [plan, setPlan] = useState(null)
  const [theme, setTheme] = useState("light")

  const toggleTheme = () => {
    const newTheme = theme === "light" ? "dark" : "light"
    setTheme(newTheme)
    document.body.setAttribute("data-theme", newTheme)
  }

  return (
    <div className="container">
      <div className="theme-row">
        <h1>AI Health-Based Diet/ Nutrition Recommender</h1>
        <button className="theme-toggle" onClick={toggleTheme}>
          {theme === "light" ? "🌙 Dark Mode" : "☀️ Light Mode"}
        </button>
      </div>

      <Form onResult={setPlan} />
      {plan && <Plan data={plan} />}
    </div>
  )
}
