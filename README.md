# 🧬 CHARM — Care, Health & AI-based Regimen Manager  
### *AI-powered Personalized Health, Diet & Nutrition Recommendation System*

---

## 📌 Overview

**CHARM** (**C**are, **H**ealth & **A**I-based **R**egimen **M**anager) is an AI-driven application that generates **personalized diet plans**, **calorie recommendations**, **daily meal insights**, and **health metrics** based on user input.

This project includes:

- A **React (Vite)** frontend with modern UI  
- A **Python backend** (Flask/FastAPI ready)  
- Full **dark/light theming**, **glassmorphism UI**, and **animated backgrounds**  
- Smooth transitions and responsive design

---

## 🌟 Features

### 🧠 AI-Powered Health & Diet Recommendations
- Personalized diet plans  
- BMI analysis  
- TDEE & calorie suggestions  
- Macro breakdown: protein, carbs, fats  
- Plans adapt based on goals: *loss / gain / maintain*

### 🎨 Modern UI & UX
- Infinite animated background  
- Blue–green themed interface  
- Glassmorphism (light mode)  
- Deep neon-glass (dark mode)  
- Smooth fade-in animations  
- Clean & responsive layout

### 🔥 Robust Architecture
- React components: `App.jsx`, `Form.jsx`, `Plan.jsx`  
- Global `styles.css` with dynamic CSS variables  
- Backend designed to integrate ML models (PyTorch-ready)

---

## 🗂️ Project Structure
```
CHARM/
│
├── backend/
│ ├── main.py
│ ├── utils/
│ ├── models/
│ ├── requirements.txt
│
├── frontend/
│ ├── App.jsx
│ ├── Form.jsx
│ ├── Plan.jsx
│ ├── styles.css
│ └── index.html
│
├── .gitignore
└── README.md

yaml
Copy code
```
---

## ⚙️ Tech Stack

### **Frontend**
- React (Vite)
- CSS (custom)
- Animated background
- Glassmorphism UI
- Theme toggler (light/dark)

### **Backend**
- Python 3.x  
- Flask or FastAPI  
- PyTorch (optional for ML)  
- NumPy, Pandas  

---

## 🚀 Getting Started

Link will be published soon...

## 🧪 Sample API Request
json
Copy code
{
  "age": 24,
  "gender": "male",
  "height": 175,
  "weight": 70,
  "activity": "moderate",
  "goal": "weight_loss"
}
## 🧭 Usage Flow
User fills form

Backend calculates:

BMI

TDEE

Goal-based calories

Macro breakdown

Personalized diet plan is generated

React UI displays it with animations & theming

## 🧩 Future Enhancements
Food image-to-nutrition detection

Recipe generator using AI

Fitness routine generator

Daily progress tracking

Reminder system

Integration with Google Fit / Apple Health

Deployment to cloud (Vercel + Render)

## 🤝 Contributing
Contributions are welcome!
Feel free to submit issues or pull requests.

## 🛡 License
This project is under the MIT License.

❤️ Credits
LLM-powered logic assistance

Nutrition science formulas

Modern UI inspiration (glassmorphism + neon UI)
