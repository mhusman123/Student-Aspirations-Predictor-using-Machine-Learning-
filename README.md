# 🎓 Student Career Aspiration Prediction Using Machine Learning

> An AI-powered web application that predicts a student's ideal career path based on academic scores, study habits, and personal traits — built with **Scikit-Learn** and **Streamlit**.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

Students often struggle to identify the right career path early in their academic journey. This project uses a **Random Forest Classifier** to predict which of **17 career categories** best matches a student's profile, helping in career guidance and educational planning.

The system takes **14 input features** (gender, part-time job status, absence days, extracurricular activities, weekly self-study hours, and scores in 7 subjects) and outputs ranked career recommendations with confidence probabilities.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **ML Pipeline** | `StandardScaler` + `RandomForestClassifier` (200 estimators, balanced class weights) saved as a single pickle file |
| **17 Career Paths** | Lawyer, Doctor, Software Engineer, Artist, Teacher, Scientist, Banker, Writer, Designer, Game Developer, and more |
| **14 Input Features** | Gender, part-time job, absence days, extracurricular activities, weekly study hours, and 7 subject scores (Math, History, Physics, Chemistry, Biology, English, Geography) |
| **Modern Dark UI** | Glassmorphism design with gradient text, animated cards, and smooth transitions |
| **Interactive Charts** | Bar chart, Radar chart, and Donut chart via Plotly — all dark-themed |
| **Score Summary** | Auto-computed total/average score, highest subject, and study-hour metrics |
| **Subject Profile Radar** | Visual breakdown of academic strengths across all 7 subjects |

---

## 🖥️ Screenshots

### Landing Page
- Dark gradient hero section with animated title
- Stats row: 17 career paths · 2,000+ student records · ~81% accuracy · 14 features
- Three hoverable feature cards (Smart Analysis, Top-5 Predictions, Instant Results)
- Glowing "Get Started" CTA button

### Predictor Page
- Clean form layout: Personal Info (4 cols) → Study Habits → Subject Scores (4 cols)
- Full-width gradient "Predict Career" button

### Results
- **Top-5 Career Cards** — ranked with icons, career name, and gradient probability
- **3 Tabs**: Bar Chart (top 10), Radar Chart (top 6), Donut Chart (top 8)
- **Score Summary** — total/700, average, highest subject, weekly study hours
- **Subject Profile Radar** — your academic shape at a glance

---

## 📂 Project Structure

```
├── app.py                        # Streamlit web app with modern UI/UX
├── train_and_save_model.py       # Model training script (saves model_pipeline.pkl)
├── model_pipeline.pkl            # Trained ML pipeline (generated after training)
├── student-scores.csv            # Original dataset (2,000+ records)
├── train_data.csv                # Preprocessed training data
├── test_data.csv                 # Preprocessed test data
├── requirements.txt              # Python dependencies
├── Predicting Career Aspirations of Students Using Machine Learning.ipynb
│                                 # Jupyter notebook with EDA & experiments
├── LICENSE                       # MIT License
└── README.md                     # This file
```

---

## 🛠️ Tech Stack

- **Language**: Python 3.10+
- **ML Framework**: Scikit-Learn (RandomForestClassifier, StandardScaler, Pipeline)
- **Web Framework**: Streamlit
- **Visualisation**: Plotly (Bar, Radar/Scatterpolar, Donut/Pie)
- **Data**: Pandas, NumPy

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/Student-Career-Aspiration-Prediction-Using-Machine-Learning.git
cd Student-Career-Aspiration-Prediction-Using-Machine-Learning

# 2. Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model (first time only)
python train_and_save_model.py

# 5. Run the app
streamlit run app.py
```

The app will open at **http://localhost:8501**.

---

## 📊 Model Details

| Parameter | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Estimators | 200 |
| Class Weights | Balanced |
| Preprocessing | StandardScaler |
| Train/Test Split | 80 / 20 |
| Test Accuracy | **~81%** |

### Target Classes (17)

| # | Career | # | Career |
|---|---|---|---|
| 0 | Lawyer | 9 | Banker |
| 1 | Doctor | 10 | Writer |
| 2 | Government Officer | 11 | Accountant |
| 3 | Artist | 12 | Designer |
| 4 | Unknown | 13 | Construction Engineer |
| 5 | Software Engineer | 14 | Game Developer |
| 6 | Teacher | 15 | Stock Investor |
| 7 | Business Owner | 16 | Real Estate Developer |
| 8 | Scientist | | |

### Input Features (14)

| Feature | Type |
|---|---|
| gender | Binary (0 = Male, 1 = Female) |
| part_time_job | Binary (0 / 1) |
| absence_days | Integer |
| extracurricular_activities | Binary (0 / 1) |
| weekly_self_study_hours | Integer |
| math_score | Integer (0–100) |
| history_score | Integer (0–100) |
| physics_score | Integer (0–100) |
| chemistry_score | Integer (0–100) |
| biology_score | Integer (0–100) |
| english_score | Integer (0–100) |
| geography_score | Integer (0–100) |
| total_score | Computed (sum of 7 subjects) |
| average_score | Computed (total / 7) |

---

## 🎨 UI/UX Design

The web interface uses a **dark glassmorphism** design language:

- **Color Palette**: Deep navy background (`#0F0F1A` → `#1A1A2E`), purple primary (`#6C5CE7`), pink accent (`#FD79A8`)
- **Typography**: Poppins (headings) + Inter (body) via Google Fonts
- **Cards**: Frosted-glass effect with `backdrop-filter: blur(20px)` and subtle borders
- **Animations**: `fadeInUp` entrance animations, hover lift/glow transitions
- **Charts**: Transparent backgrounds matching the dark theme, custom color scales
- **Responsive**: Adapts layout on smaller screens

---

## 🔮 Future Work

- Enhance accuracy with ensemble methods (LightGBM, CatBoost, Stacking)
- Add SMOTE for class balancing on under-represented careers
- Deploy to a cloud host (Streamlit Cloud, Railway, or Heroku)
- Add batch prediction via CSV upload
- User authentication and history tracking

---

## 👥 Contributors

- **Muhammad Usman**
- **Abdullah**
- **Inayatullah**
- **Afzal**

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  All rights reserved by <strong>Muhammad Usman Mahar</strong>
</p>
