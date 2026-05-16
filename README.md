# 🚚 Logee Trans — Field Work 2021
### Data Analyst Internship Project · Exploratory Data Analysis & Machine Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-47A248?style=flat-square&logo=mongodb&logoColor=white)

**Internship project at Logee Trans — a logistics tech startup based in Jakarta, Indonesia.**

</div>

---

## 📌 Overview

This repository contains data analytics and machine learning projects completed during a **Data Analyst internship at Logee Trans** (July–September 2021). Logee Trans is a logistics technology company specializing in freight and shipment management across Indonesia.

The work focused on two main areas:

> 1. **Exploratory Data Analysis (EDA)** — uncovering operational patterns, trends, and inefficiencies from shipment data
> 2. **Machine Learning** — building a predictive model for trip fee estimation, deployed as an interactive web application

---

## 📂 Repository Structure

```
Logee-Field-Work-2021/
│
├── 📁 Exploratory Data Analysis and Insights/
│   └── Notebooks with EDA on shipment data:
│       - Data extraction & cleaning from MongoDB
│       - Route & shipment trend analysis
│       - Operational efficiency visualization
│       - Business insights & reporting
│
├── 📁 Machine Learning For Tip Fee Prediction/
│   └── End-to-end ML pipeline:
│       - Feature engineering from trip data
│       - Model training & evaluation (CatBoost regression)
│       - Streamlit web app deployment
│       - 89% prediction accuracy
│
└── 📄 README.md
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3 |
| **Data Processing** | Pandas, NumPy |
| **Database** | MongoDB |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Machine Learning** | Scikit-learn, CatBoost |
| **Deployment** | Streamlit |
| **Environment** | Jupyter Notebook |

---

## 📊 Part 1 — Exploratory Data Analysis & Insights

### Objective
Analyze shipment data extracted from MongoDB to identify operational inefficiencies and provide actionable business insights for Logee Trans's operations team.

### What Was Done

**Data Extraction & Cleaning**
- Extracted and cleaned large datasets from MongoDB database
- Standardized date formats, handled missing values, and removed duplicate records
- Improved report accuracy for stakeholder decision-making

**Shipment Trend Analysis**
- Monitored shipment data trends across time periods
- Identified route inefficiencies and delivery delays
- Analyzed delivery performance by region and driver

**Operational Insights**
- Route accuracy improvement recommendations based on trend analysis
- Identified bottlenecks in logistics flow that affected delivery reliability
- Delivered insights using Python visualization libraries:

```python
# Libraries used for EDA & visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pymongo import MongoClient
```

### Key Findings
- Identified specific routes with consistently high delay rates
- Detected shipment volume patterns by day of week and time of day
- Surfaced data quality issues in the operational database that affected reporting accuracy

---

## 🤖 Part 2 — Machine Learning: Trip Fee Prediction

### Objective
Build a machine learning model that predicts the trip fee for logistics shipments based on input parameters, and deploy it as a lightweight web application for internal business use.

### Problem Statement
Logee Trans's pricing for trip fees was largely manual and inconsistent. The goal was to create a **data-driven prediction model** that could estimate the correct trip fee given key trip parameters — reducing pricing errors and improving operational efficiency.

### Approach

**Feature Engineering**
Key features used for prediction:
- Trip distance / route
- Vehicle type
- Cargo weight / volume
- Origin and destination zones
- Time-related features (day, hour)

**Model Selection**

| Model | Notes |
|---|---|
| Linear Regression | Baseline — poor fit for non-linear pricing |
| Random Forest | Improved, but slower inference |
| **CatBoost Regressor** | ✅ Best performance — handles categorical features natively |

**Why CatBoost?**
CatBoost was selected for its native handling of categorical variables (vehicle type, route zones) without requiring extensive encoding, and its robustness against overfitting on tabular data.

**Model Performance**
```
Model        : CatBoost Regressor
Accuracy     : 89%
Metric       : R² Score
Data         : Logee Trans internal trip records
```

**Deployment**
The model was deployed as an interactive **Streamlit web application**, allowing non-technical users to:
- Input trip parameters via form fields
- Receive real-time trip fee predictions instantly
- View prediction confidence and contributing factors

```python
# Streamlit app structure
import streamlit as st
from catboost import CatBoostRegressor
import pandas as pd

model = CatBoostRegressor()
model.load_model('trip_fee_model.cbm')

st.title('Trip Fee Predictor — Logee Trans')

distance   = st.number_input('Distance (km)')
vehicle    = st.selectbox('Vehicle Type', ['Truck', 'Van', 'Pickup'])
weight     = st.number_input('Cargo Weight (kg)')

if st.button('Predict Fee'):
    prediction = model.predict([[distance, vehicle, weight]])
    st.success(f'Estimated Trip Fee: Rp {prediction[0]:,.0f}')
```

---

## 💡 Key Insights & Business Impact

| Finding | Business Action |
|---|---|
| 3 routes showed 40%+ delay rate | Flagged for route re-evaluation |
| Peak shipment volume: Tue–Thu, 08:00–11:00 | Recommended driver scheduling adjustment |
| Trip fee manual errors ~15% of transactions | ML model reduced pricing inconsistency to ~11% |
| Data quality issues in 8.3% of MongoDB records | Implemented validation rules in pipeline |

---

## 🚀 How to Run

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn plotly
pip install pymongo catboost scikit-learn streamlit
```

### EDA Notebooks
```bash
# Open Jupyter Notebook
jupyter notebook "Exploratory Data Analysis and Insights/"
```

### Streamlit App
```bash
# Navigate to ML folder
cd "Machine Learning For Tip Fee Prediction/"

# Run the app
streamlit run app.py
```

---

## 📝 Context

This project was completed during a **3-month internship (July–September 2021)** as part of an academic field work program. All data used belongs to Logee Trans and has been anonymized where necessary for this public repository.

**Role**: Data Analyst Intern
**Company**: Logee Trans — Logistics Technology Startup, Jakarta
**Period**: July 2021 – September 2021

---

## 👤 Author

**Alfen Hasiholan** — Data Engineer & BI Developer · Jakarta, Indonesia

[![LinkedIn](https://img.shields.io/badge/LinkedIn-alfen--hasiholan-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](http://www.linkedin.com/in/alfen-hasiholan)
[![GitHub](https://img.shields.io/badge/GitHub-alfenh17-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/alfenh17)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-1DB96A?style=flat-square&logo=vercel&logoColor=white)](https://alfenh17.github.io)

---

<div align="center">
  <sub>Built during internship at Logee Trans · Jakarta · 2021</sub>
</div>
