# 🚀 Unmaksing Team Rocket
In real-world scenarios like security and fraud, threats may not look alike — but often act alike. Inspired by Team Rocket’s shared intent despite different roles, this project explores learning behavioral patterns to detect threat-like actions.

---

## 🎯 Objective

- Build a binary classifier to detect threat-like behavior
- Tackle an imbalanced dataset (82:18 non-threat to threat)
- Use models like Random Forest and XGBoost
- Evaluate using accuracy, precision, recall, and ROC-AUC

 ---
 
## 📊 Results

### 🔁 Random Forest Classifier   

| Metric    | Value     |                                
|-----------|-----------|                                
| Accuracy  | `~0.916`   |                              
| ROC-AUC   | `~0.80`   |                              
                                                                   
### 🔘 XGBoost Classifier

| Metric     | Value     |
|------------|-----------|
| Accuracy   | `~0.905`   |
| ROC-AUC    | `~0.79`   |

---

## 📁 File Structure

```
Unmasking_Team_Rocket/
│
├── CODES
|    ├── Unmasking_Team_Rocket.ipynb            🔹 Jupyter notebook containing entire ML Workflow
|    ├── unmaksing_team_rocket.py               🔹 Python File
|    ├── Team_Rocket.png                        🔹 Image embeded in notebook
|    └── pokemon_team_rocket_dataset.csv        🔹 Dataset
|
├── CODES
|    ├── xgb_model.pkl                          🔹 
|    ├── model_features.pkl                     🔹 
|    └── app.py                                 🔹 Streamlit code for deployment
|
├── LICENSE                                     🔹 MIT License
└── README.md                                   🔹 This file !!
```

---

## 👤 Author
Anuj Kulkarni - aka - steam-bell-92

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
