# 🚀 Unmaksing Team Rocket
In real-world scenarios like security and fraud detection, threats may not always look alike — but they often act alike. Inspired by Team Rocket’s consistent intent despite changing roles and disguises, this project explores how learning behavioral patterns can help identify threat-like actions, even when surface appearances vary.

---

## 🎯 Objective

- Build a binary classifier to detect threat-like behavior (here detect presence of Team Rocket Member)
- Tackle an imbalanced dataset (82:18 non-threat to threat)
- Use models like Random Forest and XGBoost
- Evaluate using accuracy, precision, recall, and ROC-AUC

> *To protect the world from threat…*  
> *To catch attackers you won’t forget!*  
> *Random Forest Classifier!*  
> *XGBoost Classifier!*  
> *Surrender now…*  
> *Or prepare to blast off with insight!*

 ---

 ## Tech Stack / Libraries Used
 
|      Tool / Library        |                Task                 | 
|----------------------------|-------------------------------------|
| `**Python**`               | Core programming language           |
| `**pandas**`               | Data manipulation and preprocessing | 
| `**NumPy**`                | Numerical computations              |
| `**scikit-learn**`         | Model training and evaluation       |
| `**matplotlib / seaborn**` | Static data visualization           |
| `**plotly**`               | Interactive visualizations          |
| `**Streamlit**`            | Web app deployment                  |
| `**joblib**`               |  Saving and loading ML models       |

---
 
## 📊 Results

### 🔁 Random Forest Classifier   

| Metric    | Value     |                                
|-----------|-----------|                                
| Accuracy  | `~0.916`  |                              
| ROC-AUC   | `~0.80`   |                              
                                                                   
### 🔘 XGBoost Classifier

| Metric     | Value     |
|------------|-----------|
| Accuracy   | `~0.905`  |
| ROC-AUC    | `~0.79`   |

🔗 **Live App**: [Click here to try it out](https://huggingface.co/spaces/steam-bell-92/Unmasking_Team_Rocket)
[![Hugging Face Spaces](https://img.shields.io/badge/Hosted%20on-Hugging%20Face-blue?logo=huggingface)](https://huggingface.co/spaces/steam-bell-92/Unmasking_Team_Rocket)
---

## 📁 File Structure

```
Unmasking_Team_Rocket/
│
├── CODES
|    ├── Unmasking_Team_Rocket.ipynb            🔹 Jupyter notebook containing entire ML Workflow
|    ├── unmaksing_team_rocket.py               🔹 Python File
|    ├── Team_Rocket.png                        🔹 Image embedded in notebook
|    └── pokemon_team_rocket_dataset.csv        🔹 Dataset
|
├── WEB
|    ├── xgb_model.pkl                          🔹 Gathers best model with its parameters
|    ├── model_features.pkl                     🔹 Gathers model features (columns for prediction)
|    ├── requirements.txt                       🔹 Things required to make deployment work
|    └── app.py                                 🔹 Streamlit code for deployment
|
├── LICENSE                                     🔹 MIT License
└── README.md                                   🔹 This file !!
```

---

## 👤 Author
Anuj Kulkarni — aka — steam-bell-92

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
