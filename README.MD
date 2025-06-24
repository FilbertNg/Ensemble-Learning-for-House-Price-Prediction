# 🏠 Ensemble Learning for House Price Prediction

## 🔍 Project Overview  
This project leverages **ensemble learning techniques** to predict house prices based on the Kaggle [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) dataset.

### 📈 Workflow Summary  
1. **Data Preprocessing**  
   - Duplicate Handling  
   - Missing Value Imputation  
   - Encoding Categorical Features  
   - Feature Scaling  
   - Log Transformation  
   - Feature Selection via Correlation Analysis  

2. **Ensemble Modeling**  
   - **Bagging Models:**  
     - Random Forest Regressor  
     - Extra Trees Regressor  
   - **Boosting Models:**  
     - Gradient Boosting Regressor  
     - XGBoost  

3. **Hyperparameter Tuning**  
   - Conducted using **Optuna**, a powerful framework for automated hyperparameter optimization.

---

## 🧠 Tools & Technologies  
- Python  
- Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `optuna`, `xgboost`  
- Streamlit: For building an interactive app  

---

## 🎮 Try the Demo (Recommended!)  
Experience the model in action via a **Streamlit-powered interactive demo**:

### 🧩 "Guess the Price" Game  
The app presents a randomly generated house profile with its features. You’ll see the model’s predicted price and be asked:

> **“Is the real price higher or lower?”**

Can you beat the model?

👉 **[Launch the Demo](https://ensemble-learning-for-house-price-prediction.streamlit.app/)**

---

## 📚 Acknowledgements  
- Kaggle House Prices Dataset:  
  [https://www.kaggle.com/c/house-prices-advanced-regression-techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
