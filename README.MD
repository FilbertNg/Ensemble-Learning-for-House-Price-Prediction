# 🏠 Ensemble Learning for House Price Prediction


## 🔍 Project Overview
This project aims to predict house prices using advanced ensemble learning techniques. The dataset comes from the Kaggle House Prices - Advanced Regression Techniques competition.

Approach: Data preprocessing → ensemble modeling → hyperparameter tuning.

Preprocessing Techniques:
- Duplicat Handling
- Missing Value Handling
- Encoding
- Feature Scaling
- Log Transformation
- Features Selection using Correlation Analysis
  
Ensemble Learning Mode:
1. Bagging
   - Random Forest Regressor
   - Extra Trees Regressor
2. Boosting
   - Gradient Boosting Regressor
   - XGBoost
  
Hyperparameter Tuning using Optuna.

## 🧠 Tools & Technologies
- Python
- Pandas, NumPy, Scikit-learn
- Optuna for hyperparameter tuning
- Matplotlib (for visualization)
- Streamlit (for interactive app)

## 🎮 Try the Demo (Recommended!)
Explore the model in action with an interactive Streamlit demo!
  
### 🧩 Guess the Price Game:
The app will display a randomly generated house with its features. You'll be shown the model's predicted price and asked:
"Is the real price higher or lower?"

See how close the model is — and how well you can guess!

### ▶️ How to Run the Demo
1. Clone the repository
2. Install dependencies: <pre><code>```bash pip install -r requirements.txt```</code></pre>
3. Run the app: <pre><code>```bash streamlit run demo_app.py ```</code></pre>


## 📚 Acknowledgements
- Kaggle Housing Dataset: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
