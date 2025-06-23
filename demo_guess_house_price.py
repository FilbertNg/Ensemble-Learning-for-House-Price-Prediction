import streamlit as st
import pandas as pd
import numpy as np
import random
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# --- LOAD EVERYTHING ---

# Load selected features
with open('features', 'rb') as fp:
    features = pickle.load(fp)

# Load original and preprocessed data
original_df = pd.read_csv('train.csv')[features]
preprocessed_df = pd.read_csv('preprocessed_data.csv')

# Set "Id" as index for both 
original_df.set_index("Id", inplace=True)
preprocessed_df.set_index("Id", inplace=True)

# Prepare feature matrix and target
predictor = features[1:-1]  # exclude 'Id' and 'SalePrice'
target = features[-1]
X = preprocessed_df[predictor]
y = preprocessed_df[target]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Setup Gradient Boosting with best params
best_gb_params = {
    'learning_rate': 0.06404377375487695,
    'max_depth': 6,
    'min_samples_split': 0.11104367985135397,
    'min_samples_leaf': 0.019217918231543812,
    'subsample': 0.6346935822600632,
    'max_features': 'log2'
}

best_gb = GradientBoostingRegressor(**best_gb_params, random_state=42)
best_gb.fit(X_train, y_train)

# selected features to display
selected_features = ['GrLivArea', 'OverallQual', 'Neighborhood', 'GarageCars', 'YearBuilt', 'TotalBsmtSF', 'KitchenQual', 'FullBath', 'MSZoning']

# Human-readable names 
feature_labels = {
    'GrLivArea': 'Above Ground Living Area (sq ft)',
    'OverallQual': 'Overall Quality',
    'Neighborhood': 'Neighborhood (In Ames)',
    'GarageCars': 'Garage Capacity (Cars)',
    'YearBuilt': 'Year Built',
    'TotalBsmtSF': 'Total Basement Area (sq ft)',
    'KitchenQual': 'Kitchen Quality',
    'FullBath': 'Full Bathrooms',
    'MSZoning': 'Zoning Classification'
}

# Mapping dictionaries
neighborhood_map = {
    'Blmngtn': 'Bloomington Heights', 'Blueste': 'Bluestem', 'BrDale': 'Briardale', 'BrkSide': 'Brookside',
    'ClearCr': 'Clear Creek', 'CollgCr': 'College Creek', 'Crawfor': 'Crawford', 'Edwards': 'Edwards',
    'Gilbert': 'Gilbert', 'IDOTRR': 'Iowa DOT and Rail Road', 'MeadowV': 'Meadow Village', 'Mitchel': 'Mitchell',
    'Names': 'North Ames', 'NoRidge': 'Northridge', 'NPkVill': 'Northpark Villa', 'NridgHt': 'Northridge Heights',
    'NWAmes': 'Northwest Ames', 'OldTown': 'Old Town', 'SWISU': 'S/W of ISU', 'Sawyer': 'Sawyer',
    'SawyerW': 'Sawyer West', 'Somerst': 'Somerset', 'StoneBr': 'Stone Brook', 'Timber': 'Timberland',
    'Veenker': 'Veenker'
}

overallqual_map = {
    10: 'Very Excellent', 9: 'Excellent', 8: 'Very Good', 7: 'Good',
    6: 'Above Average', 5: 'Average', 4: 'Below Average', 3: 'Fair',
    2: 'Poor', 1: 'Very Poor'
}

kitchenqual_map = {
    'Ex': 'Excellent', 'Gd': 'Good', 'TA': 'Typical/Average', 'Fa': 'Fair', 'Po': 'Poor'
}

mszoning_map = {
    'A': 'Agriculture', 'C': 'Commercial', 'FV': 'Floating Village Residential',
    'I': 'Industrial', 'RH': 'Residential High Density', 'RL': 'Residential Low Density',
    'RP': 'Residential Low Density Park', 'RM': 'Residential Medium Density'
}


# --- STREAMLIT APP ---
st.set_page_config(page_title="Guess the House Price", layout="centered")

# Session state init
if 'selected_id' not in st.session_state or st.button("🔁 Try Another House"):
    # Choose random ID from test set
    valid_test_ids = list(set(X_test.index) & set(preprocessed_df.index))
    selected_id = random.choice(valid_test_ids)
    st.session_state.selected_id = selected_id
    st.session_state.guessed = False

# Use selected ID
selected_id = st.session_state.selected_id

# Prepare data
X_row = preprocessed_df.loc[[selected_id]].drop(columns="SalePrice", errors="ignore")
X_row = X_row[X_train.columns]


# Predict
log_pred = best_gb.predict(X_row)[0]
predicted_price = np.expm1(log_pred)
actual_price = original_df.loc[selected_id, "SalePrice"]


# Show features
st.markdown(f"### 🏠 House ID: `{selected_id}`")
st.markdown("Here's some info about the house:")
# Select features
selected_features = list(feature_labels.keys())
values = original_df.loc[selected_id, selected_features].copy()

# Apply value mappings
values['Neighborhood'] = neighborhood_map.get(values['Neighborhood'], values['Neighborhood'])
values['OverallQual'] = overallqual_map.get(values['OverallQual'], values['OverallQual'])
values['KitchenQual'] = kitchenqual_map.get(values['KitchenQual'], values['KitchenQual'])
values['MSZoning'] = mszoning_map.get(values['MSZoning'], values['MSZoning'])

# Build final display table
display_df = pd.DataFrame({
    "Feature": [feature_labels[f] for f in selected_features],
    "Value": values.astype(str).values
})
st.dataframe(display_df)


# Show predicted price
st.markdown(f"## 🤖 Our model predicts the house price to be:")
st.markdown(f"### **${predicted_price:,.2f}**")

st.markdown("### ❓ Do you think the **actual price** is higher or lower than the prediction?")

# Buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("⬆️ Higher"):
        st.session_state.guess = "higher"
        st.session_state.guessed = True
with col2:
    if st.button("⬇️ Lower"):
        st.session_state.guess = "lower"
        st.session_state.guessed = True

# Show result
if st.session_state.guessed:
    st.markdown("---")
    st.markdown(f"### ✅ Actual Price: **${actual_price:,.2f}**")

    correct = (
        (actual_price > predicted_price and st.session_state.guess == "higher") or
        (actual_price < predicted_price and st.session_state.guess == "lower")
    )

    if correct:
        st.success("🎉 You guessed it right!")
    else:
        st.error("😢 Oops, that's not correct.")


