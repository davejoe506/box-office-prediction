import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import shap
import joblib

# --- Path configuration ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "clean", "tmdb_movies_features.csv")
VIZ_DIR = os.path.join(PROJECT_ROOT, "visualizations")

os.makedirs(VIZ_DIR, exist_ok=True)

def evaluate_model(name, y_true_dollars, y_pred_dollars):
    """Calculates metrics on the actual dollar amounts."""
    rmse = np.sqrt(mean_squared_error(y_true_dollars, y_pred_dollars))
    mae = mean_absolute_error(y_true_dollars, y_pred_dollars)
    r2 = r2_score(y_true_dollars, y_pred_dollars)
    
    print(f"\n--- {name} Performance ---")
    print(f"R² Score: {r2:.3f}")
    print(f"RMSE:     ${rmse:,.0f}")
    print(f"MAE:      ${mae:,.0f}")
    
    return r2, rmse, mae

def main():
    print("Starting modeling phase (XGBoost + SHAP)...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return
        
    df = pd.read_csv(INPUT_FILE)
    
    # 1. Prevent data leakage
    print("Dropping non-predictive and 'future' columns...")
    cols_to_drop = [
        'id', 'title', 'original_language', 
        'release_year', 'release_month', 'release_day_of_week',
        'budget', 'revenue', 'revenue_adj', 
        'vote_average', 'vote_count', 'popularity', 
        'season' 
    ]
    
    # Ensure no missing critical values
    df = df.dropna(subset=['runtime', 'budget_adj', 'revenue_adj'])
    
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # Train model to predict the log of revenue, which normalizes extreme blockbusters
    y_log = np.log1p(df['revenue_adj']) 
    y_actual_dollars = df['revenue_adj']
    
    print(f"Features used ({X.shape[1]}): {list(X.columns[:5])}...")
    
    # 2. Train/test split
    X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)
    _, _, _, y_test_dollars = train_test_split(X, y_actual_dollars, test_size=0.2, random_state=42)
    
    print(f"Training on {len(X_train)} movies, testing on {len(X_test)} movies.")
    
    # 3. XGBoost
    print("Training XGBoost regressor...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=300,        # Number of trees
        learning_rate=0.05,      # How aggressively it corrects mistakes
        max_depth=5,             # Complexity of each tree
        subsample=0.8,           # Prevents overfitting
        colsample_bytree=0.8,    # Randomly selects features for each tree
        random_state=42
    )
    xgb_model.fit(X_train, y_train_log)
    
    # 4. Evaluation
    # Convert log predictions back to actual dollars using expm1
    xgb_preds_log = xgb_model.predict(X_test)
    xgb_preds_dollars = np.expm1(xgb_preds_log)
    
    evaluate_model("XGBoost (log-transformed)", y_test_dollars, xgb_preds_dollars)
    
    # 5. Generate SHAP values
    print("Generating SHAP feature importance chart...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    
    # Get current figure and axes
    fig = plt.gcf()
    ax = plt.gca()
    
    # Modify titles and labels
    ax.set_title('SHAP Value Summary: How Features Drive Box Office', fontsize=16, pad=20)
    ax.set_xlabel('SHAP Value (Impact on Predicted Log-Revenue)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "08_shap_summary.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Save
    import joblib
    joblib.dump(xgb_model, os.path.join(PROJECT_ROOT, "model.pkl"))
    joblib.dump(list(X.columns), os.path.join(PROJECT_ROOT, "model_features.pkl"))

    print("-" * 30)
    print(f"Modeling complete. Check the visualizations folder for the SHAP chart.")

if __name__ == "__main__":
    main()