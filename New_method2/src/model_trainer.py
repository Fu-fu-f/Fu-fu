# src/model_trainer.py
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from . import config

class ModelTrainer:
    def __init__(self, data_path=None, models_dir='trained_models'):
        self.data_path = data_path or config.FINAL_FILE
        self.models_dir = models_dir
        self.df = None
        self.feature_cols = []
        # User requested ONLY viability
        self.targets = ['viability']
        
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def load_and_prepare_data(self):
        """Loads data and prepares features (OHE, etc)."""
        print(f"Loading training data from {self.data_path}...")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}. Please run 'run_pipeline.py' first.")
            
        self.df = pd.read_csv(self.data_path)
        
        # --- Feature Engineering ---
        # 1. One-hot encode 'cooling rate' if it exists
        # We need to handle this carefully to match future inference expectations
        if 'cooling rate' in self.df.columns:
            self.df = pd.get_dummies(self.df, columns=['cooling rate'], prefix='cooling_rate', dummy_na=False)
        
        # 2. Define Feature Columns
        # Everything that is NOT metadata or target is a feature
        # FIXED: Explicitly exclude 'recovery' and 'viability' regardless of what we are training
        all_targets = ['viability', 'recovery']
        meta_cols = ['doubling time', config.RAW_INGREDIENT_COL] + all_targets
        
        # Ensure we only pick numeric columns as features
        potential_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_cols = [c for c in potential_features if c not in meta_cols]
        
        # 3. Remove constant columns (variance = 0), BUT keep critical chemical ingredients even if 0 currently
        # (This differs slightly from old logic which kept 'dmso'. We'll trust the data unless it's strictly empty)
        # Actually, let's stick to the safe side: remove columns that differ in 0 or 1 value only?
        # Standard cleaning:
        # self.feature_cols = [c for c in self.feature_cols if self.df[c].nunique() > 1]
        
        print(f"Training with {len(self.feature_cols)} features: {self.feature_cols}")

    def train(self):
        """Trains models for all targets."""
        if self.df is None:
            self.load_and_prepare_data()

        results = {}

        for target in self.targets:
            if target not in self.df.columns:
                print(f"Skipping target '{target}' (not in data).")
                continue
                
            print(f"\n=== Training Model for {target.capitalize()} ===")
            
            # Filter rows where target is present
            train_df = self.df.dropna(subset=[target])
            X = train_df[self.feature_cols]
            y = train_df[target]
            
            if len(X) < 10:
                print(f"Not enough data to train for {target} ({len(X)} rows).")
                continue

            # XGBoost Hyperparameters (preserved from original project)
            xgb_params = {
                'objective': 'reg:squarederror',
                'n_estimators': 200,
                'learning_rate': 0.05,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1
            }
            
            model = xgb.XGBRegressor(**xgb_params)
            
            # --- Evaluation (CV) ---
            # 5-fold CV for robust metric
            # Use Stacking Regressor: XGB + Random Forest
            from sklearn.ensemble import RandomForestRegressor, StackingRegressor
            from sklearn.linear_model import LinearRegression
            
            estimators = [
                ('xgb', xgb.XGBRegressor(**xgb_params)),
                ('rf', RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1))
            ]
            
            # Final estimator: Linear combination (avoids overfitting compared to another tree)
            stack = StackingRegressor(
                estimators=estimators,
                final_estimator=LinearRegression()
            )
            
            scores = cross_val_score(stack, X, y, cv=5, scoring='r2')
            print(f"Cross-Validation R² (Stacking XGB+RF): {scores.mean():.4f} (±{scores.std():.4f})")
            
            # --- Final Fit ---
            stack.fit(X, y)
            results[target] = stack
            
            # Save
            save_path = os.path.join(self.models_dir, f"{target}_model.joblib")
            joblib.dump(stack, save_path)
            print(f"Saved stacked model to {save_path}")
            
            # Feature Importance
            # Stacking doesn't have direct feature_importances_. We'll look at the base RF model for interpretation.
            print("\nInterpretation (from Base Random Forest):")
            self._print_importance(stack.named_estimators_['rf'], X.columns)

    def _print_importance(self, model, feature_names):
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            print(f"Feature Importances ({len(feature_names)} features):")
            for f in range(len(feature_names)):
                print(f"  {feature_names[indices[f]]}: {importances[indices[f]]:.4f}")
        else:
            print("Model does not support feature importance inspection.")

