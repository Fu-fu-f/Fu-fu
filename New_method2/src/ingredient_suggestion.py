# src/recommender.py
import joblib
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
from . import config
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class Recommender:
    def __init__(self, model_path='trained_models/viability_model.joblib'):
        self.model = joblib.load(model_path)
        # Load feature names from a dummy run or config
        # Ideally we should have saved feature names with the model.
        # For now, we reconstruct them based on the config logic (Hybrid approach)
        self.features = self._get_feature_list()
        
    def _get_feature_list(self):
        # Hardcoding the 17 features we saw in training for robustness
        # In a perfect world, this comes from metadata
        return [
            'eg(%)', 'dmso(%)', 'trehalose(%)', 'fbs(%)', 'sucrose(%)', 
            'fs(%)', 'glycerol(%)', 'mannitol(%)', 'creatine(%)', 
            'cooh-pll(%)', 'huxuc-90011(%)', 'dmem(%)', 
            'other_cpas(%)', 'other_sugars(%)', 'other_polymers_proteins(%)', 
            'total_peg(%)', 'other_additives(%)'
        ]

    def optimize_batch(self, n=8):
        """
        Generates 'n' diverse recipes using different strategies.
        """
        recipes = []
        strategies = [
            ("Conservative (Best Model)", {}),
            ("High Sugar Exploration", {'total_sugars(%)': (5.0, 20.0)}),
            ("High Polymer Exploration", {'total_polymers_proteins(%)': (1.0, 10.0)}),
            ("High PEG Exploration", {'total_peg(%)': (1.0, 10.0)}),
            ("High Antioxidant Exploration", {'antioxidants_additives(%)': (0.5, 5.0)}),
            ("CPA Cocktail (Mixed CPAs)", {'other_cpas(%)': (1.0, 10.0)}),
            ("Trehalose Focus", {'trehalose(%)': (2.0, 15.0)}),
            ("FBS Heavy", {'fbs(%)': (5.0, 20.0)})
        ]
        
        # Ensure we don't exceed n
        selected_strategies = strategies[:n]
        
        print(f"--- algorithm calculating: Generating {len(selected_strategies)} Diverse Recipes ---")
        
        for name, constraints in selected_strategies:
            print(f"  > Optimizing Strategy: {name}...")
            
            # Create custom bounds based on constraints
            current_bounds = [(0.0, 20.0)] * len(self.features) # Default
            
            # Apply specific constraints
            for feat, (low, high) in constraints.items():
                if feat in self.features:
                    idx = self.features.index(feat)
                    current_bounds[idx] = (low, high)
            
            best_x, score = self.optimize(bounds=current_bounds, verbose=False)
            recipes.append({
                'strategy': name,
                'vector': best_x,
                'score': score
            })
            
        return recipes

        return recipes

    def suggest_batch_experiment(self):
        recipes = self.optimize_batch(n=8)
        
        # Save to file for the Feedback Loop Script
        # We need to construct the 'all_ingredient' string for future appending
        save_data = []
        
        print("\n  8 Target Recipes for Active Learning  ")
        print("==================================================")
        
        for i, r in enumerate(recipes):
            print(f"\nRecipe #{i+1}: {r['strategy']}")
            print(f"Predicted Viability: {r['score']:.2f}%")
            
            # Sort by concentration
            ingredients = list(zip(self.features, r['vector']))
            ingredients.sort(key=lambda x: x[1], reverse=True)
            
            active_ingredients = []
            ingredient_str_parts = []
            
            for name, conc in ingredients:
                if conc > 0.01:
                    # formatted like "5.00% dmso"
                    clean_name = name.replace('(%)', '').replace('(mM)', '')
                    str_part = f"{conc:.2f}% {clean_name}"
                    
                    active_ingredients.append(f"{name}: {conc:.2f}%")
                    ingredient_str_parts.append(str_part)
            
            # Create the string for Data_raw (e.g., "5.00% dmso + 2.00% trehalose")
            full_ingredient_str = " + ".join(ingredient_str_parts)
            
            print("  " + " + ".join(active_ingredients))
            print("-" * 30)
            
            save_data.append({
                'Recipe_ID': i+1,
                'Strategy': r['strategy'],
                'Predicted_Viability': r['score'],
                'Ingredient_String': full_ingredient_str
            })
            
        # Write to CSV
        pd.DataFrame(save_data).to_csv('latest_batch_recipes.csv', index=False)
        print("\n[System]: Saved recipes to 'latest_batch_recipes.csv' for the Feedback Tool.")
            
        print("\nNext Step:")
        print("1. Test these 8 diverse recipes.")
        print("2. Fills the black holes in the AI's knowledge.")
        print("3. Feed results back to Data_raw.xlsx!")

        return best_vector, predicted_viability

        return -(pred - penalty)

    def suggest_high_viability_batch(self, n=8):
        """
        Generates 8 recipes focused ONLY on High Viability.
        (Uses 17 grouped features)
        """
        recipes = []
        print(f"\n  Start Optimized Batch (High Viability Only, {n} variants)  ")
        
        for i in range(n):
            print(f"  > Optimizing Variant #{i+1}...")
            # Vary seed to find different local maxima
            vec, score = self.optimize(target_mode='max', seed=i*55)
            recipes.append({'strategy': f'Hybrid_HighVia_{i+1}', 'vector': vec, 'score': score})

        # Output and Save
        self._print_and_save_recipes(recipes)

    def _print_and_save_recipes(self, recipes):
        save_data = []
        print("\n  8 Optimized Target Recipes  ")
        print("==================================================")
        
        for i, r in enumerate(recipes):
            print(f"\nRecipe #{i+1}: {r['strategy']}")
            print(f"Predicted Viability: {r['score']:.2f}%")
            
            ingredients = list(zip(self.features, r['vector']))
            ingredients.sort(key=lambda x: x[1], reverse=True)
            
            active_ingredients = []
            ingredient_str_parts = []
            
            for name, conc in ingredients:
                if conc > 0.01:
                    clean_name = name.replace('(%)', '').replace('(mM)', '')
                    str_part = f"{conc:.2f}% {clean_name}"
                    active_ingredients.append(f"{name}: {conc:.2f}%")
                    ingredient_str_parts.append(str_part)
            
            full_ingredient_str = " + ".join(ingredient_str_parts)
            print("  " + " + ".join(active_ingredients))
            print("-" * 30)
            
            save_data.append({
                'Recipe_ID': i+1,
                'Strategy': r['strategy'],
                'Predicted_Viability': r['score'],
                'Ingredient_String': full_ingredient_str
            })
            
        pd.DataFrame(save_data).to_csv('latest_batch_recipes.csv', index=False)
        print("\n[System]: Saved recipes to 'latest_batch_recipes.csv'.")

    def optimize(self, bounds=None, target_mode='max', seed=None, diversity_ref=None, verbose=False):
        """
        target_mode: 'max' (default) or 'min'
        diversity_ref: vector to be far from
        """
        if bounds is None:
            bounds = [(0.0, 20.0)] * len(self.features)
            
        if verbose:
            print(f"--- Optimizing ({target_mode}) ---")

        # Custom objective wrapper
        def wrapper_objective(x):
            try:
                # 1. Prediction 
                input_df = pd.DataFrame([x], columns=self.features)
                pred = self.model.predict(input_df)[0]
                
                # 2. Constraints (Total < 50%)
                total = np.sum(x)
                penalty = 0
                if total > 50:
                    penalty += (total - 50) * 10
                
                # 3. Mode Logic
                # basic_score = pred - penalty # OLD BUGGY WAY
                
                if target_mode == 'min':
                    # We want to Minimize Viability (Find bad recipes). 
                    # DE minimizes the function.
                    # We want pred to be low.
                    # If penalty > 0 (violation), result should be HIGH (bad).
                    return pred + penalty
                else: 
                    # Maximize Viability.
                    # DE minimizes. So we want to minimize (-pred).
                    # If penalty > 0, result should be HIGH (bad).
                    # So (-pred) + penalty
                    
                    final_obj = -pred + penalty
                    
                    # 4. Diversity Bonus (if enabled)
                    if diversity_ref is not None:
                        dist = np.linalg.norm(x - diversity_ref)
                        # We want to MAXIMIZE distance -> Minimize negative distance
                        # Add to objective (weight it)
                        final_obj -= dist * 0.5 
                        
                    return final_obj
            except:
                return 1000 # Bad score on error

        result = differential_evolution(
            func=wrapper_objective,
            bounds=bounds,
            strategy='best1bin',
            maxiter=20,
            popsize=10,
            tol=0.1,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=seed
        )
        
        # Recover real score
        # We need to re-evaluate the prediction without penalties/inversions to show user truthful predictive score
        final_x = result.x
        input_df = pd.DataFrame([final_x], columns=self.features)
        real_pred = self.model.predict(input_df)[0]
        
        return final_x, real_pred

    def suggest_experiment(self):
        # Default to high viability batch
        self.suggest_high_viability_batch()

