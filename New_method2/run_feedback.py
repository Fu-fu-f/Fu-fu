# run_feedback.py
import pandas as pd
import os
import sys
import subprocess
from src import config

def main():
    print("=== 🔄 AI Scientist Feedback Loop 🔄 ===")
    print("This tool will help you input your experimental results and generate Generation 2 recipes.\n")

    # 1. Load pending recipes
    if not os.path.exists('latest_batch_recipes.csv'):
        print("❌ Error: No pending recipes found. Please run 'run_optimization.py' first.")
        return

    batch_df = pd.read_csv('latest_batch_recipes.csv')
    print(f"Found {len(batch_df)} pending experiments from the last batch.")
    
    new_data = []
    
    # 2. Interactive Input
    print("\nPlease enter the 'Viability' result for each recipe (0-100% or 0.0-1.0).")
    print("Press 'Enter' to skip a recipe if you haven't done it yet.\n")
    
    for _, row in batch_df.iterrows():
        print(f"\n🧪 Recipe #{row['Recipe_ID']} ({row['Strategy']})")
        print(f"   Shape: {row['Ingredient_String']}")
        
        while True:
            val = input(f"   >> Enter Viability Result: ").strip()
            
            if not val:
                print("   Skipped.")
                break
            
            try:
                # Handle inputs like "85", "85%", "0.85"
                clean_val = val.replace('%', '')
                viability_float = float(clean_val)
                # Heuristic: if user enters > 1.0, assume percentage (e.g. 85 -> 0.85). 
                # BUT wait, the model trained on what?
                # Let's check `Data_raw`. Usually viability is 0.0-1.0 or 0-100?
                # Let's save it exactly as typical.
                # Assuming original data was 0-1 range based on previous '0.79' output being "79%"? 
                # Actually previous output said "0.79%", which implies range is small.
                # Let's assume input should be consistent with existing data.
                # We will save it as is.
                
                new_row = {
                    'all_ingredient': row['Ingredient_String'],
                    'viability': viability_float,
                    'recovery': float('nan'), # Optional
                    'cooling rate': 'slow freeze', # Default assumption
                    'doubling time': 'Not explicitly stated'
                }
                new_data.append(new_row)
                print("   ✅ Recorded.")
                break
            except ValueError:
                print("   ❌ Invalid number. Please try again.")

    if not new_data:
        print("\nNo results entered. Exiting.")
        return

    # 3. Append to Data_raw.xlsx
    input_file = config.INPUT_FILE # e.g. 'Data_raw.xlsx'
    
    print(f"\n💾 Appending {len(new_data)} new results to {input_file}...")
    
    try:
        if input_file.endswith('.xlsx'):
            # Load existing
            df_existing = pd.read_excel(input_file)
            df_new = pd.DataFrame(new_data)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_excel(input_file, index=False)
        else:
            # CSV fallback
            df_existing = pd.read_csv(input_file)
            df_new = pd.DataFrame(new_data)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(input_file, index=False)
            
        print("Success! Database updated.")
        
    except Exception as e:
        print(f"❌ Failed to save data: {e}")
        return

    # 4. Auto-Run The Pipeline
    print("\n🚀 Triggering The Three Axes (Pipeline -> Training -> Optimization)...")
    
    steps = [
        ("Cleaning Data", "python run_pipeline.py"),
        ("Retraining Brain", "python run_training.py"),
        ("Generating Gen 2 Recipes", "python run_optimization.py")
    ]
    
    for name, cmd in steps:
        print(f"\n>>> Step: {name}...")
        ret = subprocess.call(cmd, shell=True)
        if ret != 0:
            print(f"❌ Error during {name}. Stopping.")
            return
            
    print("\n🎉 Cycle Complete! Generation 2 recipes are ready above.")

if __name__ == "__main__":
    main()
