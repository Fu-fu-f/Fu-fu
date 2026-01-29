#!/usr/bin/env python3
import pandas as pd
import os
import subprocess
import sys

# --- Config ---
RECIPE_FILE = 'latest_batch_recipes.csv'
RAW_DATA_FILE = 'Cryopreservative Data 2026.csv'

def run_cmd(cmd):
    print(f"\nRunning: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Error executing: {cmd}")
        return False
    return True

def main_loop():
    while True:
        print("\n" + "="*40)
        print("   CryoMN Active Learning Feedback Loop")
        print("="*40)
        print("1. View Current Batch Recipes (Last Generated)")
        print("2. Input New Experimental Results (Lab Validation)")
        print("3. Retrain and Generate Next Batch (Auto-Pipeline)")
        print("4. Exit")
        
        choice = input("\nSelect an option (1-4): ").strip()
        
        if choice == '1':
            if os.path.exists(RECIPE_FILE):
                df = pd.read_csv(RECIPE_FILE)
                print("\n--- Current Optimal Recipes ---")
                print(df.to_string(index=False))
            else:
                print("\nNo recipe file found. Run option 3 first.")
                
        elif choice == '2':
            if not os.path.exists(RECIPE_FILE):
                print("\nNo recipe file found. Please generate recipes first.")
                continue
                
            df = pd.read_csv(RECIPE_FILE)
            print("\n--- Inputing Lab Results ---")
            
            new_data_rows = []
            for idx, row in df.iterrows():
                print(f"\nRecipe ID: {row['Recipe_ID']}")
                print(f"Ingredients: {row['Ingredients']}")
                v = input("Enter Observed Viability % (Leave blank to skip this recipe): ").strip()
                
                if v:
                    try:
                        v_float = float(v)
                        # Construct a row for the main Database
                        new_row = {
                            'All ingredients in cryoprotective solution': row['Ingredients'],
                            'Viability': f"{v_float}%",
                            'Source': 'Lab',
                            'Cooling rate': 'slow freeze' # Default for validation
                        }
                        new_data_rows.append(new_row)
                    except ValueError:
                        print("Invalid input. Skipping.")
            
            if new_data_rows:
                # Load raw data and append
                try:
                    raw_df = pd.read_csv(RAW_DATA_FILE)
                except:
                    print("Could not read raw data file.")
                    continue
                
                added_df = pd.DataFrame(new_data_rows)
                # Align columns and append
                updated_df = pd.concat([raw_df, added_df], ignore_index=True)
                updated_df.to_csv(RAW_DATA_FILE, index=False)
                print(f"\nSUCCESS: {len(new_data_rows)} new lab records saved to '{RAW_DATA_FILE}'.")
                print("Note: These records are marked as 'Lab' source for High Trust logic.")
            else:
                print("\nNo new results entered.")
                
        elif choice == '3':
            print("\n--- Executing Integrated Pipeline ---")
            if not run_cmd("python3 clean_data_llm.py"): continue
            if not run_cmd("python3 run_training.py"): continue
            if not run_cmd("python3 run_optimization.py"): continue
            print("\nSUCCESS: Model update and next-gen optimization complete.")
            
        elif choice == '4':
            print("Exiting. Good luck with your experiments!")
            break
        else:
            print("Invalid selection.")

if __name__ == "__main__":
    main_loop()
