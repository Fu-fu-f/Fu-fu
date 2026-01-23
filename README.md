Cryoprotectant Optimization System


Project Structure

New_method1 (The Top 8 System)
Feature: 8 Dimensions.
    *   Strictly filters data. Only 8 ingredients exist: `DMSO`, `EG`, `Glycerol`, `Trehalose`, `Sucrose`, `Mannitol`, `Creatine`, `FBS`.
Result Style: High-concentration recipes, predicting ~98% viability (aggressive).

New_method2 (The Hybrid System)

Features: 17 Dimensions.
    *   It uses the Top 8 ingredients + Chemical Groups.
    *   Rare ingredients are grouped into `Other_Sugars`, `Total_PEG`, `Other_Polymers`, etc.
Best For: Discovery (探索). When you want to see if adding complex additives (like PEG or specific Polymers) helps.
Result Style: Complex recipes, often predicting ~75-80% viability (cautious).

---

How to run 

Both folders work exactly the same way. Choose your folder (`cd New_method1` or `cd New_method2`) and follow this loop:

Step 1:
python run_optimization.py

Output: Generates 8 optimized recipes.
File: Saves them to `latest_batch_recipes.csv`.

Step 2: Lab Experiment
Go to the lab. Test the 8 recipes.
Record the Viability (e.g., 85).

Step 3: Feedback & train

python run_feedback.py

Input: Enter your experimental results interactive.
Automation: The system will automatically:
    1.  Update `Data_raw.xlsx` (The main Database).
    2.  Clean data (`run_pipeline.py`).
    3.  Retrain models (`run_training.py`).
    4.  Generate Generation N+1 recipes (`run_optimization.py`).

---

System Architecture

Data_raw.xlsx:database.
'src/ingredient_suggestion.py': The Optimization Engine (Differential Evolution).
'src/model_trainer.py': The Prediction Engine (Stacking Regressor: XGBoost + Random Forest).
'src/config.p' and 'src/data_processor.py': The Rulebook (Chemical groupings and settings).

---

Note
Safety: New_method1 often suggests high concentrations (e.g., 20% DMSO). need to verify solubility and toxicity limits.
Data: Never delete #Data_raw.xlsx# . You can fix wrong entries manually in Excel and rerun #run_feedback.py#.