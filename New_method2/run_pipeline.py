# run_pipeline.py
import sys
import os

# Ensure we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_processor import DataProcessor

def main():
    print("=== Starting Unified Cryobiology Pipeline ===")
    
    try:
        processor = DataProcessor(input_path='Data_raw.xlsx') # Or .csv
        
        # 1. Load & Clean Empty Rows
        processor.load_data()
        
        # 1b. Remove Rapid Freeze (Legacy Pre-filter)
        processor.remove_rapid_freeze()
        
        # 2. Parse Ingredients & Normalize Units
        processor.parse_ingredients()
        
        # 2b. Remove Impossible Sums (>100%)
        processor.remove_impossible_sums()
        
        # 3. Clean Target Values (Viability/Recovery)
        processor.clean_target_columns()
        
        # 3b. Remove Missing Targets
        processor.remove_missing_targets()
        
        # 4. Save
        processor.save(output_path='final_data.csv')
        
        print("\nSUCCESS: Data pipeline completed successfully.")
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
