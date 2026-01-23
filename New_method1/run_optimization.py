# run_optimization.py
import sys
import os

# Ensure we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ingredient_suggestion import Recommender

def main():
    try:
        rec = Recommender()
        print("--- Mode: Top 8 Strict High Viability ---")
        rec.suggest_batch_experiment()
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
