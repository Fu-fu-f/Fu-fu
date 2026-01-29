# run_training.py
import sys
import os

# Ensure we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model_trainer import ModelTrainer

def main():
    print("=== Starting Model Training Pipeline ===")
    
    try:
        trainer = ModelTrainer(models_dir='trained_models')
        trainer.train()
        
        print("\nSUCCESS: Models trained and saved.")
        
    except Exception as e:
        print(f"\n❌ TRAINING ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
