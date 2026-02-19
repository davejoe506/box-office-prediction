import subprocess
import sys
import os

# --- Path configuration ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")

# The exact order the pipeline must run in
PIPELINE = [
    "01_fetch_data.py",
    "02_clean_data.py",
    "03_feature_engineering.py",
    "04_eda.py",
    "05_modeling.py"
]

def main():
    print("Starting box office prediction pipeline...")
    print("=" * 50)
    
    for script_name in PIPELINE:
        script_path = os.path.join(SCRIPTS_DIR, script_name)
        
        if not os.path.exists(script_path):
            print(f"Error: Could not find {script_name} in {SCRIPTS_DIR}")
            sys.exit(1)
            
        print(f"\nRunning {script_name}...")
        
        # Execute script and wait for it to finish
        result = subprocess.run([sys.executable, script_path])
        
        # Fail-safe: If the script crashed, stop whole pipeline
        if result.returncode != 0:
            print(f"\nPipeline failed during {script_name}. Stopping execution.")
            sys.exit(1)
            
    print("\n" + "=" * 50)
    print("Pipeline executed successfully.")
    print("Check the 'visualizations' folder for new charts.")
    print("Run 'streamlit run app.py' to test the final model.")

if __name__ == "__main__":
    main()