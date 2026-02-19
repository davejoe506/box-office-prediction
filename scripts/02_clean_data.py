import pandas as pd
import numpy as np
import os
import cpi
from datetime import datetime

# --- Path configuration ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "tmdb_movies_raw.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "clean", "tmdb_movies_clean.csv")

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

def main():
    print("Starting data cleaning...")
    
    # 1. Update CPI data
    print("Updating CPI data from Bureau of Labor Statistics...")
    try:
        cpi.update()
    except Exception as e:
        print(f"Warning: Could not update CPI data ({e}). Using cached data.")
    
    # 2. Load data
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found at {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"Original row count: {len(df)}")
    
    # 3. Drop duplicates
    df.drop_duplicates(subset=['id'], inplace=True)
    
    # 4. Filter garbage data
    df = df[df['budget'] > 10000].copy() 
    df = df[df['revenue'] > 10000].copy()
    print(f"Rows after removing low budget/revenue: {len(df)}")
    
    # 5. Date engineering
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df.dropna(subset=['release_date'], inplace=True)
    
    df['release_year'] = df['release_date'].dt.year
    df['release_month'] = df['release_date'].dt.month
    df['release_day_of_week'] = df['release_date'].dt.day_name()
    
    # 6. Inflation adjustment
    print("Adjusting for inflation...")
    unique_years = df['release_year'].dropna().unique()
    inflation_multipliers = {}
    current_year = datetime.now().year
    
    for year in unique_years:
        y = int(year)
        if y >= current_year:
            inflation_multipliers[y] = 1.0
        else:
            try:
                inflation_multipliers[y] = cpi.inflate(1, y)
            except:
                inflation_multipliers[y] = 1.0
                
    df['inflation_factor'] = df['release_year'].map(inflation_multipliers)
    df['budget_adj'] = df['budget'] * df['inflation_factor']
    df['revenue_adj'] = df['revenue'] * df['inflation_factor']
    
    # 6. Save
    cols_to_keep = [
        'id', 'title', 'release_date', 'release_year', 'release_month', 
        'release_day_of_week', 'original_language', 'popularity', 
        'vote_average', 'vote_count', 'budget', 'revenue', 
        'budget_adj', 'revenue_adj', 'runtime', 'genres',
        'belongs_to_collection', 'cast', 'crew'
    ]
    
    final_cols = [c for c in cols_to_keep if c in df.columns]
    df = df[final_cols]
    
    df.to_csv(OUTPUT_FILE, index=False)
    print("-" * 30)
    print("Data cleaning complete.")
    print(f"Final dataset size: {len(df)} movies")
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()