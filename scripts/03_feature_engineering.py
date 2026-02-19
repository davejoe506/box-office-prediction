import pandas as pd
import ast
import os
import json

# --- Path configuration ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "clean", "tmdb_movies_clean.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "clean", "tmdb_movies_features.csv")

def get_genre_names(genre_str):
    """Parses genre data to extract genre."""
    try:
        if pd.isna(genre_str): return []
        genres = json.loads(genre_str)
        return [g['name'] for g in genres]
    except:
        return []

def get_season(month):
    """Categorizes movies based on the movie season they are released."""
    if pd.isna(month): return 'Unknown'
    month = int(month)
    if month in [5, 6, 7]: return 'Summer_Blockbuster'
    elif month in [11, 12]: return 'Holiday_Season'
    elif month in [1, 2, 8, 9]: return 'Dump_Months'
    else: return 'Spring_Fall'

def get_director(crew_str):
    """Parses crew JSON to find director."""
    try:
        if pd.isna(crew_str): return "Unknown"
        crew = json.loads(crew_str)
        for member in crew:
            if member.get('job') == 'Director':
                return member.get('name')
        return "Unknown"
    except:
        return "Unknown"

def get_top_actor(cast_str):
    """Parses cast JSON to find #1 billed actor."""
    try:
        if pd.isna(cast_str): return "Unknown"
        cast = json.loads(cast_str)
        if len(cast) > 0:
            return cast[0].get('name')
        return "Unknown"
    except:
        return "Unknown"

def is_franchise(collection_str):
    """Checks if movie belongs to a larger franchise collection."""
    try:
        if pd.isna(collection_str) or collection_str == 'null': return 0
        coll = json.loads(collection_str)
        if coll and isinstance(coll, dict): return 1
        return 0
    except:
        return 0

def main():
    print("Starting feature engineering...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return
        
    df = pd.read_csv(INPUT_FILE)
    
    # 1. Genres & Seasons
    print("Processing genres & seasons...")
    df['genre_list'] = df['genres'].apply(get_genre_names)
    genre_dummies = df['genre_list'].str.join('|').str.get_dummies()
    genre_dummies = genre_dummies.add_prefix('genre_')
    df = pd.concat([df, genre_dummies], axis=1)
    
    df['season'] = df['release_month'].apply(get_season)
    season_dummies = pd.get_dummies(df['season'], prefix='season').astype(int)
    df = pd.concat([df, season_dummies], axis=1)

    # 2. Franchise Flag
    print("Tagging franchises...")
    df['is_franchise'] = df['belongs_to_collection'].apply(is_franchise)

    # 3. Extract talent
    print("Extracting directors and top cast...")
    df['director'] = df['crew'].apply(get_director)
    df['top_actor'] = df['cast'].apply(get_top_actor)

    # 4. Prevent data leakage by using rolling historical averages
    print("Calculating historical star power (rolling averages)...")
    # Must sort by date first, so that only past data is used
    df = df.sort_values('release_date').reset_index(drop=True)
    
    director_rev, director_count = {}, {}
    actor_rev, actor_count = {}, {}
    dir_scores, actor_scores = [], []
    
    for idx, row in df.iterrows():
        d = row['director']
        a = row['top_actor']
        rev = row['revenue_adj']
        
        # Director score
        if d != "Unknown" and d in director_rev:
            # Average of all past movies
            dir_scores.append(director_rev[d] / director_count[d])
            director_rev[d] += rev
            director_count[d] += 1
        else:
            dir_scores.append(0) # First time this director is seen, score is 0
            if d != "Unknown":
                director_rev[d] = rev
                director_count[d] = 1
                
        # Actor score
        if a != "Unknown" and a in actor_rev:
            actor_scores.append(actor_rev[a] / actor_count[a])
            actor_rev[a] += rev
            actor_count[a] += 1
        else:
            actor_scores.append(0)
            if a != "Unknown":
                actor_rev[a] = rev
                actor_count[a] = 1

    # Convert to millions for the model to digest easily
    df['director_score'] = pd.Series(dir_scores) / 1_000_000
    df['actor_score'] = pd.Series(actor_scores) / 1_000_000

    # 5. Clean up
    cols_to_drop = [
        'genres', 'genre_list', 'release_date', 'belongs_to_collection', 
        'cast', 'crew', 'director', 'top_actor'
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # 6. Save
    df.to_csv(OUTPUT_FILE, index=False)
    print("-" * 30)
    print("Feature engineering complete.")
    print("Added 'is_franchise', 'director_score', and 'actor_score'.")
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()