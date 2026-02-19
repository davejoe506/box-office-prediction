import os
import time
import requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import json

# --- Path configuration ---
load_dotenv()
API_KEY = os.getenv("TMDB_API_KEY")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "tmdb_movies_raw.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not API_KEY:
    raise ValueError("API Key not found. Please check your .env file.")

def fetch_detailed_movies_by_year(year, pages_to_fetch=10):
    """
    1: Discover top movies for the year.
    2: Fetch financial data + CAST/CREW + FRANCHISE data.
    """
    discover_url = "https://api.themoviedb.org/3/discover/movie"
    details_url = "https://api.themoviedb.org/3/movie/{}"
    
    detailed_movies = []
    
    for page in tqdm(range(1, pages_to_fetch + 1), desc=f"Year {year}"):
        discover_params = {
            "api_key": API_KEY,
            "language": "en-US",
            "sort_by": "popularity.desc",
            "primary_release_year": year,
            "page": page,
            "with_release_type": "3|2", # Theatrical releases
            "region": "US"
        }
        
        try:
            # 1. Get the list of movies
            res = requests.get(discover_url, params=discover_params)
            if res.status_code != 200:
                time.sleep(2)
                continue
                
            results = res.json().get("results", [])
            
            # 2. Get the financial + credits details for each movie
            for movie in results:
                movie_id = movie["id"]
                
                detail_params = {
                    "api_key": API_KEY, 
                    "language": "en-US",
                    "append_to_response": "credits" 
                }
                
                detail_res = requests.get(details_url.format(movie_id), params=detail_params)
                
                if detail_res.status_code == 200:
                    data = detail_res.json()
                    
                    # To keep CSV from becoming too large, only extract what is needed
                    movie_dict = {
                        "id": data.get("id"),
                        "title": data.get("title"),
                        "release_date": data.get("release_date"),
                        "budget": data.get("budget"),
                        "revenue": data.get("revenue"),
                        "runtime": data.get("runtime"),
                        "popularity": data.get("popularity"),
                        "vote_average": data.get("vote_average"),
                        "vote_count": data.get("vote_count"),
                        "original_language": data.get("original_language"),
                        
                        # Store complex JSON lists as strings to parse later
                        "genres": json.dumps(data.get("genres", [])),
                        "belongs_to_collection": json.dumps(data.get("belongs_to_collection")),
                        "cast": json.dumps(data.get("credits", {}).get("cast", [])),
                        "crew": json.dumps(data.get("credits", {}).get("crew", []))
                    }
                    
                    detailed_movies.append(movie_dict)
                
                # TMDB rate limit respect
                time.sleep(0.05)
                
        except Exception as e:
            print(f"Error on year {year} page {page}: {e}")
            
    return detailed_movies

def main():
    all_movies = []
    start_year = 2000
    end_year = 2024
    
    print(f"Starting TMDB data fetch with credits & franchises ({start_year}-{end_year})")
    print("-" * 50)

    for year in range(start_year, end_year + 1):
        year_data = fetch_detailed_movies_by_year(year, pages_to_fetch=10)
        all_movies.extend(year_data)
        
        # Save incremental progress
        temp_df = pd.DataFrame(all_movies)
        temp_df.to_csv(OUTPUT_FILE, index=False)
        
    print("-" * 50)
    print(f"Download complete.")
    print(f"Total detailed movies fetched: {len(all_movies)}")
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()