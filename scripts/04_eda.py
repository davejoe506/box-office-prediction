import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Path configuration ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "clean", "tmdb_movies_features.csv")
VIZ_DIR = os.path.join(PROJECT_ROOT, "visualizations")

os.makedirs(VIZ_DIR, exist_ok=True)

def main():
    print("Starting Exploratory Data Analysis...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return
        
    df = pd.read_csv(INPUT_FILE)
    sns.set_theme(style="whitegrid")
    
    revenue_m = df['revenue_adj'] / 1_000_000
    budget_m = df['budget_adj'] / 1_000_000

    # Charts 1 & 2: Distributions & ROI
    print("Generating charts for distribution & ROI...")
    plt.figure(figsize=(10, 6))
    sns.histplot(revenue_m, bins=50, kde=True, color='forestgreen')
    plt.title('Distribution of Global Box Office Revenue (Adjusted)', fontsize=16)
    plt.xlabel('Global Revenue (Millions, 2024 US Dollars)')
    plt.xlim(0, 1500)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "01_revenue_distribution.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=budget_m, y=revenue_m, alpha=0.5, color='royalblue')
    max_val = min(budget_m.max(), 400)
    plt.plot([0, max_val], [0, max_val * 2], color='red', linestyle='--', label='Break-Even (2x Budget)')
    plt.title('Adjusted Budget vs. Adjusted Global Revenue', fontsize=16)
    plt.xlabel('Budget (Millions, 2024 US Dollars)')
    plt.ylabel('Global Revenue (Millions, 2024 US Dollars)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "02_budget_vs_revenue.png"), dpi=300)
    plt.close()

    # Chart 3: Correlation Heatmap
    print("Generating correlation heatmap...")
    plt.figure(figsize=(10, 8))
    cols_for_corr = [
        'budget_adj', 'revenue_adj', 'runtime', 
        'is_franchise', 'director_score', 'actor_score'
    ]
    # Only keep columns that actually exist in the dataframe
    cols_for_corr = [c for c in cols_for_corr if c in df.columns]
    
    corr_matrix = df[cols_for_corr].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
    plt.title('Feature Correlation Heatmap', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "03_correlation_heatmap.png"), dpi=300)
    plt.close()

    # Chart 4: Seasonality
    if 'season' in df.columns:
        print("Generating seasonality boxplot...")
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df['season'], y=revenue_m, hue=df['season'], order=['Summer_Blockbuster', 'Holiday_Season', 'Spring_Fall', 'Dump_Months'], palette="Set2", legend=False)
        plt.title('Adjusted Global Revenue by Release Season', fontsize=16)
        plt.ylabel('Global Revenue (Millions, 2024 US Dollars)')
        plt.xlabel('Release Season')
        plt.ylim(0, 1000)
        plt.tight_layout()
        plt.savefig(os.path.join(VIZ_DIR, "04_seasonality_boxplot.png"), dpi=300)
        plt.close()

    # Chart 5: Runtime
    if 'runtime' in df.columns:
        print("Generating runtime vs. revenue scatterplot...")
        plt.figure(figsize=(10, 6))
        valid_runtime = df[(df['runtime'] > 60) & (df['runtime'] < 240)]
        sns.scatterplot(x=valid_runtime['runtime'], y=valid_runtime['revenue_adj'] / 1_000_000, alpha=0.5, color='purple')
        plt.title('Movie Runtime vs. Adjusted Global Revenue', fontsize=16)
        plt.xlabel('Runtime (Minutes)')
        plt.ylabel('Global Revenue (Millions, 2024 US Dollars)')
        plt.tight_layout()
        plt.savefig(os.path.join(VIZ_DIR, "05_runtime_vs_revenue.png"), dpi=300)
        plt.close()

    # Chart 6: Franchise Premium
    if 'is_franchise' in df.columns:
        print("Generating franchise boxplot...")
        plt.figure(figsize=(8, 6))
        df['Franchise_Label'] = df['is_franchise'].map({1: 'Franchise / Sequel', 0: 'Original / Standalone'})
        sns.boxplot(x=df['Franchise_Label'], y=revenue_m, hue=df['Franchise_Label'], palette="Set1", legend=False)
        plt.title('The Franchise Premium: Adjusted Global Revenue Comparison', fontsize=16)
        plt.ylabel('Global Revenue (Millions, 2024 US Dollars)')
        plt.xlabel('')
        plt.ylim(0, 1500)
        plt.tight_layout()
        plt.savefig(os.path.join(VIZ_DIR, "06_franchise_premium.png"), dpi=300)
        plt.close()

    # Chart 7: Director Track Record
    if 'director_score' in df.columns:
        print("Generating director track record scatter...")
        plt.figure(figsize=(10, 6))
        # Remove directors with 0 score (first time directors) so they don't crowd the y-axis
        established_directors = df[df['director_score'] > 0]
        sns.scatterplot(x=established_directors['director_score'], y=established_directors['revenue_adj'] / 1_000_000, alpha=0.5, color='darkorange')
        
        # Add a trendline
        sns.regplot(x=established_directors['director_score'], y=established_directors['revenue_adj'] / 1_000_000, scatter=False, color='black', line_kws={"linestyle": "--"})
        
        plt.title('Director Track Record vs. Adjusted Global Revenue', fontsize=16)
        plt.xlabel('Director Rolling Average Revenue Prior to Release (Millions, US Dollars)')
        plt.ylabel('Global Revenue (Millions, 2024 US Dollars)')
        plt.tight_layout()
        plt.savefig(os.path.join(VIZ_DIR, "07_director_track_record.png"), dpi=300)
        plt.close()

    print("-" * 30)
    print(f"EDA complete. Check the '{VIZ_DIR}' folder.")

if __name__ == "__main__":
    main()