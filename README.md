# Box Office Revenue Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://davejoe506-box-office.streamlit.app/)

**Live Web Application:** [Click here to launch the Box Office Predictor](https://davejoe506-box-office.streamlit.app/)

---

## Table of Contents
* [Project Context](#project-context)
* [Tech Stack](#tech-stack)
* [Project Structure](#project-structure)
* [Installation & Setup](#-installation--setup)
* [Data Pipeline & Modeling](#data-pipeline--modeling)
* [Usage (Streamlit Web Interface)](#usage-streamlit-web-interface)
* [Challenges, Limitations, and Future Work](#challenges-limitations-and-future-work)

---

## Project Context
This project applies machine learning to predict a movie's worldwide theatrical box office revenue based entirely on pre-release features (budget, genre, runtime, cast/director track records, release seasonality, etc). By building an automated data pipeline, this project transforms raw historical movie data into a predictive model designed to forecast blockbuster success."

Key insights from the Exploratory Data Analysis (EDA) revealed a strong linear relationship between production spend and financial success. The final XGBoost model, augmented with log transformation and SHAP value explainability, confirmed that an inflation-adjusted budget and an established franchise property are the strongest predictors of financial success.

---

## Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, XGBoost
* **Explainable AI:** SHAP
* **Visualization:** Matplotlib, Seaborn
* **Deployment:** Streamlit
* **APIs:** The Movie Database (TMDB) API, Python `cpi` library

---

## Project Structure

```text
box-office-prediction/
│
├── .env                   # API keys (not committed)
├── requirements.txt       # Python dependencies
├── README.md              # Documentation
│
├── data/                  # Data artifacts (ignored by Git)
│   ├── raw/               # Raw TMDB data
│   └── clean/             # Processed TMDB data
│
├── scripts/               # Sequential data pipeline and modeling scripts
│   ├── 01_fetch_data.py
│   ├── 02_clean_data.py
│   ├── 03_feature_engineering.py
│   ├── 04_eda.py
│   └── 05_modeling.py
│
├── visualizations/        # Generated EDA and SHAP charts
│
├── app.py                 # Web application (Streamlit)
└── run_pipeline.py        # Master script to run all steps
```

---

## Installation & Setup

1. **Environment Setup**<br/>
Clone the repo and create a virtual environment:

```bash
git clone [https://github.com/davejoe506/box-office-prediction.git](https://github.com/davejoe506/box-office-prediction.git)
cd box-office-prediction

python -m venv venv
source venv/bin/activate
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Install dependencies:

```bash
pip install -r requirements.txt
playwright install
```

2. **Configuration**  
Create a .env file in the root directory and add your TMDB API key:

```
TMDB_API_KEY=your_api_key_here
```

## Building the Database
To reproduce the entire data pipeline and train the XGBoost model from scratch, simply run the orchestration script from the root directory:

```bash
python run_pipeline.py
```

This master script sequentially executes the following architecture:
* `01_fetch_data.py`: Pulls raw metadata and financial data for over 5,000 theatrical releases using the TMDB API.
* `02_clean_data.py`: Handles missing values, engineers datetime objects, and standardizes financial metrics to 2024 US dollars using the Bureau of Labor Statistics CPI API.
* `03_feature_engineering.py`: One-hot encodes genres, maps release months to seasons, and calculates rolling historical box office averages for directors and top actors to prevent data leakage.
* `04_eda.py`: Generates visualizations to understand revenue distribution and feature correlations.
* `05_modeling.py`: Drops post-release data, applies a log transformation, trains an XGBoost Regressor (R² ~0.62), and generates SHAP values for model explainability.

---

## Usage (Streamlit Web Interface)

To run the interactive prediction app locally:

```bash
streamlit run app.py
```

This launches a web interface where users can input hypothetical movie details (budget, runtime, director's past box office, etc.) to generate real-time global revenue predictions.

---

## Challenges, Current Limitations & Future Work

### Challenges & Current Limitations ###
* **The Cold Start Problem:** Because the dataset begins in the year 2000, established actors and directors are treated as having no prior box office history for their first release in the dataset. A rolling historical average was used to allow the model to self-correct this talent score over time.
* **Missing Ghost Variables:** The model explains a significant portion of revenue variance using only pre-release metadata. However, it's plausible that the remaining variance is tied to proprietary studio data that is not publicly available, such as marketing budgets and test screening scores.

### Future Work ###
* **Resolving the Cold Start Problem:** Fetching historical TMDB data from 1980–1999 to allow the pipeline to more accurately calculate talent scores for veteran directors and actors before evaluating their post-2000 releases.
* **Incorporating Pre-Release Proxy Metrics:** Since exact marketing budgets are closely guarded studio secrets, future iterations could engineer proxy features for pre-release hype. This might involve scraping YouTube trailer views, Wikipedia page traffic, or Google Trends data in the 30 days prior to a premiere.
* **Model Optimization:** Implementing an optimization framework like GridSearchCV to systematically test and tune the XGBoost hyperparameters, specifically targeting a further reduction in the Mean Absolute Error (MAE).
**Cloud Orchestration:** Migrating the local `run_pipeline.py` execution to a robust data orchestration platform (such as Kestra deployed on GCP) to transition the project from a local script into a fully automated, cloud-hosted data pipeline.

---
