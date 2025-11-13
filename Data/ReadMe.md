# 📂 Data Folder — xGenius Project

The `Data/` folder contains all datasets used in the **xGenius Machine Learning project** for soccer player performance prediction.  
This folder is organized to **separate raw, preprocessed, and final datasets** to ensure **reproducibility, clarity, and ease of use**.

--- 
---

## **Subfolders Explained**

### 1️⃣ raw/
- Contains **unmodified, original datasets** downloaded from public football statistics sources.  
- Each CSV corresponds to one of the **Top 5 European leagues**.  
- **Purpose:** Keep original data intact for reproducibility.  
- **Example files:**
  - `PremierLeague.csv` — Player stats for Premier League  
  - `LaLiga.csv` — Player stats for La Liga  
  - `SerieA.csv`, `Bundesliga.csv`, `Ligue1.csv` — Other league datasets  

### 2️⃣ preprocessed/
- Contains **cleaned and transformed datasets** after initial preprocessing.  
- **Actions performed:**  
  - Handle missing values  
  - Standardize column names  
  - Normalize numeric metrics (e.g., minutes, shots, goals)  
  - Preliminary calculations of ratios or derived features if needed  
- **Purpose:** Provides **ready-to-merge data** for modeling and feature engineering.  

### 3️⃣ final/
- Contains the **merged dataset** combining all 5 leagues, ready for modeling.  
- **Characteristics:**  
  - Includes all leagues in a single DataFrame  
  - Cleaned, normalized, and ready for machine learning  
  - Contains engineered features used for model training:  
    - `xGg` — Expected Goals per Game  
    - `xGdiff` — Difference between xGg and actual goals  
    - `position_ratio` — Influence based on player position  
    - `league_ratio` — League strength factor  
- **Purpose:** This dataset is the main input for **src scripts and notebooks**.

---

## **Usage Notes**
- Always **start with the `raw/` data** if you want to reproduce preprocessing.  
- Use `preprocessed/` for **clean, intermediate datasets**.  
- Use `final/` for **direct modeling and evaluation**.  
- **Do not modify `raw/` files** to maintain reproducibility.

---

## **Data Source**
- Public football statistics for **Europe’s Top 5 Leagues**.  
- Sources include official league websites and open football statistics repositories.

---

## **Tips for New Users**
- For **column definitions and transformations**, refer to:  
  - `01_DataAnalysis.ipynb` in the [`Notebooks/`](../Notebooks/) folder  
  - `data_loader.py` and `data_preprocessing.py` in [`src/`](../src/)  

---

> ⚠️ **Important:** Follow the pipeline from `raw/` → `preprocessed/` → `final/` to ensure consistency and reproducibility.
 



