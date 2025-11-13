<!-- # ⚽ Model Card: xGenius — Machine Learning for Soccer Player Performance Prediction

### 🧩 Overview
**xGenius** is a **predictive machine learning model** designed to evaluate and forecast **professional soccer players’ performance** using advanced metrics such as Expected Goals (xG) and actual goals (aG).  
By comparing predicted xG with real-world outcomes, **xGenius** identifies efficient and underperforming players across Europe’s top 5 leagues — helping coaches, analysts, and clubs make more informed, data-driven decisions.

This project combines **data science, feature engineering, and explainable AI (SHAP)** to transform complex performance data into actionable insights for player evaluation, scouting, and salary prediction.

---

### 🎯 Objectives
- Predict **Expected Goals (xG)** using performance metrics  
- Compare predicted xG with **actual goals (aG)** to measure player efficiency  
- Analyze **feature importance** using SHAP for transparency  
- Visualize player performance trends across leagues and positions  

---

### ⚙️ Tech Stack
| Category | Tools |
|-----------|-------|
| **Programming** | Python, Pandas, NumPy |
| **Modeling** | Scikit-learn, XGBoost, Random Forest, Linear Regression |
| **Visualization** | Matplotlib, Seaborn |
| **Explainability** | SHAP (SHapley Additive exPlanations) |
| **Version Control** | Git, GitHub |
| **Environment** | VS Code, Jupyter Notebook |

---

### 📊 Dataset
**Data Source:** Public football statistics from Europe’s Top 5 Leagues  
(Premier League, La Liga, Bundesliga, Serie A, Ligue 1)

**Sample Features:**
- `time`, `games`, `goals`, `xG`, `assists`, `key_passes`, `shots`,  
- `xGChain`, `xGBuildup`, `npxG`, `position_ratio`, `league_ratio`

**Engineered Metrics:**
- `gpg` — Goals per game  
- `xGg` — Expected goals per game  
- `xGdiff` — (xGg - aGg) performance difference  
- `weekly_gross_base_salary_gbp` — Salary impact variable  

---

### 🧮 Model Pipeline
1. **Data Cleaning** – Handle missing & inconsistent data  
2. **Feature Engineering** – Create efficiency and position-based ratios  
3. **Feature Selection** – Correlation threshold (±0.5)  
4. **Model Training** – Train multiple models for xGg and aGg prediction  
5. **Evaluation** – Compare results using R², MAE, RMSE  
6. **Explainability** – Use SHAP values for feature contribution analysis  

---

### 📈 Results
- The model accurately predicts Expected Goals (xGg) with strong correlation to actual goals (aGg).  
- Players with higher **xGg − aGg** differences show lower real-world efficiency.  
- SHAP analysis highlights `shots`, `xGChain`, and `key_passes` as key contributors to xG prediction.  

---

### 🔍 Key Visualizations
- 📊 Correlation Heatmap of key metrics  
- 🧮 SHAP summary plot showing top impactful features  
- ⚽ Top 10 most efficient and least efficient players  
- 💰 Regression analysis between xGg and weekly salary  

---

### 🧠 Explainability with SHAP
SHAP (SHapley Additive exPlanations) helps understand **why** the model makes certain predictions:  
- Displays each feature’s contribution to predicted xG  
- Enables transparent decision-making in player evaluation  

---

### 🗂 Project Structure

Here is a brief explanation of each folder and key file in the project:

- [`Data/`](./Data/) – All datasets used for training and analysis  
  - `raw/` – Original raw CSVs for each league (`SerieA`, `LaLiga`, `PremierLeague`, `Bundesliga`, `Ligue1`)  
  - `preprocessed/` – Cleaned and processed data ready for modeling  
  - `final/` – Final merged dataset used for feature engineering and training  
  > For more detailed information about the datasets, data sources, and processing steps, please see the [README inside the Data folder](./Data/README.md).

- [`Notebooks/`](./Notebooks/) – Jupyter notebooks for exploration, feature engineering, modeling, and evaluation  
  - `01_DataAnalysis.ipynb` – Explore and visualize raw data  
  - `02_DataPreProcessing.ipynb` – Clean, handle missing values, and prepare data  
  - `03_FeatureEngineering.ipynb` – Create new features like `xGdiff`, `position_ratio`  
  - `04_ModelPrototyping.ipynb` – Train multiple ML models quickly  
  - `05_ModelEvaluation.ipynb` – Evaluate models, generate charts, and compare results  

- [`src/`](./src/) – Source code with reusable classes and functions  
  - `data_loader.py` – Load and merge multiple CSV files into a single dataset  
  - `data_preprocessing.py` – Handle cleaning, missing values, and basic transformations  
  - `feature_engineering.py` – Create new features and perform transformations  
  - `modelprototyping.py` – Train models like Linear Regression, Random Forest, Stacking  
  - `evaluation.py` – Evaluate models systematically (metrics, plots)  
  - `utils.py` – Helper functions for plotting, saving/loading files, and other reusable utilities  
  - `__init__.py` – Makes the `src` folder a Python package  

- [`models/`](./models/) – Store trained ML models and preprocessing objects (`.pkl`)  
- [`results/`](./results/) – Store evaluation outputs, tables, metrics, and logs  
- [`reports/`](./reports/) – Figures, plots, and PDF reports for presentations or publications  
- [`requirements.txt`](./requirements.txt) – List of required Python packages  
- [`LICENSE`](./LICENSE) – Project license  

---

### 🚀 Future Work
- Integrate **time-series analysis** for season-long trends  
- Expand to **team-level efficiency modeling**  
- Build a **web dashboard** for visual insights  
- Explore **Deep Learning (LSTM / Transformer)** for player performance forecasting  

---

### 🧾 Citation
If you use this project for academic or portfolio purposes, please cite:

> **Malikov, D. (2025). xGenius: Machine Learning for Soccer Player Performance Prediction. GitHub Repository.**  
> [https://github.com/davronbekmalikov/xGenius-Machine-Learning-for-Soccer-Player-Performance-Prediction](https://github.com/davronbekmalikov/xGenius-Machine-Learning-for-Soccer-Player-Performance-Prediction)  

---

### 💼 Author
**👤 Davronbek Malikov**  
Machine Learning & Deep Learning Engineer  
📍 South Korea  
📧 [Your Email Here]  
🔗 [LinkedIn Profile or Portfolio Link]

---

> 🧠 *xGenius — Smart Analytics for Smarter Football Decisions* -->


# ⚽ Model Card: xGenius — Machine Learning for Soccer Player Performance Prediction

### 🧩 Overview
**xGenius** is a **predictive machine learning model** designed to evaluate and forecast **professional soccer players’ performance** using advanced metrics such as Expected Goals (xG) and actual goals (aG).  
By comparing predicted xG with real-world outcomes, **xGenius** identifies efficient and underperforming players across Europe’s top 5 leagues — helping coaches, analysts, and clubs make more informed, data-driven decisions.

This project combines **data science, feature engineering, and explainable AI (SHAP)** to transform complex performance data into actionable insights for player evaluation, scouting, and salary prediction.

---

### 🎯 Objectives
- Predict **Expected Goals (xG)** using performance metrics  
- Compare predicted xG with **actual goals (aG)** to measure player efficiency  
- Analyze **feature importance** using SHAP for transparency  
- Visualize player performance trends across leagues and positions  

---

### ⚙️ Tech Stack
| Category | Tools |
|-----------|-------|
| **Programming** | Python, Pandas, NumPy |
| **Modeling** | Scikit-learn, XGBoost, Random Forest, Linear Regression |
| **Visualization** | Matplotlib, Seaborn |
| **Explainability** | SHAP (SHapley Additive exPlanations) |
| **Version Control** | Git, GitHub |
| **Environment** | VS Code, Jupyter Notebook |

---

### 📊 Dataset
**Data Source:** Public football statistics from Europe’s Top 5 Leagues  
(Premier League, La Liga, Bundesliga, Serie A, Ligue 1)

**Sample Features:**
- `time`, `games`, `goals`, `xG`, `assists`, `key_passes`, `shots`,  
- `xGChain`, `xGBuildup`, `npxG`, `position_ratio`, `league_ratio`

**Engineered Metrics:**
- `gpg` — Goals per game  
- `xGg` — Expected goals per game  
- `xGdiff` — (xGg - aGg) performance difference  
- `weekly_gross_base_salary_gbp` — Salary impact variable  

---

### 🧮 Model Pipeline
1. **Data Cleaning** – Handle missing & inconsistent data  
2. **Feature Engineering** – Create efficiency and position-based ratios  
3. **Feature Selection** – Correlation threshold (±0.5)  
4. **Model Training** – Train multiple models for xGg and aGg prediction  
5. **Evaluation** – Compare results using R², MAE, RMSE  
6. **Explainability** – Use SHAP values for feature contribution analysis  

---

### 📈 Results
- The model accurately predicts Expected Goals (xGg) with strong correlation to actual goals (aGg).  
- Players with higher **xGg − aGg** differences show lower real-world efficiency.  
- SHAP analysis highlights `shots`, `xGChain`, and `key_passes` as key contributors to xG prediction.  

---

### 🔍 Key Visualizations
- 📊 Correlation Heatmap of key metrics  
- 🧮 SHAP summary plot showing top impactful features  
- ⚽ Top 10 most efficient and least efficient players  
- 💰 Regression analysis between xGg and weekly salary  

---

### 🧠 Explainability with SHAP
SHAP (SHapley Additive exPlanations) helps understand **why** the model makes certain predictions:  
- Displays each feature’s contribution to predicted xG  
- Enables transparent decision-making in player evaluation  

---

### 🗂 Project Structure

Here is a brief explanation of each folder and key file in the project:

- [`Data/`](./Data/) – All datasets used for training and analysis  
  - `raw/` – Original raw CSVs for each league (`SerieA`, `LaLiga`, `PremierLeague`, `Bundesliga`, `Ligue1`)  
  - `preprocessed/` – Cleaned and processed data ready for modeling  
  - `final/` – Final merged dataset used for feature engineering and training  
  > For more detailed information about the datasets, data sources, and processing steps, please see the [README inside the Data folder](./Data/README.md).

- [`Notebooks/`](./Notebooks/) – Jupyter notebooks for exploration, feature engineering, modeling, and evaluation  
  - `01_DataAnalysis.ipynb` – Explore and visualize raw data  
  - `02_DataPreProcessing.ipynb` – Clean, handle missing values, and prepare data  
  - `03_FeatureEngineering.ipynb` – Create new features like `xGdiff`, `position_ratio`  
  - `04_ModelPrototyping.ipynb` – Train multiple ML models quickly  
  - `05_ModelEvaluation.ipynb` – Evaluate models, generate charts, and compare results  

- [`src/`](./src/) – Source code with reusable classes and functions  
  - `data_loader.py` – Load and merge multiple CSV files into a single dataset  
  - `data_preprocessing.py` – Handle cleaning, missing values, and basic transformations  
  - `feature_engineering.py` – Create new features and perform transformations  
  - `modelprototyping.py` – Train models like Linear Regression, Random Forest, Stacking  
  - `evaluation.py` – Evaluate models systematically (metrics, plots)  
  - `utils.py` – Helper functions for plotting, saving/loading files, and other reusable utilities  
  - `__init__.py` – Makes the `src` folder a Python package  

- [`models/`](./models/) – Store trained ML models and preprocessing objects (`.pkl`)  
- [`results/`](./results/) – Store evaluation outputs, tables, metrics, and logs  
- [`reports/`](./reports/) – Figures, plots, and PDF reports for presentations or publications  
- [`requirements.txt`](./requirements.txt) – List of required Python packages  
- [`LICENSE`](./LICENSE) – Project license  

---

### 🚀 Future Work
- Integrate **time-series analysis** for season-long trends  
- Expand to **team-level efficiency modeling**  
- Build a **web dashboard** for visual insights  
- Explore **Deep Learning (LSTM / Transformer)** for player performance forecasting  

---

### 🧾 Citation
If you use this project for academic or portfolio purposes, please cite:

> **Malikov, D. (2025). xGenius: Machine Learning for Soccer Player Performance Prediction. GitHub Repository.**  
> [https://github.com/davronbekmalikov/xGenius-Machine-Learning-for-Soccer-Player-Performance-Prediction](https://github.com/davronbekmalikov/xGenius-Machine-Learning-for-Soccer-Player-Performance-Prediction)  

---

### 💼 Author
**👤 Davronbek Malikov**  
Machine Learning & Deep Learning Engineer  
📍 South Korea  
📧 [Your Email Here]  
🔗 [LinkedIn Profile or Portfolio Link]

---

> 🧠 *xGenius — Smart Analytics for Smarter Football Decisions*
