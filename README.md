# Cell Cryopreservation Optimization

This project focuses on optimizing cell cryopreservation protocols by leveraging machine learning models to predict cell viability and recovery rates. It includes scripts for data preprocessing, model training (Random Forest and XGBoost), and optimization using both Differential Evolution and Bayesian Optimization techniques.

## Project Structure

*(Other files like `Beginning.py`, `convert.py`, `first_process.py`, `second_process.py` are helper scripts for the data processing pipeline)*

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Fu-fu-f/Cell-Cryopreservation-Optimization.git
    cd Cell-Cryopreservation-Optimization
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run the Project

The project involves a sequence of steps from data processing to model training and finally optimization.

### 1. Data Preprocessing

The initial data is in `Data_raw.xlsx`. The following scripts process it sequentially:

-   `Beginning.py`: Performs initial cleaning.
-   `convert.py`: Converts the Excel file to CSV (`Data_raw.csv`).
-   `first_process.py`: Expands ingredient columns (`processed_data.csv`).
-   `second_process.py`: Combines similar columns to produce `final_data.csv`.

You can run them in order, although the final processed data (`final_data.csv`) is already included in the repository.

### 2. Model Training

You can train the models yourself or use the pre-trained models provided.

-   **To train XGBoost models:**
    ```bash
    python XGboost/train_models.py
    ```
    This will generate `Recovery_model.joblib` and `Viability_model.joblib` inside `XGboost/trained_models/`.

-   **To train Random Forest models:**
    ```bash
    python "Algorithm random forest/train_rf_models.py"
    ```
    This will generate `Recovery_model_rf.joblib` and `Viability_model_rf.joblib` inside `Algorithm random forest/trained_rf_models/`.

### 3. Protocol Optimization

Once the models are trained (or using the provided ones), you can run the optimization scripts. These scripts are interactive and will prompt you for input.

-   **Differential Evolution with XGBoost models:**
    ```bash
    python XGboost/differential_evolution_optimization.py
    ```

-   **Differential Evolution with Random Forest models:**
    ```bash
    python "Algorithm random forest/random_forest.py"
    ```

-   **Bayesian Optimization with XGBoost models:**
    ```bash
    python xgboost_bayesian/xgboost_bayesian_optimization.py
    ```