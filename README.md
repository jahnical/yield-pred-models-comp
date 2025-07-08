# Comparative Yield Models

This project provides a fully reproducible workflow for comparative analysis of crop yield estimation using remote sensing (Sentinel-2) and machine learning. It includes deep learning (ViT, CNN-LSTM), XGBoost, and linear regression baselines, with all code, data handling, and evaluation logic modularized for clarity and extensibility.

## Reproducibility Checklist

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd comparative-yield-models
   ```

2. **Set up the environment**
   - Create and activate a virtual environment:
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     ```
   - Install all dependencies:
     ```bash
     pip install -r requirements.txt
     ```
   - All package versions are pinned for consistency.

3. **Prepare the data**
   - Place Sentinel-2 TIFFs and CSVs in the correct folders (see below).
   - Example: imagery in `Yield_GEE_S2_ByField/`, metadata in `data/field_images.csv`.
   - Use the same data splits and CSVs as in the original experiments for exact reproducibility.

4. **Run experiments**
   - Example command to run a model:
     ```bash
     python src/main.py vit --csv data/field_images.csv --data_root Yield_GEE_S2_ByField
     ```
   - For hyperparameter tuning:
     ```bash
     python src/main.py vit_lstm --tune --csv data/field_images.csv --data_root Yield_GEE_S2_ByField
     ```
   - All results, metrics, and model checkpoints are saved with clear naming conventions.

5. **Results and analysis**
   - Metrics (RMSE, MAE, R², inference time) are computed using the same helper functions for all models.
   - Cross-validation is deterministic and based on field names.
   - Use the provided Jupyter notebook (`analysis.ipynb`) for statistical analysis and visualization of results.

6. **License**
   - This project is licensed under GPL-3.0. All derivative works must also be open source under the same license.

7. **Support**
   - For questions or issues, open an issue on the repository or contact the maintainer.
