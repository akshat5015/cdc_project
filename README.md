# Satellite Imagery-Based Property Valuation

A multimodal regression pipeline that predicts property market value using both tabular data and satellite imagery. This project integrates "curb appeal" and neighborhood characteristics (like green cover or road density) into traditional pricing models.

## Project Overview

This project demonstrates how combining traditional tabular features with visual features extracted from satellite imagery can improve property valuation accuracy. The system uses:
- **Tabular Data**: Traditional real estate features (bedrooms, bathrooms, square footage, location, etc.)
- **Satellite Imagery**: Visual features extracted using a CNN (ResNet18) to capture environmental context

## Repository Structure

```
satellite/
├── data/
│   ├── train(1)(train(1)).csv      # Training dataset
│   └── test2.xlsx                   # Test dataset
├── property_images/                 # Satellite images for training data
├── test_images/                     # Satellite images for test data
├── data_fetcher.py                  # Script to download satellite images
├── preprocessing.ipynb              # Data preprocessing, EDA, and feature engineering
├── main.ipynb                       # Main training notebook (multimodal model)
├── comparison.ipynb                 # Comparison: Tabular vs Hybrid models
├── champion_hybrid_house_model_v1.pkl  # Trained model
└── README.md                        # This file
```

## Setup Instructions

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for faster image processing)
- Anaconda or virtual environment

### Installation

1. **Clone the repository** (or download the project files)

2. **Create a virtual environment**:
   ```bash
   conda create -n satellite_env python=3.9
   conda activate satellite_env
   ```

3. **Install required packages**:
   ```bash
   pip install pandas numpy scikit-learn xgboost catboost lightgbm
   pip install torch torchvision
   pip install matplotlib seaborn pillow
   pip install contextily geopandas tqdm joblib openpyxl
   ```

### Alternative: Install from requirements.txt
```bash
pip install -r requirements.txt
```

## Usage

### 1. Download Satellite Images

First, download the satellite images using the data fetcher script:

```bash
python data_fetcher.py
```

This script will:
- Read the training CSV file
- Fetch satellite images for each property using lat/long coordinates
- Save images to `property_images/` directory

**Note**: The script uses the `contextily` library with Esri World Imagery as the source. Make sure you have an internet connection.

### 2. Run Preprocessing and EDA

Open and run `preprocessing.ipynb` to:
- Load and explore the dataset
- Perform exploratory data analysis (EDA)
- Apply data transformations (log transformations)
- Engineer features (spatial, interaction, neighborhood features)
- Save preprocessed data

### 3. Train the Model

Open and run `main.ipynb` to:
- Extract visual features from satellite images using CNN
- Train the hybrid stacking ensemble model
- Evaluate model performance
- Save the trained model

### 4. Compare Models

Open and run `comparison.ipynb` to:
- Train a tabular-only model (baseline)
- Train a hybrid model (tabular + visual)
- Compare performance metrics
- Generate visualizations

### 5. Make Predictions on Test Data

The test prediction code is included in `main.ipynb` (Cell 21). To make predictions:

1. Ensure test images are in `test_images/` directory
2. Run the prediction cell which will:
   - Load the trained model
   - Extract visual features from test images
   - Apply the same preprocessing pipeline
   - Generate predictions
   - Save to `final_predictions.csv`

## Model Architecture

### Hybrid Stacking Ensemble

The final model uses a **Stacking Regressor** that combines:

1. **XGBoost Regressor**
   - n_estimators: 1000
   - learning_rate: 0.03
   - max_depth: 8

2. **CatBoost Regressor**
   - iterations: 1000
   - learning_rate: 0.03
   - depth: 8

3. **LightGBM Regressor**
   - n_estimators: 1000
   - learning_rate: 0.03
   - num_leaves: 63

4. **Meta-Learner**: RidgeCV (optimizes combination weights)

### Visual Feature Extraction

- **CNN Backbone**: ResNet18 (pretrained on ImageNet)
- **Feature Extraction**: Extract 512-dim feature vector, average to get single visual score
- **Integration**: Visual score added as additional feature to tabular data

## Features

### Tabular Features
- Basic: bedrooms, bathrooms, sqft_living, floors, waterfront, view, grade
- Spatial: latitude, longitude, distance to luxury hub, rotated coordinates
- Derived: house_age, living_density, quality_density, luxury_score, neighborhood_price

### Visual Features
- Visual score: Extracted from satellite imagery using CNN

## Performance

Results from the hybrid model (based on test set):

- **R² Score**: ~0.898
- **MAE**: ~$65,762
- **RMSE**: ~$126,000

**Comparison with Tabular-Only Model**:
- The hybrid model typically shows a 1-2% improvement in R² score
- Reduced MAE and RMSE compared to tabular-only baseline

## Key Files

- `data_fetcher.py`: Downloads satellite images using coordinates
- `preprocessing.ipynb`: Complete preprocessing pipeline and EDA
- `main.ipynb`: Main training pipeline with multimodal model
- `comparison.ipynb`: Model comparison (tabular vs hybrid)

## Notes

- The model uses log-transformed target variable (price) for training
- All predictions are converted back to original scale using `expm1`
- Spatial features are calculated using only training data to prevent leakage
- Visual feature extraction can take 5-10 minutes depending on GPU availability

## Troubleshooting

1. **Image download fails**: Check internet connection and ensure `contextily` is installed
2. **GPU not detected**: The code will fall back to CPU, but processing will be slower
3. **Memory errors**: Reduce batch size or process images in smaller chunks
4. **Missing images**: The model handles missing images by assigning a neutral visual score (0.5)

## License

This project is for educational purposes as part of a Data Science problem set.

## Contact

For questions or issues, please refer to the project documentation or contact the project maintainer.

