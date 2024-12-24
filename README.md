# Child Mind Institute: Problematic Internet Use Prediction

## Overview
This project predicts problematic internet use (PIU) among children based on their physical activity, demographic, and behavioral data. The dataset combines tabular and time-series data to evaluate severity levels categorized as None, Mild, Moderate, and Severe.

This is a submission of the Kaggle contest that can be found through this [link](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use)


## Features
- **Tabular data**: Includes demographic, fitness, and behavioral metrics.
- **Time-series data**: Derived from actigraphy sensors, providing detailed physical activity metrics.
- **Innovative feature engineering**: Combines domain knowledge and statistical techniques to extract meaningful insights.
- **Ensemble learning**: Combines multiple machine learning models to improve prediction accuracy.

## Tools and Libraries
- Data Manipulation: `pandas`, `polars`, `numpy`
- Visualization: `matplotlib`, `seaborn`
- Machine Learning: `scikit-learn`, `LightGBM`, `XGBoost`, `CatBoost`, `PyTorch TabNet`
- Optimization: `SciPy`

## Dataset
The dataset includes:
- **Train and test CSV files**: Contain demographic and behavioral data.
- **Parquet files**: Contain actigraphy (time-series) data.

## Project Workflow

### 1. Data Preprocessing
- **Handling missing values**: Strategies include imputation with KNN and column-specific logic.
- **Encoding categorical variables**: Applied one-hot encoding and mapping for seamless integration with machine learning models.
- **Feature engineering**: Introduced new features such as interaction terms and derived statistical properties from time-series data.

### 2. Time-Series Analysis
- Extracted statistical and frequency-domain features using FFT (Fast Fourier Transform).
- Merged time-series features with the tabular dataset to create a unified training dataset.

### 3. Model Development
- **Autoencoder**: Used for dimensionality reduction of time-series features.
- **Gradient boosting models**: Implemented LightGBM, XGBoost, and CatBoost.
- **Ensemble learning**: Combined predictions from different models using a Voting Regressor.
- **TabNet**: Incorporated advanced neural network architectures for tabular data.

### 4. Evaluation
- **Metrics**: Quadratic Weighted Kappa (QWK) was used to evaluate model performance, particularly suited for ordinal classification.
- **Cross-validation**: Employed stratified K-Folds to ensure balanced target distributions.
- **Threshold optimization**: Fine-tuned decision thresholds to improve QWK scores.

### 5. Results
- **Optimized QWK**: Achieved improvements in validation scores by leveraging threshold optimization and ensemble techniques.
- **Test predictions**: Finalized and exported predictions for submission.

## Key Insights
- The dataset exhibits class imbalance, with most samples belonging to the "None" severity level.
- Significant features include BMI, internet usage hours, and systolic blood pressure.
- The ensemble approach outperformed individual models, demonstrating the power of combining diverse predictive techniques.

## Usage
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/problematic-internet-use.git
   cd problematic-internet-use
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the pipeline:
   ```bash
   python main.py
   ```

## Future Work
- Enhance time-series feature extraction with advanced signal processing techniques.
- Explore deep learning models for better time-series classification.
- Address class imbalance using advanced sampling techniques or loss adjustments.

## Acknowledgments
Special thanks to the Child Mind Institute for providing the dataset and organizing the competition.

