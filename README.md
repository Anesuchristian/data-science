# Data Science Project

A machine learning project for text classification and feature engineering using TF-IDF and Bag of Words vectorization.

## Project Overview

This project focuses on text preprocessing, feature engineering, and classification using various NLP techniques. The main components include data cleaning, text vectorization, and model preparation for text-based classification tasks.

## Files Description

- `clean.py` - Data cleaning and preprocessing script
- `future.py` - Advanced feature engineering and vectorization
- `future.ipynb` - Jupyter notebook for exploratory data analysis
- Various `.joblib` files - Saved vectorizers and encoders
- Data files (CSV format) - Processed datasets

## Features

- Text data cleaning and preprocessing
- TF-IDF vectorization
- Bag of Words (BoW) vectorization
- Feature engineering and selection
- Label encoding for categorical variables

## Requirements

Install the required packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

## Usage

1. Run the data cleaning script:
   ```bash
   python clean.py
   ```

2. Execute feature engineering:
   ```bash
   python future.py
   ```

3. Use the Jupyter notebook for interactive analysis:
   ```bash
   jupyter notebook future.ipynb
   ```

## Data Processing Pipeline

1. **Data Cleaning**: Handle missing values, remove duplicates, text normalization
2. **Feature Engineering**: Create TF-IDF and BoW features
3. **Encoding**: Label encoding for categorical variables
4. **Vectorization**: Transform text data into numerical features

## Output Files

- `cleaned_train.csv` - Cleaned training data
- `tfidf_vectors.csv` - TF-IDF vectorized features
- `bow_vectors.csv` - Bag of Words vectorized features
- `combined_data_with_tfidf.csv` - Combined dataset with all features
- `feature_info.csv` - Feature information and statistics

## Model Artifacts

- `tfidf_vectorizer.joblib` - Trained TF-IDF vectorizer
- `bow_vectorizer.joblib` - Trained BoW vectorizer
- `category_encoder.joblib` - Label encoder for categories

## Contributing

Feel free to contribute to this project by submitting pull requests or reporting issues.

## License

This project is open source and available under the MIT License.
