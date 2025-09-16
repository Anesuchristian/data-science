import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
import re
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

print("Starting advanced feature engineering...")

# Load the dataset
df = pd.read_csv('cleaned_train.csv')
print(f"Dataset loaded: {df.shape[0]} rows, {df.columns.tolist()}")

# Handle missing values in the StudentExplanation column
text_data = df['StudentExplanation'].fillna('').astype(str).tolist()

# Basic text statistics features
print("Creating text statistics features...")
df['text_length'] = df['StudentExplanation'].fillna('').astype(str).str.len()
df['word_count'] = df['StudentExplanation'].fillna('').astype(str).str.split().str.len()
df['sentence_count'] = df['StudentExplanation'].fillna('').astype(str).str.count('\.')
df['exclamation_count'] = df['StudentExplanation'].fillna('').astype(str).str.count('!')
df['question_count'] = df['StudentExplanation'].fillna('').astype(str).str.count('\?')
df['uppercase_count'] = df['StudentExplanation'].fillna('').astype(str).str.count('[A-Z]')
df['digit_count'] = df['StudentExplanation'].fillna('').astype(str).str.count('\d')

# Mathematical expression features
print("Creating mathematical expression features...")
df['has_fraction'] = df['StudentExplanation'].fillna('').astype(str).str.contains(r'\d+/\d+').astype(int)
df['has_decimal'] = df['StudentExplanation'].fillna('').astype(str).str.contains(r'\d+\.\d+').astype(int)
df['has_percentage'] = df['StudentExplanation'].fillna('').astype(str).str.contains(r'\d+%').astype(int)
df['math_operations'] = df['StudentExplanation'].fillna('').astype(str).str.count(r'[\+\-\*\/\=]')

# Encode categorical features
print("Encoding categorical features...")
le_category = LabelEncoder()
df['Category_encoded'] = le_category.fit_transform(df['Category'])

# Initialize the TF-IDF Vectorizer with additional parameters
print("Creating TF-IDF features...")
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    max_features=500,
    ngram_range=(1, 2),
    min_df=2,  # Ignore terms that appear in less than 2 documents
    max_df=0.95  # Ignore terms that appear in more than 95% of documents
)

# Fit and transform the text data
tfidf_matrix = vectorizer.fit_transform(text_data)

# Convert to DataFrame
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=[f'tfidf_{col}' for col in vectorizer.get_feature_names_out()]
)

# Create bag of words features for comparison
print("Creating Bag of Words features...")
count_vectorizer = CountVectorizer(
    lowercase=True,
    stop_words='english',
    max_features=200,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

bow_matrix = count_vectorizer.fit_transform(text_data)
bow_df = pd.DataFrame(
    bow_matrix.toarray(),
    columns=[f'bow_{col}' for col in count_vectorizer.get_feature_names_out()]
)

# Combine all features
print("Combining all features...")
feature_columns = ['text_length', 'word_count', 'sentence_count', 'exclamation_count', 
                  'question_count', 'uppercase_count', 'digit_count', 'has_fraction',
                  'has_decimal', 'has_percentage', 'math_operations', 'Category_encoded']

# Create the combined dataset
combined_df = pd.concat([
    df.reset_index(drop=True),
    tfidf_df.reset_index(drop=True),
    bow_df.reset_index(drop=True)
], axis=1)

# Save the results
print("Saving results...")
tfidf_df.to_csv('tfidf_vectors.csv', index=False)
bow_df.to_csv('bow_vectors.csv', index=False)
combined_df.to_csv('combined_data_with_tfidf.csv', index=False)

# Save feature importance information
feature_info = pd.DataFrame({
    'feature_name': feature_columns,
    'feature_type': ['text_stat'] * 7 + ['math_feature'] * 4 + ['categorical']
})
feature_info.to_csv('feature_info.csv', index=False)

# Save the vectorizers
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
joblib.dump(count_vectorizer, 'bow_vectorizer.joblib')
joblib.dump(le_category, 'category_encoder.joblib')

# Print summary statistics
print("\n=== Feature Engineering Summary ===")
print(f"Original features: {len(df.columns)}")
print(f"TF-IDF features: {tfidf_df.shape[1]}")
print(f"Bag of Words features: {bow_df.shape[1]}")
print(f"Total features in combined dataset: {combined_df.shape[1]}")
print(f"Text statistics features: {len([col for col in feature_columns if 'text_' in col or 'word_' in col or 'sentence_' in col or 'exclamation_' in col or 'question_' in col or 'uppercase_' in col or 'digit_' in col])}")
print(f"Mathematical features: {len([col for col in feature_columns if 'has_' in col or 'math_' in col])}")

print("\nFiles created:")
print("- tfidf_vectors.csv (TF-IDF features)")
print("- bow_vectors.csv (Bag of Words features)")
print("- combined_data_with_tfidf.csv (All features combined)")
print("- feature_info.csv (Feature metadata)")
print("- tfidf_vectorizer.joblib (TF-IDF model)")
print("- bow_vectorizer.joblib (Bag of Words model)")
print("- category_encoder.joblib (Category encoder)")

print("\nFeature engineering completed successfully!")