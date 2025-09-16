import pandas as pd

# Load your file (adjust path if needed)
df = pd.read_csv("train.csv")

# Preview first rows
df.head()

# Shape
print("Shape:", df.shape)

# Info
df.info()

# Missing values
print(df.isnull().sum())

# Drop 'row_id' if it's not needed
df = df.drop(columns=['row_id'])

# Handle missing values in 'Misconception'
df['Misconception'] = df['Misconception'].fillna("None")

# Remove duplicates
df = df.drop_duplicates()

# Define text cleaning function
def clean_text(text):
    if isinstance(text, str):
        text = text.strip().lower()                # lowercase + strip spaces
        text = text.replace("\\( ", "").replace(" \\)", "")  # remove LaTeX brackets
    return text

# Apply cleaning to text columns
df['QuestionText'] = df['QuestionText'].apply(clean_text)
df['MC_Answer'] = df['MC_Answer'].apply(clean_text)
df['StudentExplanation'] = df['StudentExplanation'].apply(clean_text)

# Check cleaned data
df.head()

df.to_csv("cleaned_train.csv", index=False)
print("✅ Cleaned dataset saved as cleaned_train.csv")
