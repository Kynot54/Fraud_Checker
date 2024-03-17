import spacy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import random

# Print Statements were for Debugging and Keeping Track of Progress

# Load Spacy
nlp = spacy.load('en_core_web_sm')

# Label Mapping - Mapping the Data for the Model so Positive, Neutral, or Negative
# is consistent across datasets
label_mapping = {
    0:'negative',
    4: 'positive'
}

# Required since dataset for "sentiment140.csv" so large
dtype_enum = {
    'label': int,
    'sentiment': str
}

print("Loading Data...")

dataset_1 = pd.read_csv('sentiment140.csv', encoding='latin-1', header=None, dtype=dtype_enum, low_memory=False)


# Trimming the dataset to 39,998 rows
rows_to_drop = 998576
random.seed(42)
rows_top_drop_indicies = random.sample(range(len(dataset_1)), rows_to_drop)
trimmed_dataset = dataset_1.drop(rows_top_drop_indicies)
trimmed_dataset.reset_index(drop=True, inplace=True)
trimmed_dataset.to_csv('trimmed_dataset_sentiment140.csv', index=False)

# Read the trimmed dataset
dataset_1 = pd.read_csv('trimmed_dataset_sentiment140.csv', encoding='latin-1', header=None, dtype=dtype_enum)
dataset_1.columns = ["sentiment", "time", "date", "query", "username", "text"]

# Dropping columns that are not needed
dataset_1 = dataset_1.drop(columns=["time", "date", "query", "username"])


# Start of IMDB Dataset (Dataset 2)
dataset_2 = pd.read_csv('IMDB_Dataset.csv', encoding='latin-1', header=None, dtype=dtype_enum)

# Trimming the dataset to 10,000 rows
rows_to_drop = 19582
random.seed(42)
rows_top_drop_indicies = random.sample(range(len(dataset_1)), rows_to_drop)
trimmed_dataset = dataset_2.drop(rows_top_drop_indicies)
trimmed_dataset.reset_index(drop=True, inplace=True)
trimmed_dataset.to_csv('trimmed_dataset_IMDB.csv', index=False)

# Read the trimmed dataset
dataset_2 = pd.read_csv("trimmed_dataset_IMDB.csv", encoding='latin-1', header=None, dtype=dtype_enum)
dataset_2.columns = ["text", "sentiment"]


print("Data Loaded")

# Mapping the Data for the Model so Positive or Negative
dataset_1["sentiment"] = dataset_1["sentiment"].map(label_mapping)
dataset_2["sentiment"] = dataset_2["sentiment"].map(label_mapping)

print("Data Mapped")

# Dropping NaN values
dataset_1 = dataset_1.dropna(subset=["text"])
dataset_2 = dataset_2.dropna(subset=["text"])
dataset_1 = dataset_1.dropna(subset=["sentiment"])
dataset_2 = dataset_2.dropna(subset=["sentiment"])

print("Data Cleaned")

# Combining the two datasets
combined_dataset = pd.concat([dataset_1, dataset_2], ignore_index=True)  

sentiment_counts = combined_dataset['sentiment'].value_counts()

print(f"Sentiment Counts: \n{sentiment_counts}")

# Used to Reduce the numbers of "negative" datums and reduced the number of "positive" datums to get an even distribution for batch Œﬁprocessing
min_samples = min(sentiment_counts) - 65 

print(min_samples)

# Creating a new dataframe with the needed columns
balanced_dataset = pd.DataFrame(columns=combined_dataset.columns)

print(balanced_dataset)

# Used to Reduce the numbers of "negative" datums and add values to the newly created dataframe
for sentiment, count in sentiment_counts.items():
    samples = combined_dataset[combined_dataset['sentiment'] == sentiment].sample(min_samples)
    balanced_dataset = pd.concat([balanced_dataset, samples], ignore_index=True)    
    
print(balanced_dataset["sentiment"].value_counts())

print("Data Combined")

# Set the X and Y values for Scikit Learn
X = balanced_dataset['text']
Y = balanced_dataset['sentiment']

print("Data Split")

# Batch Processing the Data into Tokens so it does get 'hung up' when running
batch_size = 500
X_processed = []

for i in range(0, len(X), batch_size):
    batch = X[i:i + batch_size]
    print(f"Processing Batch {i} - {i + batch_size}")
    processed_batch = [" ".join([token.text for token in nlp(text)]) for text in batch]
    print(f"Batch {i} - {i + batch_size} Processed")
    X_processed.extend(processed_batch)

X = X_processed
print("Data Tokenized")

# Vectorizing the Data and Creating the Model for User Input on Server
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(X)

joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

print("Data Vectorized")

# Trainning Set - 70%
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)

# Validation Set - 20%
X_val,  X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.67, random_state=42)

print("Data Split")

# Creating the Scalable Vector Machine Model
scalable_vector_machine =  SVC()
scalable_vector_machine.fit(X_train, Y_train)

print("Model Trained")

# Remaining Test Set - 10%
Y_pred = scalable_vector_machine.predict(X_test)
model_accuracy = accuracy_score(Y_test, Y_pred)

print("Model Tested")

# Printing the Accuracy and Confusion Matrix

print(f"Confusion Matrix: \n{confusion_matrix(Y_test, Y_pred)}")

print(f"Accuracy: {model_accuracy}")

# Saving the Model

joblib.dump(scalable_vector_machine, 'sentiment_model.pkl')