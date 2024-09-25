import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, 
confusion_matrix
import snowballstemmer  # Lightweight stemming package
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize SnowballStemmer from snowballstemmer package
stemmer = snowballstemmer.stemmer('english')

# Manual stopwords (sklearn's built-in stopwords)
from sklearn.feature_extraction import text
stop_words = text.ENGLISH_STOP_WORDS

# Data Preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(f'[{string.punctuation}]', '', text)
    # Tokenize and remove stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    # Apply stemming using snowballstemmer
    tokens = [stemmer.stemWord(word) for word in tokens]
    return ' '.join(tokens)

# Read the data from CSV file
df = pd.read_csv('mail_data.csv')  # Replace with your file path
df['Message'] = df['Message'].apply(preprocess_text)

# Map 0 to 'ham' and 1 to 'spam'
df['Category'] = df['Category'].map({0: 'ham', 1: 'spam'})

# Features (message content) and target (ham or spam)
X = df['Message']
y = df['Category']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
random_state=42)

# Convert the text data to numerical format using TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = tfidf_vectorizer.transform(X_test).toarray()

# Train SVM classifier
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_tfidf, y_train)
y_pred_svm = svm_model.predict(X_test_tfidf)

# Train Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_tfidf, y_train)
y_pred_rf = rf_model.predict(X_test_tfidf)

# Evaluation function
def evaluate_model(y_test, y_pred, model_name):
    print(f"Evaluation for {model_name}:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Classification Report:\n{classification_report(y_test, 
y_pred)}")
    cm = confusion_matrix(y_test, y_pred, labels=['ham', 'spam'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['ham', 
'spam'], yticklabels=['ham', 'spam'])
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Evaluate SVM model
evaluate_model(y_test, y_pred_svm, 'SVM')

# Evaluate Random Forest model
evaluate_model(y_test, y_pred_rf, 'Random Forest')

