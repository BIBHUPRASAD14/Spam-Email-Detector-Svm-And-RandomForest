# Step 1: Import Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings("ignore")

# Step 2: Load Data
# Replace this with actual data loading code
# For demonstration, we'll create a sample DataFrame
# data = {
#     'text': [
#         'Congratulations! You have won a $1000 Walmart gift card.',
#         'Dear friend, please let me know if you received my previous 
message.',
#         'Win big money now!',
#         'Hello, I hope you are doing well. Let’s catch up soon.',
#         'Get cheap medications now!',
#         'Your account has been compromised. Please update your 
password.',
#     ],
#     'label': [1, 0, 1, 0, 1, 0]  # 1 for spam, 0 for not spam
# }
data= "mail_data.csv"

df = pd.read_csv(data)

# Step 3: Preprocess Data
X = df['Message']
y = df['Category']

# Convert text data to TF-IDF features
vectorizer = TfidfVectorizer()
X_features = vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_features, y, 
test_size=0.2, random_state=42)

# Step 4: Train Models

# Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Support Vector Machine (SVM)
svm_clf = SVC(kernel='linear', random_state=42)
svm_clf.fit(X_train, y_train)

# Step 5: Evaluate Models
# Random Forest Predictions
rf_predictions = rf_clf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)

# SVM Predictions
svm_predictions = svm_clf.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)

print(f"Random Forest Accuracy: {rf_accuracy:.2f}")
print(f"SVM Accuracy: {svm_accuracy:.2f}")

# Detailed Classification Report
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_predictions))

print("\nSVM Classification Report:")
print(classification_report(y_test, svm_predictions))

# Step 6: Plot Accuracy Comparison
models = ['Random Forest', 'SVM']
accuracies = [rf_accuracy, svm_accuracy]

plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color=['blue', 'green'])
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.ylim(0, 1)  # Set y-axis limits to 0-1 for better comparison
plt.grid(axis='y')

# Show the plot
plt.show()

