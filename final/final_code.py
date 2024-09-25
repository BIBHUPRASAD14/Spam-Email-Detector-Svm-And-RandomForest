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
data = "mail_data.csv"
df = pd.read_csv(data)

# Change Category from 0 and 1 to 'ham' and 'spam'
df['Category'] = df['Category'].replace({0: 'ham', 1: 'spam'})

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
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

svm_clf = SVC(kernel='linear', random_state=42)
svm_clf.fit(X_train, y_train)

# Step 5: Evaluate Models
rf_predictions = rf_clf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)

svm_predictions = svm_clf.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)

print(f"Random Forest Accuracy: {rf_accuracy:.2f}")
print(f"SVM Accuracy: {svm_accuracy:.2f}")

print("\nRandom Forest Classification Report:")
rf_report = classification_report(y_test, rf_predictions, 
output_dict=True)
print(classification_report(y_test, rf_predictions))

print("\nSVM Classification Report:")
svm_report = classification_report(y_test, svm_predictions, 
output_dict=True)
print(classification_report(y_test, svm_predictions))

# Step 6: Plot Accuracy Comparison
models = ['Random Forest', 'SVM']
accuracies = [rf_accuracy, svm_accuracy]

plt.figure(figsize=(12, 6))

# Accuracy Comparison Plot
plt.subplot(1, 2, 1)
bars = plt.bar(models, accuracies, color=['blue', 'green'])
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.ylim(0, 1)  # Set y-axis limits to 0-1 for better comparison
plt.grid(axis='y')

# text indicating which model has greater accuracy
if rf_accuracy > svm_accuracy:
    greater_model = "Random Forest"
else:
    greater_model = "SVM"

plt.text(0.5, -0.1, f"{greater_model} has the greater accuracy", 
ha='center', fontsize=12, transform=plt.gca().transAxes)

# Classification Report Plot
labels = ['ham', 'spam']
rf_precision = [rf_report[label]['precision'] for label in labels]
rf_recall = [rf_report[label]['recall'] for label in labels]
rf_f1 = [rf_report[label]['f1-score'] for label in labels]

svm_precision = [svm_report[label]['precision'] for label in labels]
svm_recall = [svm_report[label]['recall'] for label in labels]
svm_f1 = [svm_report[label]['f1-score'] for label in labels]

x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

# Random Forest
plt.subplot(1, 2, 2)
bar1 = plt.bar(x - width, rf_precision, width, label='Precision (RF)', 
color='blue')
bar2 = plt.bar(x, rf_recall, width, label='Recall (RF)', 
color='lightblue')
bar3 = plt.bar(x + width, rf_f1, width, label='F1-Score (RF)', 
color='cyan')

# SVM
bar4 = plt.bar(x - width, svm_precision, width, label='Precision (SVM)', 
color='green')
bar5 = plt.bar(x, svm_recall, width, label='Recall (SVM)', 
color='lightgreen', bottom=svm_precision)
bar6 = plt.bar(x + width, svm_f1, width, label='F1-Score (SVM)', 
color='lime', bottom=np.array(svm_precision) + np.array(svm_recall))

plt.xlabel('Class')
plt.ylabel('Scores')
plt.title('Classification Report Metrics Comparison')
plt.xticks(x, labels)
plt.legend()
plt.ylim(0, 1)

plt.tight_layout()
plt.show()

