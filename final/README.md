Spam Classification using Random Forest and SVM
This project implements a spam classification system using two machine 
learning algorithms: Random Forest and Support Vector Machine (SVM). The 
system classifies emails as either "ham" (not spam) or "spam" based on 
their content.

Table of Contents
Project Description
Data
Dependencies
Usage
Results
License
Project Description
The primary goal of this project is to build a text classification model 
that can differentiate between spam and non-spam emails. The following 
steps are performed in the code:

Data Loading: Load the dataset from a CSV file containing email messages 
and their corresponding categories (ham or spam).
Data Preprocessing: Convert text messages into numerical features using 
TF-IDF vectorization.
Model Training: Train two models: a Random Forest Classifier and a Support 
Vector Machine.
Model Evaluation: Evaluate the models' performance using accuracy and 
detailed classification reports.
Visualization: Plot the accuracy comparison of both models along with the 
metrics from the classification report.
Data
The dataset used for this project is assumed to be a CSV file named 
mail_data.csv that contains two columns:

Message: The text of the email.
Category: A binary label indicating whether the email is spam (1) or ham 
(0).
The script converts the Category values from 0 and 1 to 'ham' and 'spam' 
for easier interpretation.

Dependencies
To run this project, you need to have the following Python libraries 
installed:

pandas
numpy
matplotlib
scikit-learn
You can install these libraries using pip:

bash
Copy code
pip install pandas numpy matplotlib scikit-learn
Usage
Ensure you have your mail_data.csv file in the same directory as the 
script.
Run the script using Python:
bash
Copy code
python spam_classification.py
The script will output the accuracy of each model and display a bar chart 
comparing their accuracies. Additionally, a second plot will show the 
precision, recall, and F1 scores for both models.
Results
After running the code, you will see the accuracy of both models printed 
in the console, along with classification reports that provide insights 
into the performance of each model. The accuracy comparison will also be 
visually represented in a bar chart.

License
This project is licensed under the MIT License - see the LICENSE file for 
details.


