# Spam Classification using Random Forest and SVM

This project implements a spam classification system using two machine learning algorithms: Random Forest and Support Vector Machine (SVM). The system classifies emails as either "ham" (not spam) or "spam" based on their content.

## Table of Contents

- [Project Description](#project-description)
- [Data](#data)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Project Description

The primary goal of this project is to build a text classification model that can differentiate between spam and non-spam emails. The following steps are performed in the code:

1. **Data Loading**: Load the dataset from a CSV file containing email messages and their corresponding categories (ham or spam).
2. **Data Preprocessing**: Convert text messages into numerical features using TF-IDF vectorization.
3. **Model Training**: Train two models: a Random Forest Classifier and a Support Vector Machine.
4. **Model Evaluation**: Evaluate the models' performance using accuracy and detailed classification reports.
5. **Visualization**: Plot the accuracy comparison of both models along with the metrics from the classification report.

## Data

The dataset used for this project is assumed to be a CSV file named `mail_data.csv` that contains two columns:

- `Message`: The text of the email.
- `Category`: A binary label indicating whether the email is spam (1) or ham (0).

The script converts the `Category` values from 0 and 1 to 'ham' and 'spam' for easier interpretation.

## Dependencies

To run this project, you need to have the following Python libraries installed:

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`

You can install these libraries using pip:

```bash
pip install pandas numpy matplotlib scikit-learn
```

## Usage

1. Ensure you have your `mail_data.csv` file in the same directory as the script.
2. Run the script using Python:

```bash
   python spam_classification.py
```

3. The script will output the accuracy of each model and display a bar chart comparing their accuracies. Additionally, a second plot will show the precision, recall, and F1 scores for both models.

## Results

After running the code, you will see:

- The accuracy of both models printed in the console.
- A bar chart comparing the accuracies of the **Random Forest** and **SVM** models.
- Classification reports that provide insights into the performance of each model, including precision, recall, and F1-scores.
- A second plot showing the precision, recall, and F1-score metrics for each class (ham and spam) in both models.
- A message will also appear indicating which model has the greater accuracy.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

