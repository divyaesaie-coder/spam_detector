# spam_detector
SMS Spam Detector using TF-IDF and Multinomial Naive Bayes to classify messages as spam or ham with 97% accuracy.
# Spam Mail Detector

## QSkill Internship - Artificial Intelligence & Machine Learning

### Objective

The objective of this project is to classify SMS messages as either:

- Ham (Normal Message)
- Spam

The project uses Natural Language Processing and Machine Learning.

## Dataset

The SMS Spam Collection dataset from the UCI Machine Learning Repository is used.

The dataset contains:

- 5,572 SMS messages
- 4,825 Ham messages
- 747 Spam messages

## Technologies Used

- Python
- Pandas
- Scikit-learn

## Text Processing

The SMS text is converted into numerical features using TF-IDF.

Common English stop words are removed and all text is converted to lowercase.

## Machine Learning Model

Multinomial Naive Bayes is used for classification.

## Train-Test Split

The dataset is split into:

- 80% Training Data
- 20% Testing Data

Training samples: 4,457

Testing samples: 1,115

## Model Results

- Accuracy: 97.04%
- Spam Precision: 100%
- Spam Recall: 78%
- Spam F1 Score: 87.55%

## Confusion Matrix

```text
[[966   0]
 [ 33 116]]
