# ============================================================
# SPAM MAIL DETECTOR
# QSkill Internship
# Domain: Artificial Intelligence & Machine Learning
# ============================================================


# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ------------------------------------------------------------
# 2. LOAD THE DATASET
# ------------------------------------------------------------

df = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="latin-1"
)

print("Dataset loaded successfully!")

print("\nFirst 5 messages:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nClass distribution:")
print(df["label"].value_counts())


# ------------------------------------------------------------
# 3. CHECK AND PREPARE THE DATA
# ------------------------------------------------------------

print("\nMissing values:")
print(df.isnull().sum())

# Remove rows containing missing values
df = df.dropna()

# Convert labels into numbers
# ham = 0
# spam = 1

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# Separate messages and labels

X = df["message"]
y = df["label"]

print("\nPrepared data:")
print(df.head())

print("\nLabel distribution after conversion:")
print(y.value_counts())


# ------------------------------------------------------------
# 4. CONVERT TEXT INTO NUMERICAL FEATURES USING TF-IDF
# ------------------------------------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_tfidf = vectorizer.fit_transform(X)

print("\nTF-IDF conversion completed!")

print(
    "Original number of messages:",
    len(X)
)

print(
    "Number of TF-IDF features:",
    len(vectorizer.get_feature_names_out())
)

print(
    "TF-IDF matrix shape:",
    X_tfidf.shape
)


# ------------------------------------------------------------
# 5. SPLIT DATA INTO TRAINING AND TESTING SETS
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nData split completed!")

print(
    "Training data:",
    X_train.shape
)

print(
    "Testing data:",
    X_test.shape
)


# ------------------------------------------------------------
# 6. CREATE THE NAIVE BAYES MODEL
# ------------------------------------------------------------

model = MultinomialNB()


# ------------------------------------------------------------
# 7. TRAIN THE MODEL
# ------------------------------------------------------------

model.fit(
    X_train,
    y_train
)

print("\nModel training completed!")


# ------------------------------------------------------------
# 8. MAKE PREDICTIONS
# ------------------------------------------------------------

y_pred = model.predict(X_test)

print("\nPredictions completed!")


# ------------------------------------------------------------
# 9. CALCULATE ACCURACY
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:")
print(accuracy)

print(
    "Accuracy percentage:",
    accuracy * 100,
    "%"
)


# ------------------------------------------------------------
# 10. CALCULATE PRECISION
# ------------------------------------------------------------

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

print("\nPrecision:")
print(precision)


# ------------------------------------------------------------
# 11. CALCULATE F1 SCORE
# ------------------------------------------------------------

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

print("\nF1 Score:")
print(f1)


# ------------------------------------------------------------
# 12. CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Ham", "Spam"],
        zero_division=0
    )
)


# ------------------------------------------------------------
# 13. CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ------------------------------------------------------------
# 14. TEST THE MODEL WITH NEW MESSAGES
# ------------------------------------------------------------

new_messages = [
    "Congratulations! You have won a free prize. Call now!",
    "Hey, are you coming to college tomorrow?"
]

new_messages_tfidf = vectorizer.transform(
    new_messages
)

new_predictions = model.predict(
    new_messages_tfidf
)

print("\nNew Message Predictions:")

for message, prediction in zip(
    new_messages,
    new_predictions
):

    if prediction == 1:
        result = "SPAM"
    else:
        result = "HAM"

    print("\nMessage:", message)
    print("Prediction:", result)
