import pandas as pd
import numpy as np
import re
import sys
import os
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import NaiveBayes as nb  # Ensure NaiveBayes.py is in the same folder

# Set file directory to Desktop
BASE_DIR = "/Users/mac_user/Desktop/"

# Check if all required files exist
required_files = ["train2.tsv", "unmarked-include.csv"]
missing_files = [f for f in required_files if not os.path.exists(os.path.join(BASE_DIR, f))]

if missing_files:
    raise FileNotFoundError(f"Missing files: {', '.join(missing_files)} in Desktop. Please check.")

# Load the training dataset
train = pd.read_csv(os.path.join(BASE_DIR, "train2.tsv"), header=0, delimiter="\t", quoting=3)

# Function to clean text data
def review_to_words(raw_review):
    r = re.sub("[^a-zA-Z]", " ", raw_review)  # Remove special characters
    r = r.lower().split()
    words = [w for w in r if w not in stopwords.words("english")]
    return " ".join(words)

# Clean all reviews
def clean_all(reviews):
    num_reviews = len(reviews["Phrase"])
    clean_reviews = []

    for i in range(num_reviews):
        if (i + 1) % 100 == 0:
            print(f"Cleaning {i+1}/{num_reviews} reviews")
        clean_reviews.append(review_to_words(reviews["Phrase"][i]))

    return clean_reviews

# Process training set
clean_reviews = clean_all(train)

# Create vocabulary
vec = CountVectorizer(max_features=5000)
features = vec.fit_transform(clean_reviews).toarray()
vocab = vec.get_feature_names_out()

# Train Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(features, train["Sentiment Value"])

# Load test set
test = pd.read_csv(os.path.join(BASE_DIR, "unmarked-include.csv"), header=0, delimiter="\t", quoting=3)

clean_test = clean_all(test)
test_features = vec.transform(clean_test).toarray()

# Predict sentiments
result = classifier.predict(test_features)

# Print results
def result_stats(result, test):
    result_diff = result - test["Sentiment Value"]

    for i in range(len(test)):
        print(f"{test['Phrase'][i]}")
        print(f"Predicted: {result[i]}, Actual: {test['Sentiment Value'][i]}, Difference: {result_diff[i]}")

    accuracy = np.mean(result == test["Sentiment Value"]) * 100
    print(f"Accuracy: {accuracy:.2f}%")

result_stats(result, test)

# Train custom Naive Bayes
classes = [0, 1, 2, 3, 4]
classifier2 = nb.NaiveBayes(clean_reviews, train["Sentiment Value"], classes, vocab, clean_test, test["Sentiment Value"])
