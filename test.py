# Libraries
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import pandas as pd
import pickle
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="model.pkl")
parser.add_argument("--vectorizer_path", type=str, default="vectorizer.pkl")
args = parser.parse_args()

# Load the saved model and vectorizer
with open(args.model_path, 'rb') as f:
    clf = pickle.load(f)
    
with open(args.vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

# New samples
new_sentences = ['This is a positive sample', 'This don\'t like the cricket', 'ill make it up to you when i get there']

# Transform the new samples using the saved vectorizer
X_test = vectorizer.transform(new_sentences)

# Use the saved model to predict the labels for the new samples
predicted_labels = clf.predict(X_test)

print(predicted_labels)  # Output: [1, 0]
