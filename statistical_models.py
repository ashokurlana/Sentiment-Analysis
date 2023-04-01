# Libraries
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import argparse
import pandas as pd
from tqdm import tqdm
import pickle
import csv

## Necessary Sklearn Libraries
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import xgboost, numpy, textblob, string
from sklearn.datasets import make_classification

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="data.csv")
parser.add_argument("--output_dir", type=str, help = "Provide the output path to save the vectorizer and trained model", default=".")
parser.add_argument("--model_name", type=str, required=True, choices=['NB', 'LR', 'SVM', 'RF', 'GB' ], default="LR")
parser.add_argument("--analyzer", type=str, required=True, choices=['word', 'ngram', 'char' ], default="word")
args = parser.parse_args()

def save_vectorizer(model_name, analyzer, vectors_file):
	with open(args.output_dir+"vectorizer_"+model_name+"_"+analyzer+".pkl", 'wb') as f:
		pickle.dump(vectors_file, f)

def save_model(model_name, analyzer, classifier):
	with open(args.output_dir+ model_name+"_"+analyzer+".pkl", 'wb') as f:
		pickle.dump(classifier, f)

# load the dataset
trainDF = pd.read_csv('data.csv')
print(trainDF.head())

# Split the train data into train and validation sets
train_x, test_x, train_y, test_y = model_selection.train_test_split(trainDF['sentence'], trainDF['label'], test_size=0.2, random_state=42)
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
test_y = encoder.fit_transform(test_y)
# print(test_y)
classes = list(encoder.classes_)
print("Total number of classes in the dataset: ", len(classes))

## Feature Engineering: TF-IDF Vectorizer
if args.analyzer == "word":
	# word level tf-idf
	tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern = r'\w{1,}', max_features=5000)
	tfidf_vect.fit(trainDF['sentence'])
	xtrain_tfidf =  tfidf_vect.transform(train_x)
	xvalid_tfidf =  tfidf_vect.transform(test_x)
	save_vectorizer(args.model_name, args.analyzer, tfidf_vect)
elif args.analyzer == "ngram":
	# ngram level tf-idf 
	tfidf_vect_ngram = TfidfVectorizer(analyzer='word', ngram_range=(2,3), max_features=25000)
	tfidf_vect_ngram.fit(trainDF['sentence'])
	xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
	xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(test_x)
	save_vectorizer(args.model_name, args.analyzer, tfidf_vect)
else:
	# characters level tf-idf
	tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', ngram_range=(2,3), max_features=25000)	
	tfidf_vect_ngram_chars.fit(trainDF['sentence'])
	xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
	xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(test_x) 
	save_vectorizer(args.model_name, args.analyzer, tfidf_vect)

## Model Training
def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    save_model(args.model_name, args.analyzer, classifier)

    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)

    if is_neural_net:
        predictions = predictions.argmax(axis=-1)

    
    acc = metrics.accuracy_score(predictions, test_y)
    f1 = metrics.f1_score(predictions, test_y, average='weighted')
    #print(classification_report(predictions, test_y, target_names = list(encoder.classes_)))
    return acc, f1

## Logistic Regression
if args.model_name == "LR":
	if args.analyzer=="word":
		# Linear Classifier on Word Level TF IDF Vectors
		accuracy, f1_score = train_model(linear_model.LogisticRegression(multi_class='multinomial', solver='sag', max_iter=25000), xtrain_tfidf, train_y, xvalid_tfidf)
		print("LR, WordLevel TF-IDF: ", accuracy, f1_score)
	elif args.analyzer=="ngram":
		# Linear Classifier on Ngram Level TF IDF Vectors
		accuracy, f1_score = train_model(linear_model.LogisticRegression(multi_class='multinomial', solver='sag', max_iter=25000), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
		print("LR, N-Gram Vectors: ", accuracy, f1_score)
	else:
		# Linear Classifier on Character Level TF IDF Vectors
		accuracy, f1_score = train_model(linear_model.LogisticRegression(multi_class='multinomial', solver='sag', max_iter=25000), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
		print("LR, CharLevel Vectors: ", accuracy, f1_score)
elif args.model_name == "NB":
	if args.analyzer == "word":
		# Naive Bayes on Word Level TF IDF Vectors
		accuracy, f1_score = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
		print("NB, WordLevel TF-IDF: ", accuracy, f1_score)
	elif args.analyzer == "ngram":
		# Naive Bayes on Ngram Level TF IDF Vectors
		accuracy, f1_score = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
		print("NB, N-Gram Vectors: ", accuracy, f1_score)
	else:
		# Naive Bayes on Character Level TF IDF Vectors
		accuracy, f1_score = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
		print("NB, CharLevel Vectors: ", accuracy, f1_score)
elif args.model_name == "SVM":
	if args.analyzer == "word":
		# SVM on TF IDF Vectors
		accuracy = train_model(svm.SVC(), xtrain_tfidf, train_y, xvalid_tfidf)
		print("SVM, TF-IDF Vectors: ", accuracy)
	elif args.analyzer == "ngram":
		# SVM on Ngram Level TF IDF Vectors
		accuracy = train_model(svm.SVC(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
		print("SVM, Ngram Level Vectors: ", accuracy)
	else:
		# # SVM on Character Level TF IDF Vectors
		accuracy = train_model(svm.SVC(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
		print("SVM, Character Level Vectors: ", accuracy)
elif args.model_name == "RF":
	if args.analyzer == "word":
		# RF on Word Level TF IDF Vectors
		accuracy = train_model(ensemble.RandomForestClassifier(n_estimators=10, min_samples_split=2, n_jobs=-1), xtrain_tfidf, train_y, xvalid_tfidf)
		print("RF, WordLevel TF-IDF: ", accuracy)
	elif args.analyzer == "ngram":
		# RF on Word Level TF IDF Vectors
		accuracy = train_model(ensemble.RandomForestClassifier(n_estimators=10, min_samples_split=2, n_jobs=-1), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
		print("RF, NgramLevel TF-IDF: ", accuracy)
	else:
		# RF on Word Level TF IDF Vectors
		accuracy = train_model(ensemble.RandomForestClassifier(n_estimators=10, min_samples_split=2, n_jobs=-1), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
		print("RF, Character Level TF-IDF: ", accuracy)
elif args.model_name =="GB":
	if args.analyzer == "word":
		# Extereme Gradient Boosting on Word Level TF IDF Vectors
		accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf.tocsc(), train_y, xvalid_tfidf.tocsc())
		print("Xgb, WordLevel TF-IDF: ", accuracy)
	else:
		# Extereme Gradient Boosting on Character Level TF IDF Vectors
		accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf_ngram_chars.tocsc(), train_y, xvalid_tfidf_ngram_chars.tocsc())
		print("Xgb, CharLevel Vectors: ", accuracy)
