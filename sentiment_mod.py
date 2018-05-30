import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify import ClassifierI
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from statistics import mode
import pickle

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
            self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

short_pos = open("short_reviews/positive.txt", "r").read()
short_neg = open("short_reviews/negative.txt", "r").read()

documents_f = open("pickled_algs/documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()

word_features_f = open("pickled_algs/word_features.pickle", "rb")
word_features = pickle.load(word_features_f)
word_features_f.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

open_file = open("pickled_algs/featuresets.pickle", "rb")
featuresets = pickle.load(open_file)
open_file.close()

random.shuffle(featuresets)

training_set = featuresets[:10000]
testing_set = featuresets[10000:]

open_file = open("pickled_algs/naivebayes.pickle", "rb")
classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickled_algs/MNB_classifier.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickled_algs/BNB_classifier.pickle", "rb")
BNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickled_algs/LogisticRegression_Classifier.pickle", "rb")
LogisticRegression_Classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickled_algs/SGD_Classifier.pickle", "rb")
SGD_Classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickled_algs/SVC_Classifier.pickle", "rb")
SVC_Classifier = pickle.load(open_file)
open_file.close()


open_file = open("pickled_algs/NuSVC_Classifier.pickle", "rb")
NuSVC_Classifier = pickle.load(open_file)
open_file.close()

voted_classifier = VoteClassifier(classifier,
                                    MNB_classifier,
                                    BNB_classifier,
                                    LogisticRegression_Classifier,
                                    SGD_Classifier,
                                    SVC_Classifier,
                                    NuSVC_Classifier)

def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats), voted_classifier.confidence(feats)
