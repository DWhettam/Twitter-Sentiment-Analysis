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

all_words = []
documents = []


#  j is adjective, r is adverb, and v is verb
allowed_word_types = ["J"]

for p in short_pos.split('\n'):
    documents.append( (p, "pos") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())


for p in short_neg.split('\n'):
    documents.append( (p, "neg") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

save_documents = open("pickled_algs/documents.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()

all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:5000]

save_features = open("pickled_algs/word_features.pickle","wb")
pickle.dump(word_features, save_features)
save_features.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]
save_featuresets = open("pickled_algs/featuresets.pickle","wb")
pickle.dump(featuresets, save_featuresets)
save_featuresets.close()

random.shuffle(featuresets)

training_set = featuresets[:10000]
testing_set = featuresets[10000:]

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes accuracy:", nltk.classify.accuracy(classifier, testing_set) * 100, "%")

save_classifier = open("pickled_algs/naivebayes.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()


MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy:", nltk.classify.accuracy(MNB_classifier, testing_set) * 100, "%")

save_classifier = open("pickled_algs/MNB_classifier.pickle","wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()


BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
print("BNB_classifier accuracy:", nltk.classify.accuracy(BNB_classifier, testing_set) * 100, "%")

save_classifier = open("pickled_algs/BNB_classifier.pickle","wb")
pickle.dump(BNB_classifier, save_classifier)
save_classifier.close()


LogisticRegression_Classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_Classifier.train(training_set)
print("LogisticRegression_Classifier accuracy:", nltk.classify.accuracy(LogisticRegression_Classifier, testing_set) * 100, "%")

save_classifier = open("pickled_algs/LogisticRegression_Classifier.pickle","wb")
pickle.dump(LogisticRegression_Classifier, save_classifier)
save_classifier.close()


SGD_Classifier = SklearnClassifier(SGDClassifier())
SGD_Classifier.train(training_set)
print("SGD_Classifier accuracy:", nltk.classify.accuracy(SGD_Classifier, testing_set) * 100, "%")

save_classifier = open("pickled_algs/SGD_Classifier.pickle","wb")
pickle.dump(SGD_Classifier, save_classifier)
save_classifier.close()


SVC_Classifier = SklearnClassifier(SVC())
SVC_Classifier.train(training_set)
print("SVC_Classifier accuracy:", nltk.classify.accuracy(SVC_Classifier, testing_set) * 100, "%")

save_classifier = open("pickled_algs/SVC_Classifier.pickle","wb")
pickle.dump(SVC_Classifier, save_classifier)
save_classifier.close()


NuSVC_Classifier = SklearnClassifier(NuSVC())
NuSVC_Classifier.train(training_set)
print("NuSVC_Classifier accuracy:", nltk.classify.accuracy(NuSVC_Classifier, testing_set) * 100, "%")

save_classifier = open("pickled_algs/NuSVC_Classifier.pickle","wb")
pickle.dump(NuSVC_Classifier, save_classifier)
save_classifier.close()



voted_classifier = VoteClassifier(classifier,
                                    MNB_classifier,
                                    BNB_classifier,
                                    LogisticRegression_Classifier,
                                    SGD_Classifier,
                                    SVC_Classifier,
                                    NuSVC_Classifier)

print("Voted classifer accuracy:", (nltk.classify.accuracy(voted_classifier, testing_set)) * 100, "%")

def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats), voted_classifier.confidence(feats)
