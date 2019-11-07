from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


def model_logistic_regression(xtrain_tfidf, xvalid_tfidf, ytrain):
    lr = LogisticRegression()
    clf = OneVsRestClassifier(lr)
    clf.fit(xtrain_tfidf, ytrain)
    y_pred = clf.predict(xvalid_tfidf)
    return y_pred