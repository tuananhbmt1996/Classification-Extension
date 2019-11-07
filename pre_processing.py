from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import MultiLabelBinarizer


def binarizer_labels(data):
    multilabel_binarizer = MultiLabelBinarizer()
    multilabel_binarizer.fit(data)
    # transform target variable
    y = multilabel_binarizer.transform(data)
    return y, multilabel_binarizer


def getPrecision_Recall_F1(validate, prediction, average):
    if(average == 'macro'):
        precision = precision_score(validate, prediction, average = 'macro')
        recall = recall_score(validate, prediction,  average = 'macro')
        f1 = f1_score(validate, prediction, average = 'macro')
        return precision,recall,f1
    if(average == 'micro'):
        precision = precision_score(validate, prediction, average = 'micro')
        recall = recall_score(validate, prediction,  average = 'micro')
        f1 = f1_score(validate, prediction, average = 'micro')
        return precision,recall,f1


def kfold_splits_time(numberOfK,data):
    kf = KFold(n_splits=numberOfK,random_state= 5,shuffle= True)
    kf.get_n_splits(data)
    return kf


def get_data_after_split(data,labels,test_size,random_state):
    xtrain, xvalidate, ytrain, yvalidate = train_test_split(data, labels,
                                                            test_size=test_size, random_state=random_state)
    return xtrain, xvalidate, ytrain, yvalidate


def get_tfidf_features(features, max_df, data, labels, test_size, random_state):
    tfidf_vectorizer = TfidfVectorizer(max_df=max_df, max_features=features)
    xtrain, xvalidate, ytrain, yvalidate = get_data_after_split(data,labels,test_size,random_state)
    xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
    xvalid_tfidf = tfidf_vectorizer.transform(xvalidate)
    return xtrain_tfidf, xvalid_tfidf, ytrain, yvalidate
