from sklearn.decomposition import NMF
import numpy as np
import pre_processing
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse


def nmf_topics_model(no_topics,data, labels, test_size, random_state):
    xtrain, xvalidate, ytrain, yvalidate = pre_processing.get_data_after_split(data, labels, test_size, random_state)
    tfidf_vec = TfidfVectorizer(dtype=np.float32, sublinear_tf=True, use_idf=True, smooth_idf=True,max_df=0.8, max_features=10000)
    xtrain_tfidf = tfidf_vec.fit_transform(xtrain.values.astype('U'))
    xvalid_tfidf = tfidf_vec.transform(xvalidate.values.astype('U'))
    nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(xtrain_tfidf)
    return nmf, tfidf_vec, xtrain_tfidf, xvalid_tfidf


def stack_tfidf_nmf(tfidf_matrix,nmf_matrix):
    nmf_tfidf = np.column_stack((tfidf_matrix.toarray(),nmf_matrix))
    nmf_tfidf = sparse.csr_matrix(nmf_tfidf)
    return nmf_tfidf


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic %d:" % (topic_idx))
        print (" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))


def topic_model_main(no_topics,data, labels, test_size, random_state):
    print("### Vectorizingggg ###")
    nmf, tfidf_vectorizer, xtrain_tfidf, xvalid_tfidf = nmf_topics_model(no_topics,data, labels, test_size, random_state)
    print("### Topic modeling ###")
    training_features = nmf.transform(xtrain_tfidf)
    testing_features = nmf.transform(xvalid_tfidf)
    print(" ### concetinate tfidf and nmf ###")
    nmf_tfidf_train = stack_tfidf_nmf(xtrain_tfidf, training_features)
    nmf_tfidf_validate = stack_tfidf_nmf(xvalid_tfidf, testing_features)
    print(" ### finish topic modelling ###")
    return nmf_tfidf_train, nmf_tfidf_validate
