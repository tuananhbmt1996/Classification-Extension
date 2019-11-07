import pandas as pd
import topic_model
import clean_data as cd
import pre_processing
import logistic_regression
import deep_mlp
import deep_cnn
import nltk
import re
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from keras.models import Sequential
from keras import layers
from keras.layers import  Flatten, Dropout
import warnings
import mlp
import numpy as np
warnings.filterwarnings('ignore')

data_link = "data/movie.metadata.tsv"
file_link = "data/plot_summaries.txt"


def read_data(data_link):
    data = pd.read_csv(data_link, sep='\t', header=None)
    data.columns = ["movie_id", 1, "movie_name", 3, 4, 5, 6, 7, "genre"]
    movies = cd.read_plot_from_corpus(file_link)
    movies = cd.merge_data(movies, data)
    movies["genre_new"] = cd.convert_genres(movies)
    movies = cd.remove_empty_rows(movies)
    clean_plot = movies['plot'].apply(lambda x: cd.clean_text(x))
    movies['clean_plot'] = clean_plot
    movies['clean_plot'] = movies['clean_plot'].apply(lambda x: cd.remove_stopwords(x))
    return movies


def main(select_mlp=False, select_lr=False,select_deep_mlp=False,select_deep_cnn=False):
    print("#### reading data ###")
    movies = read_data(data_link)
    print("### binarizer labels  ###")
    y, multilabel_binarizer = pre_processing.binarizer_labels(movies['genre_new'])
    print("### tfidf + nmf vectorizer")
    no_topics = 50
    print("topic numbers: " + str(no_topics))
    xtrain_tfidf, xvalid_tfidf, ytrain, yvalidate = pre_processing.get_tfidf_features(10000, 0.8, movies['clean_plot'],
                                                                                      y, 0.2, 9)
    nmf_tfidf_train, nmf_tfidf_validate = topic_model.topic_model_main(no_topics, movies['clean_plot'], y, 0.2, 9)
    if select_mlp:
        print("### training mlp models ###")
        prediction_mlp = mlp.model_mlp(nmf_tfidf_train, nmf_tfidf_validate, ytrain)
        print("### accuracy mlp ###")
        precision_mlp, recall_mlp, f1_score_mlp = \
            pre_processing.getPrecision_Recall_F1(yvalidate, prediction_mlp, "micro")
        print(f1_score_mlp)

    if select_lr:
        print("### training logistic regression models ###")
        prediction_lr = logistic_regression.model_logistic_regression(nmf_tfidf_train, nmf_tfidf_validate, ytrain)
        print("### accuracy logistic regression ###")
        precision_lr, recall_lr, f1_score_lr = \
            pre_processing.getPrecision_Recall_F1(yvalidate, prediction_lr, "micro")
        print(f1_score_lr)

    if select_deep_mlp:
        print("### training deep mlp models ###")
        first_layer = 1000
        second_layer = 363
        epochs = 1
        batch_size = 64
        prediction_deep_mlp = deep_mlp.model_deep_mlp\
            (nmf_tfidf_train,nmf_tfidf_validate,ytrain,yvalidate,first_layer,second_layer,epochs,batch_size)
        print("### accuracy of deep mlp models  ###")

    if select_deep_cnn:
        print("### training deep cnn models ###")
        second_layer = 363
        epochs = 1
        batch_size = 64
        accuracy = deep_cnn.model_deep_cnn\
            (nmf_tfidf_train,nmf_tfidf_validate,ytrain,yvalidate,second_layer,epochs,batch_size)
        print("### accuracy of deep cnn models  ###")
        print("Testing Accuracy:  {:.4f}".format(accuracy))

main(select_deep_mlp=True)
