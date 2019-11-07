from sklearn.neural_network import MLPClassifier


def model_mlp(xtrain_tfidf, xvalid_tfidf, ytrain):
    mlp = MLPClassifier(hidden_layer_sizes=(30, 30, 30))
    mlp.fit(xtrain_tfidf,ytrain)
    predictions_tfidf_mlp = mlp.predict(xvalid_tfidf)
    return predictions_tfidf_mlp
