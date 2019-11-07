from keras.models import Sequential
from keras import layers
from keras.layers import  Flatten, Dropout


def model_deep_mlp(nmf_tfidf_train,nmf_tfidf_validate,ytrain,yvalidate,first_layer_dense, second_layer_dense,epochs,batch_size):
    input_dim = nmf_tfidf_train.shape[1]  # Number of features
    model = Sequential()
    # Activation relu = Exponential linear unit.
    model.add(layers.Dense(first_layer_dense, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(second_layer_dense, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    # epochs = full training cycle on training set, after each epochs, restart from the 1st point data.
    # batch_size = split the data base of this number, 1000 data with batch_size 100,
    # data will be trained 10 times each epochs
    history = model.fit(nmf_tfidf_train, ytrain,
                        epochs=epochs,
                        verbose=False,
                        validation_data=(nmf_tfidf_validate, yvalidate),
                        batch_size=batch_size)
    loss, accuracy = model.evaluate(nmf_tfidf_validate, yvalidate, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))
    return model.predict_classes(nmf_tfidf_validate,batch_size=batch_size,verbose=0)


