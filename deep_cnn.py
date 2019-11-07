from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D


max_features = 10000
maxlen = 10050
embedding_dims = 50
filters = 250
kernel_size = 3


def model_deep_cnn(nmf_tfidf_train,nmf_tfidf_validate,ytrain,yvalidate,second_layer_dense,epochs,batch_size):
    print(' ### Build deep cnn model... ###')
    input_dim = nmf_tfidf_train.shape[1]  # Number of features
    model = Sequential()
    print('### Embedding words ###')
    model.add(Embedding(max_features,
                        embedding_dims,
                        input_length=maxlen))
    print('### Adding Conv1D')
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(input_dim))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(second_layer_dense))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(nmf_tfidf_train, ytrain,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(nmf_tfidf_validate, yvalidate))
    loss, accuracy = model.evaluate(nmf_tfidf_validate, yvalidate, verbose=False)
    return accuracy
