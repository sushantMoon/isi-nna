import numpy as np
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.callbacks import TensorBoard


class SalesPredictionLSTM(object):

    def __init__(self, layers, dropout, batch_size, epochs, validation_split):
        """
            Build RNN (LSTM) model on top of Keras and Tensorflow
        """
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.layers = layers
        self.dropout = dropout
        self.model = None

    def build_model(self):
        self.model = Sequential()
        self.model.add(Dense(32, input_shape=(self.layers[0], 1)))
        self.model.add(LSTM(self.layers[1], return_sequences=True))
        self.model.add(Dropout(self.dropout))
        self.model.add(LSTM(self.layers[2], return_sequences=False))
        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(self.layers[3]))
        self.model.add(Activation("tanh"))
        self.model.compile(loss="mean_squared_error", optimizer="rmsprop", metrics=['accuracy'])

    def train(self, x_train, y_train):
        """
            Function to train the model
        """
        tensorboard = TensorBoard(log_dir='logs', update_freq='batch')
        self.model.fit(
            x_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split,
            callbacks=[tensorboard]
        )

    def predict(self, history):
        """
            Function to make the predictions
        """
        prediction = self.model.predict(history)
        prediction = np.reshape(prediction, (prediction.size,))
        return prediction
    
    def save_weights(self, path):
        self.model.save_weights(path)
        return 
    
    def load_weights(self, path):
        self.model.load_weights(path)
        self.model.compile(loss="mean_squared_error", optimizer="rmsprop", metrics=['accuracy'])
        return 
        
        
