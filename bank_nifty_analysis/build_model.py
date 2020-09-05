# build_model.py
from pathlib import Path

import pandas as pd 
import tensorflow as tf 
from tensorflow.keras.layers import Dense, Dropout, LSTM 
import numpy as np

class LSTMModel():

    def __init__(self, lstm_units, 
                       output_shape):

        self.lstm_units = lstm_units
        self.output_shape = output_shape
        self.model = tf.keras.Sequential()
        self.features = None
        self.target = None


    def read_processed_data(self):

        path = Path(Path().cwd()).resolve()
        files = list(path.glob("*.npy"))
        features = np.load(str(files[0]))
        train_len = int(len(features) * 0.70)
        train_set = features[:train_len]
        test_set = features[train_len:]
        target = np.load(str(files[1]))
        target_train = target[:train_len]
        target_test = target[train_len:]
        self.features = train_set.reshape((train_set.shape[0], 1, train_set.shape[1]))
        self.target = target_train.reshape((-1,1))
        self.input_shape = features.shape
        print(f"data loaded features{self.features.shape} target:{self.target.shape}")
        return self.features, self.target


    def build_model(self):

        self.model.add(LSTM(self.lstm_units, 
                            input_shape=(1, 2),
                            activation="relu",
                            return_sequences=True
                            ))


        self.model.add(LSTM(50, activation='relu', return_sequences=True))

        self.model.add(LSTM(50, activation='relu', return_sequences=True))

        self.model.add(LSTM(50, activation='relu', return_sequences=True))

        self.model.add(Dense(10, activation="relu"))
        self.model.add(Dense(units=self.output_shape ))

        self.model.compile(optimizer='adam', loss='mape', metrics=['mape', 'accuracy'])
        self.model.save("Nifty_bank_model.h5")
        print(self.model.summary(), end='\n')

    
    def train(self, epochs):

        self.model = self.model.fit(x=self.features, 
                                    y=self.target,
                                    epochs=epochs,
                                    verbose=True,
                                    batch_size=32)
        

    @classmethod
    def main(cls):

        lstm = LSTMModel(lstm_units=256, output_shape=1)
        lstm.read_processed_data()
        lstm.build_model()
        lstm.train(epochs=70)


if __name__ == '__main__':
    LSTMModel.main()

        


        
    
