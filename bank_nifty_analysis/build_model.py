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
        target = np.load(str(files[1]))
        self.features = features.reshape((features.shape[0], 1, features.shape[1]))
        self.target = target.reshape((-1,1))
        self.input_shape = features.shape
        print(f"data loaded features{self.features.shape} target:{self.target.shape}")
        return self.features, self.target


    def build_model(self):

        self.model.add(LSTM(units=self.lstm_units, 
                            input_shape=self.input_shape,
                            return_sequences=True))
        self.model.add(Dropout(0.2))

        self.model.add(LSTM(50, activation='relu', return_sequences=True))
        self.model.add(Dropout(0.2))

        self.model.add(LSTM(50, activation='relu', return_sequences=True))
        self.model.add(Dropout(0.2))

        self.model.add(LSTM(50, activation='relu', return_sequences=True))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(units=self.output_shape,activation='softmax'))
        self.model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae', 'mape'])
        self.model.save("Nifty_bank_model.h5")
        print(self.model.summary(), end='\n')

    
    def train(self):

        self.model = self.model.fit(x=self.features, 
                                    y=self.target,
                                    epochs=70,
                                    verbose=True,
                                    batch_size=32)
        

    @classmethod
    def main(cls):

        lstm = LSTMModel(lstm_units=50, output_shape=1)
        lstm.read_processed_data()
        lstm.build_model()
        lstm.train()


if __name__ == '__main__':
    LSTMModel.main()

        


        
    
