# build_model.py
from pathlib import Path

import pandas as pd 
import tensorflow as tf 
from tensorflow.keras.layers import Dense, Dropout, LSTM 
import numpy as np

class LSTMModel():

    def __init__(self, lstm_units=int, input_shape=tuple, output_shape=tuple):

        self.lstm_units = lstm_units
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = tf.keras.Sequential()


    def build_model(self):

        self.model.add(LSTM(units=self.lstm_units, input_shape=self.input_shape, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=2,activation='softmax'))
        self.model.compile(optimizer='adam', loss='mse', metrics=["mse"])
        print(self.model.summary(), end='\n')

    
    def train(self):

        features = None
        target = None
        path = Path(Path().cwd()).resolve()
        files = list(path.glob("*.npy"))
        features = np.load(str(files[0]))
        target = np.load(str(files[1]))
        features = features.reshape((features.shape[0], 1, features.shape[1]))
        target = target.reshape((-1,1))
        print(f"data loaded features{features.shape} target:{target.shape}")
        self.model = self.model.fit(x=features, 
                                    y=target,
                                    epochs=30,
                                    verbose=True,
                                    batch_size=64)
        

    @classmethod
    def main(cls):

        lstm = LSTMModel(lstm_units=256, input_shape=(250, 6), output_shape=1)
        lstm.build_model()
        lstm.train()


if __name__ == '__main__':
    LSTMModel.main()

        


        
    
