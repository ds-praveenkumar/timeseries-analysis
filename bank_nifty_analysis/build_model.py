# build_model.py
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
import tensorflow as tf 
tf.set_random_seed(123)
from tensorflow.keras.layers import Dense, Dropout, LSTM 
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
from joblib import load


class LSTMModel():

    def __init__(self, lstm_units, 
                       output_shape):

        self.lstm_units = lstm_units
        self.output_shape = output_shape
        self.model = tf.keras.Sequential()
        self.features = None
        self.target = None
        self.history = None
        self.x_test = None
        self.y_test = None



    def read_processed_data(self):

        path = Path(Path().cwd()).resolve()
        files = list(path.glob("*.npy"))
        features = np.load(str(files[0]))
        train_len = int(len(features) * 0.75)
        train_set = features[:train_len]
        test_set = features[train_len:]
        target = np.load(str(files[1]))
        target_train = target[:train_len]
        target_test = target[train_len:]
        self.x_test = test_set
        self.y_test = target_test
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
        self.model.add(Dense(units=self.output_shape, activation="sigmoid" ))

        self.model.compile(optimizer='adam', loss='mape', metrics=['mape'])
        self.model.save("Nifty_bank_model.h5")
        print(self.model.summary(), end='\n')

    
    def train(self, epochs):

        checkpoint = ModelCheckpoint(filepath="./model", 
                                    save_best_only=True,
                                    save_weights_only=False,
                                    mode='auto')

        self.history = self.model.fit(x=self.features, 
                                    y=self.target,
                                    epochs=epochs,
                                    verbose=True,
                                    batch_size=32,
                                    callbacks=[checkpoint])
        return self.history

    def evaluate(self,
                 y,
                 x=range(30),
                 title= "training loss per epoch",
                 xlabel="epochs",
                 ylabel="loss",
                 file="evalute.png",
                 ):

        
        print("showing plot")
        plot = sns.lineplot( y=y, x=x)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        fig = plot.get_figure()
        fig.savefig(file)
        plt.show()

    
    def predict(self):

        print("saving prediction")
        predictions = self.model.predict(self.x_test)
        mn = load("transformer.joblib")
        predictions = mn.inverse_transform(predictions)
        print(predictions[0])
        pd.DataFrame(predictions).to_csv("predictions.csv", index=False)
        plot = sns.lineplot(x=self.y_test.shape[0], y=self.y_test )
        plot = sns.lineplot( y=predictions[0], x=predictions.shape[0])
        plt.title("Forecast Plot")
        plt.xlabel("sample")
        plt.ylabel("price")
        fig = plot.get_figure()   
        fig.savefig("prediction.png")
        plt.show()
        
        

    @classmethod
    def main(cls):

        lstm = LSTMModel(lstm_units=256, output_shape=1)
        lstm.read_processed_data()
        lstm.build_model()
        history = lstm.train(epochs=30)
        lstm.evaluate(y=history.history["loss"])
        lstm.predict()


if __name__ == '__main__':
    LSTMModel.main()

        


        
    
