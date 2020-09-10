# data_prep.py

import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from joblib import dump


class BankNiftyPrediction():

    __slots__ = ["path", "df", "features", "target_col"]

    def __init__(self, path, features, target_col):
        self.path  = path
        self.df = pd.DataFrame()
        self.features = features
        self.target_col = target_col
        FILES = list(self.path.glob("*.csv"))
        print(f'total files found: {len(FILES)}')
        for file in FILES:
            df_ = pd.read_csv(str(file), parse_dates=["Date"])
            df_['Date'] = pd.to_datetime(df_["Date"])
            self.df =  self.df.append(df_)
            self.df = self.df.sort_values(by="Date")


    def generate_features(self):

        for lag in range(self.features):
            self.df["lag"+str(lag+1)] = self.df[self.target_col].shift(lag+1)
        print(self.df.columns)
        self.df = self.df.dropna()
        return self.df


    def scale_data(self):

        df = self.df
        close_ndarray = df[self.target_col].values.reshape((-1,1))
        mn = MinMaxScaler(feature_range=(0, 1))
        self.df["transformed_feature"] = mn.fit_transform(close_ndarray) 
        dump(mn, "transformer.joblib" )
        print(" transformer saved" )
        print(self.df.transformed_feature[:1])
        return self.df, mn
        
    
    def prepare_training_data(self):

        features_array = self.df.iloc[:, 7:].values
        target = self.df[self.target_col].values
        print(features_array[:1])
        np.save("target.npy", target)
        np.save("features.npy", features_array)
        print(f'saved processed data')
        return target, features_array

    @classmethod
    def main(cls):

        DATA_PATH = Path(Path().cwd()).resolve() / "data"
        features=2
        target_col="Close"
        bkobj = BankNiftyPrediction(path=DATA_PATH, features=features, target_col=target_col )
        bkobj.generate_features()
        bkobj.scale_data()
        bkobj.prepare_training_data()

if __name__ == "__main__":
    BankNiftyPrediction.main()
    
    







