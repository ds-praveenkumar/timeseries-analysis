# main.py

import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


class BankNiftyPrediction():

    __slots__ = ["path", "df"]

    def __init__(self, path):
        self.path  = path
        self.df = pd.DataFrame()
    

    def read_csv(self):

        FILES = list(self.path.glob("*.csv"))
        print(f'total files found: {len(FILES)}')
        for file in FILES:
            df_ = pd.read_csv(str(file))
            self.df =  self.df.append(df_)
        return self.df
    


    def scale_data(self, col_to_transform):
        self.df = self.read_csv()
        close_ndarray = self.df[col_to_transform].values.reshape((-1,1))
        mn = MinMaxScaler(feature_range=(0, 1))
        self.df["transformed_feature"] = mn.fit_transform(close_ndarray) 
        print(self.df.transformed_feature[:5])
        return self.df, mn

    def main(self ):
        pass
        

if __name__ == "__main__":
    # read data
    DATA_PATH = Path(Path().cwd()).resolve() / "data"
    bkobj = BankNiftyPrediction(DATA_PATH)
    bkobj.scale_data(col_to_transform="Close")
    







