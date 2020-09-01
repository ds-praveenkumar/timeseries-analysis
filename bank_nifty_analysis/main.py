# main.py

import pandas as pd
from pathlib import Path


# read data
DATA_PATH = Path(Path().cwd()).resolve() / "data"
FILES = list(DATA_PATH.glob("*.csv"))

print(f'total files: {len(FILES)}')
for file in FILES:
    df = pd.DataFrame()
    df_ = pd.read_csv(str(file))
    df = df.append(df_)

print(df.columns)



