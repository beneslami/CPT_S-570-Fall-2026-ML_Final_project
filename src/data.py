"""
Developed by Benaymin Eslami
"""

import pandas as pd

class Data:
    def __init__(self, symbol, train_path: str, eval_path: str, test_path: str, timeframe: str):
        self.symbol = symbol
        self.timeframe = timeframe
        self.train_path = train_path
        self.eval_path = eval_path
        self.test_path = test_path
        self.df = pd.DataFrame()
        self.train_data = pd.DataFrame()
        self.eval_data = pd.DataFrame()
        self.test_data = pd.DataFrame()
        self.column = []
        self.__load() 

    ## Private methods
    def __load(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.train_data = pd.read_csv(self.train_path)
        self.eval_data = pd.read_csv(self.eval_path)
        self.test_data = pd.read_csv(self.test_path)
        self.column = self.train_data.columns.tolist()
        self.train_data = self.__clean(self.train_data)
        self.eval_data = self.__clean(self.eval_data)
        self.test_data = self.__clean(self.test_data)
        return self.train_data, self.eval_data, self.test_data

    def __clean(self, df: pd.DataFrame) -> pd.DataFrame:
        assert not df.empty, "DataFrame is empty"
        df = df.dropna().reset_index(drop=True)
        return df
    
    ## Public methods
    def get_train_data(self) -> pd.DataFrame:
        return self.train_data

    def get_eval_data(self) -> pd.DataFrame:
        return self.eval_data

    def get_test_data(self) -> pd.DataFrame:
        return self.test_data

    def get_features(self) -> list[str]:
        return self.column