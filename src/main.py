from data import Data
import pathlib
from ML_interface import Model, ML
from models.logistic_regression import LogisticRegressionModel

parent_dir = pathlib.Path(__file__).parent.parent
dataset_dir = parent_dir / "datapoints"
symbols = [p.name for p in pathlib.Path(dataset_dir).glob("*") if p.is_dir()]
timeframes = ["1m", "5m", "15m", "30m", "1h"]
def main():
    symbol = "BTC"
    timeframe = "1h"

    ## 1- Dataset preparation
    train_dataset = f"{dataset_dir}/{symbol}/{symbol}_{timeframe}.csv"
    eval_dataset = f"{dataset_dir}/{symbol}/{symbol}_{timeframe}.csv"
    test_dataset = f"{dataset_dir}/{symbol}/{symbol}_{timeframe}.csv"
    dataObj = Data(symbol, train_dataset, eval_dataset, test_dataset, timeframe)

    train_data = dataObj.get_train_data()
    test_data = dataObj.get_test_data()

    ## 2- Model initialization
    model: Model = LogisticRegressionModel(symbol, max_iter=1000, random_state=42)   # backend: the real implementation
    ml = ML(model)  # front end: only knows the Model interface

    ## 3- feature engineering
    ml.feature_engineering(train_data, test_data)
    
    ## 4- Model training, evaluation, and testing
    ml.train()

    ## 5- Model testing
    ml.test()

    ## 6- Model evaluation
    ml.evaluate()
    

if __name__ == "__main__":
    main()