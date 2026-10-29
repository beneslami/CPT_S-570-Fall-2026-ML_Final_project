import argparse
from data import Data
import pathlib
from ML_interface import Model, ML
from models.logistic_regression import LogisticRegressionModel
from models.RNNGRU import RNNGRU

parent_dir = pathlib.Path(__file__).parent.parent
dataset_dir = parent_dir / "datapoints"
symbols = [p.name for p in pathlib.Path(dataset_dir).glob("*") if p.is_dir()]
timeframes = ["1m", "5m", "15m", "30m", "1h"]
def main():

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", type=str, help="BTC, XAU, etc.")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe: (1m, 5m, 15m, 30m, 1h, 4h)") 
    parser.add_argument("--model", choices=["logistic", "gru"], default="gru")
    args = parser.parse_args()
    symbol = args.symbol
    timeframe = args.timeframe

    ## 1- Dataset preparation
    csv_path = dataset_dir / args.symbol / f"{args.symbol}_{args.timeframe}.csv"
    dataObj = Data(args.symbol, str(csv_path), str(csv_path), str(csv_path), args.timeframe)
    all_data = dataObj.get_train_data().sort_values("time").reset_index(drop=True)

    split_at = int(len(all_data) * 0.90)
    train_data = all_data.iloc[:split_at].copy()
    test_data = all_data.iloc[split_at:].copy()
    
    ## 2- Model initialization
    if args.model == "gru":
            backend: Model = RNNGRU(args.symbol, timeframe=args.timeframe, random_state=42)
    else:
        backend = LogisticRegressionModel(
            args.symbol, timeframe=args.timeframe, max_iter=100000, random_state=42)

    ml = ML(backend)  # front end: only knows the Model interface

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