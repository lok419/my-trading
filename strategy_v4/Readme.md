# Return predictions

## Aims to create a standardized pipeline for return predictions

1. Data Layer: single process to generate all raw features, which will then be shared across all models. Ideally save into a DB.

2. Model Layer: select the features from (1) and create a return prediction model. Choices of preprocessing, train/test split, derived features are decided here

3. Evaluate Layer: Given output from model layer, evaluate the return prediction accuray using standard metrics like R2, MSE, MAE, F2...

4. Backtest layer: Given the return prediction, create a simple strategy and evaluate the PNL metrics

5. Output layer: the best performed model will evenually used in <b>strategy_v2</b> to generate signals for my real portfolios

## Why we need such framework

Everytime we build models, we almost need to redo (1)-(5), most of the time are spent in preparing the data and evaluate the models. Purpose of the framework is once we have this settled, we could spend most of time (~80%) in <b>Model Layer</b> to improve the performance without worrying about other stuff.

## Tech Stack

1. Data Layer: Ideally some postreg database, for now, just using parquet, as the model layer just need to read the entire data file.
2. Scheduler: Airlfow to create entire pipeline




