# load the train and test dataset
# train the algorithm
# save the metrics and params

import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
from get_data import read_params
import argparse
import joblib
import json

import mlflow

def eval_metrics(ytest,ypred):
    rmse=np.sqrt(mean_squared_error(ytest,ypred))
    mae=mean_absolute_error(ytest,ypred)
    r2=r2_score(ytest,ypred)
    return rmse,mae,r2

def train_and_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]

    alpha = config["estimators"]["ElasticNet"]["params"]["alpha"]
    l1_ratio = config["estimators"]["ElasticNet"]["params"]["l1_ratio"]

    target = [config["base"]["target_col"]]

    train = pd.read_csv(train_data_path, sep=',',)
    test = pd.read_csv(test_data_path, sep=',',)

    ytrain = train[target]
    ytest = test[target]

    xtrain = train.drop(labels=target,axis=1)
    xtest = test.drop(labels=target,axis=1)

################################## MLFLOW BEGINS ##################################

    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)

    mlflow.set_experiment(mlflow_config["experiment_name"])

    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:
        model=ElasticNet(alpha=alpha,l1_ratio=l1_ratio, random_state=random_state)
        model.fit(xtrain,ytrain)

        ypred=model.predict(xtest)

        (rmse,mae,r2) = eval_metrics(ytest,ypred)

        mlflow.log_param("alpha",alpha)
        mlflow.log_param("l1_ratio",l1_ratio)

        mlflow.log_metric("rmse", rmse) # type: ignore
        mlflow.log_metric("mae", mae) # type: ignore
        mlflow.log_metric("r2", r2) # type: ignore


        # The traditional model logging method below is also going to be commented out.
        # Instead, we will use mlflow.
        # os.makedirs(model_dir, exist_ok=True)
        # model_path = os.path.join(model_dir, "model.joblib")

        # joblib.dump(model, model_path)

        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(model, 
                                     "model", 
                                     registered_model_name=mlflow_config["registered_model_name"]
                                    )
        else:
            mlflow.sklearn.log_model(model, "model")


################################## MLFLOW ENDS ##################################

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config",default="parameters.yaml")
    parsed_args=args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)
