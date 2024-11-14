import pandas as pd
import pickle
import sys
import yaml
import os
import mlflow
from urllib.parse import urlparse
from sklearn.metrics import accuracy_score


os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/visheshgupta-BA/machinelearningpipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "visheshgupta-BA"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "0b5e32d3345131fe12c7575a8ff1a3b2cd8d2f3b"



## Load the parameters from params.yaml
params = yaml.safe_load(open('params.yaml'))['train']

def evaluate_model(model_path, data_path):
    data = pd.read_csv(data_path)
    X = data.drop(columns=['Outcome'], axis=1)
    y = data['Outcome']

    mlflow.set_tracking_uri('https://dagshub.com/visheshgupta-BA/machinelearningpipeline.mlflow')

    # load the trained model from the pickle file
    model = pickle.load(open(model_path, 'rb'))

    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)


    mlflow.log_metric('accuracy', accuracy)
    print(f'Model evaluation completed successfully. Accuracy: {accuracy}')



if __name__ == "__main__":
    evaluate_model(params['model'], params['data'])