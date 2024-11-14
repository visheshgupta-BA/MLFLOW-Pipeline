import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pickle
import sys
import yaml
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlflow.models import infer_signature
import mlflow
from urllib.parse import urlparse


os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/visheshgupta-BA/machinelearningpipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "visheshgupta-BA"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "0b5e32d3345131fe12c7575a8ff1a3b2cd8d2f3b"

def hyperparameter_tuning(X_train, y_train, param_grid):
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search


## Load the parameters from params.yaml

params = yaml.safe_load(open('params.yaml'))['train']




def train_model(data_path, model_path, random_state, n_estimators, max_depth):
    
    data = pd.read_csv(data_path)
    X = data.drop(columns=['Outcome'], axis=1)
    y = data['Outcome']

    mlflow.set_tracking_uri('https://dagshub.com/visheshgupta-BA/machinelearningpipeline.mlflow')
    mlflow.start_run()

# Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    signature = infer_signature(X_train, y_train)

    # Define the hyperparameter grid
    param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
    }

    # Perform hyperparameter tuning
    grid_search = hyperparameter_tuning(X_train, y_train, param_grid)

    # get the best model
    best_model = grid_search.best_estimator_

    # predict and evaluate the model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('best_n_estimators', grid_search.best_params_['n_estimators'])
    mlflow.log_param('best_max_depth', grid_search.best_params_['max_depth'])
    mlflow.log_param('best_min_samples_split', grid_search.best_params_['min_samples_split'])
    mlflow.log_param('best_min_samples_leaf', grid_search.best_params_['min_samples_leaf'])
    mlflow.log_text(str(cm),"confusion_matrix.txt")
    mlflow.log_text(cr,"classification_report.txt")


    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    if tracking_url_type_store!= "file":
        mlflow.sklearn.log_model(best_model, "model", registered_model_name="Best Model")
    
    else:
        mlflow.sklearn.log_model(best_model, "model", infer_signature=signature)
    

    ## create the directory to save the model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # save the model to a pickle file
    with open(model_path, 'wb') as file:
        pickle.dump(best_model, file)

    print(f'Model training completed successfully. Model saved to {model_path}')



if __name__ == "__main__":
    train_model(params['data'], params['model'], params['random_state'], params['n_estimators'], params['max_depth'])
    



