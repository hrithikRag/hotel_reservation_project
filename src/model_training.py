import os
import sys
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *
from utils.common_functions import read_yaml, load_data
from scipy.stats import randint
import mlflow
import mlflow.sklearn

logger = get_logger(__name__)

class ModelTraining:

    def __init__(self, train_path, test_path, model_output_path):

        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path

        self.params_dist = LGBM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS

    
    def load_and_split_data(self):

        try:
            logger.info("Loading data from : {self.train_path}")
            train_df = load_data(self.train_path)

            logger.info("Loading data from : {self.test_path}")
            test_df = load_data(self.test_path)

            X_train = train_df.drop(columns=['booking_status'])
            y_train = train_df['booking_status']

            X_test = test_df.drop(columns=['booking_status'])
            y_test = test_df['booking_status']

            logger.info("data loaded, features and target splitted successfully")

            return X_train, y_train, X_test, y_test
        
        except Exception as e:
            logger.error(f"Error in loading data and splitting features : {e}")
            raise Exception("Error during loading and splitting data", sys)
        

    def train_lgbm(self, X_train, y_train):

        try:
            logger.info("Initializing the LGBM model")
            lgbm_model = lgb.LGBMClassifier(random_state = self.random_search_params['random_state'])

            logger.info("Initializing hyperparameter tuning")
            random_search = RandomizedSearchCV(
                                                estimator = lgbm_model,
                                                param_distributions = self.params_dist,
                                                n_iter =  self.random_search_params['n_iter'],
                                                cv =  self.random_search_params['cv'],
                                                n_jobs =  self.random_search_params['n_jobs'],
                                                verbose =  self.random_search_params['verbose'],
                                                random_state =  self.random_search_params['random_state'],
                                                scoring =  self.random_search_params['scoring'] 
                                                )
            
            logger.info("Starting hyperparameter tuning")
            random_search.fit(X_train, y_train)
            logger.info("Hyper parameter tuning done successfully")

            best_params = random_search.best_params_
            best_lgbm_model = random_search.best_estimator_
            logger.info(f'best params are : {best_params}')

            return best_lgbm_model
        
        except Exception as e :
            logger.error(f"Error in hyperparameter tuning : {e}")
            raise Exception("Failed to train LGBM model", sys) 
        

    def evaluate_model(self, model, X_test, y_test):

        try:
            logger.info("Evaluating our model on test data")

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            logger.info(f"accuracy = {accuracy}")
            logger.info(f"precision = {precision}")
            logger.info(f"recall = {recall}")
            logger.info(f"f1 = {f1}")

            return {"accuracy":accuracy , "precision":precision , "recall":recall ,"f1":f1}
        
        except Exception as e :
            logger.error(f"Error in eveluating model on test data : {e}")
            raise Exception("Failed to evaluate model on test data", sys) 
        

    def save_model(self, model):

        try:
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)

            logger.info("Saving the model")
            joblib.dump(model, self.model_output_path)
            logger.info(f"Model successfully saved to : {self.model_output_path}")

        except Exception as e :
            logger.error(f"Error in saving the model : {e}")
            raise Exception("Failed to save the model", sys) 
        

    def run(self):

        try:
            with mlflow.start_run():

                logger.info("STARTING MODEL TRAINING PIPELINE")
                logger.info("Starting MLFLOW experimentation")
                logger.info("Logging training and testing dataset to mlflow")

                mlflow.log_artifact(self.train_path, artifact_path='datasets')
                mlflow.log_artifact(self.test_path, artifact_path='datasets')

                X_train, y_train, X_test, y_test = self.load_and_split_data()
                best_lgbm_model = self.train_lgbm(X_train, y_train)
                perf_metrics = self.evaluate_model(best_lgbm_model, X_test, y_test)
                self.save_model(best_lgbm_model)

                logger.info("Logging model to mlflow")
                mlflow.log_artifact(self.model_output_path, artifact_path='models')

                logger.info("Logging params and performance metrics to mlflow")
                mlflow.log_params(best_lgbm_model.get_params())
                mlflow.log_metrics(perf_metrics)

                logger.info("MODEL TRAINING SUCCESSFULLY CONCLUDED")

        except Exception as e :
            logger.error(f"Error in model training pipeline : {e}")
            raise Exception("Failed to start model training pipeline", sys) 


if __name__=='__main__' :

    try:
        logger.info("INITIATING MODEL TRAINING PIPELINE")
        trainer = ModelTraining(PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH, MODEL_OUTPUT_PATH)
        trainer.run()

    except Exception as e :
        logger.error(f"Error in initiating model training pipeline : {e}")
        raise Exception("Failed to initiate model training pipeline", sys)   







