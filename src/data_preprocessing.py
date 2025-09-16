import os
import pandas as pd
import numpy as np
import sys
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml, load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

logger=get_logger(__name__)

class DataProcessor:

    def __init__(self, train_path, test_path, processed_dir, config_path):

        self.train_path=train_path
        self.test_path=test_path

        self.processed_dir=processed_dir

        self.config=read_yaml(config_path)

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)


    def preprocess_data(self,df): 

        try:
            logger.info("STARTING DATA PRE-PROCESSING")
            logger.info("Dropping columns : Booking_ID , arrival_year")

            df.drop(columns=['Booking_ID', 'arrival_year'],inplace=True)

            logger.info("Dropping duplicate rows")

            df.drop_duplicates(inplace=True)

            cat_cols=self.config['data_processing']['categorical_columns']
            num_cols=self.config['data_processing']['numerical_columns']

            logger.info("Performing label encoding")

            label_encoder = LabelEncoder()
            mappings={}

            for col in cat_cols:
                df[col] = label_encoder.fit_transform(df[col])
                mappings[col] = { label:code for label,code in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)) }

            logger.info("Label mappings are :")
            
            for col, mapping in mappings.items():
                logger.info(f"{col} : {mapping}")

            logger.info("Performing skewness handling")

            skew_threshold=self.config['data_processing']['skewness_threshold']
            skewness=df[num_cols].apply(lambda x: x.skew())

            for col in skewness[skewness>skew_threshold].index:
                df[col]=np.log1p(df[col])

            return df
        
        except Exception as e:
            logger.error(f"Error during pre-processing step : {e}")
            raise CustomException("Error in data pre-prcessing", sys)
        

    def balance_data(self,df):

        try: 
            logger.info("Handling imbalanced data")

            X = df.drop(columns=['booking_status'])
            y = df['booking_status']

            smote=SMOTE(random_state=42)

            X_resampled, y_resampled = smote.fit_resample(X,y)
        
            balanced_df=pd.DataFrame(X_resampled, columns=X.columns)
            balanced_df['booking_status']=y_resampled

            logger.info("Data balanced successfully")

            return balanced_df
        
        except Exception as e:
            logger.error(f"Error during data imbalance handling step : {e}")
            raise CustomException("Error in data imbalance handling", sys) 
        

    def select_features(self,df):

        try:
            logger.info("Starting feature selection")

            X = df.drop(columns=['booking_status'])
            y = df['booking_status']

            model=RandomForestClassifier(random_state=42)
            model.fit(X,y)

            feature_importance=model.feature_importances_
            feature_importance_df=pd.DataFrame({'feature':X.columns, 'importance':feature_importance})
            top_feature_importance_df=feature_importance_df.sort_values(by=['importance'], ascending=False)
            num_features_to_select=self.config['data_processing']['no_of_features']
            top_features=top_feature_importance_df['feature'].head(num_features_to_select).values

            top_10_df=df[list(top_features)+['booking_status']]

            logger.info(f'features selected : {top_features}')
            logger.info('Feature selection completed successfully')

            return top_10_df
        
        except Exception as e:
            logger.error(f"Error during feature selection : {e}")
            raise CustomException("Error in feature selection process", sys)


    def save_data(self, df, path):

        try :
            logger.info('Saving the processed data')

            df.to_csv(path, index=False)

            logger.info(f'CSV file saved successfully to : {path}')

        except Exception as e:
            logger.error(f"Error during saving the processed data : {e}")
            raise CustomException("Error in saving the processed file", sys)
        

    def process(self):

        try:
            logger.info("Loading train data from raw directory")

            train_df=load_data(self.train_path)
            train_df=self.preprocess_data(train_df)
            train_df=self.balance_data(train_df)
            train_df=self.select_features(train_df)
            self.save_data(train_df, PROCESSED_TRAIN_DATA_PATH)

            logger.info("TRAIN DATA PRE-PROCESSING CONCLUDED SUCCESSFULLY")

            logger.info("Loading test data from raw directory")

            test_df=load_data(self.test_path)
            test_df=self.preprocess_data(test_df)
            test_df=self.balance_data(test_df)
            test_df=test_df[train_df.columns]
            self.save_data(test_df, PROCESSED_TEST_DATA_PATH)

            logger.info("TEST DATA PRE-PROCESSING CONCLUDED SUCCESSFULLY")

        except Exception as e:
            logger.error(f"Error in data pre-processing pipeline : {e}")
            raise CustomException("Error in data pre-processing pipeline", sys)
        
        
if __name__ == '__main__':

    try:
        logger.info("Starting data pre-processing for train and test data")

        processor = DataProcessor(TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH)
        processor.process()

    except Exception as e:
            logger.error(f"Error in starting data pre-processing : {e}")
            raise CustomException("Error in starting data pre-processing", sys)


        






 


        






