import os
import sys
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml

logger=get_logger(__name__)

class DataIngestion:

    def __init__(self, config):
        self.config = config["data_ingestion"]
        self.bucket_name = self.config["bucket_name"]
        self.file_name = self.config["bucket_file_name"]
        self.train_test_ratio = self.config["train_ratio"]

        os.makedirs(RAW_DIR , exist_ok=True)
        logger.info( f"Ingestion object created" )
        

    def download_csv_from_gcp(self):

        try:

            logger.info( f"Data ingestion starting from bucket : {self.bucket_name} , extracting {self.file_name}" )

            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "uplifted-woods-471606-g1-3f228e9242d3.json"


            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.file_name)
            blob.download_to_filename(RAW_FILE_PATH)

            logger.info(f"CSV file successfully downloaded to {RAW_FILE_PATH}")

        except Exception as e:
            logger.error("Error while downloading CSV file")
            raise CustomException("Failed to download CSV file", sys)
        
    def split_data(self):

        try:
            logger.info(f"Starting train test split")

            data=pd.read_csv(RAW_FILE_PATH)
            train_data, test_data = train_test_split(data, test_size=1-self.train_test_ratio, random_state=42)
            train_data.to_csv(TRAIN_FILE_PATH, index=False)
            test_data.to_csv(TEST_FILE_PATH, index=False)

            logger.info(f"train data saved to {TRAIN_FILE_PATH}")
            logger.info(f"test data saved to {TEST_FILE_PATH}")

        except Exception as e:
            logger.error("Error while splitting into train and test")
            raise CustomException("Failed to split data into train and test", sys)
        
    def run(self):

        try:
            logger.info("Initiating data ingestion")

            self.download_csv_from_gcp()
            self.split_data()

            logger.info("DATA INGESTION SUCCESSFULLY COMPLETED")

        except Exception as ce:
            logger.error(f"Custom exception : {str(ce)}")

        finally:
            logger.info("DATA INGESTION CONCLUDED")


if __name__=="__main__":

    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()

