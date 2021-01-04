import os
from sklearn.model_selection import train_test_split
import numpy as np
from data_management import load_dataset,save_pipeline
import pipeline
import config
import logging
import pdb

_logger = logging.getLogger(__name__)

def run_training() ->None:
    """Train the model."""

    # read the training dataset
    data = load_dataset(file_name = config.TRAINING_DATA_FILE )
    
    
    # divide train and test
    X_train,X_test,y_train,y_test = train_test_split(
        data[config.FEATURES],data[config.TARGET],random_state = 0 ,test_size = 0.1)

    # as outliers are present in target
    # transform the target
    y_train = np.log(y_train)
    

    pipeline.price_pipe.fit(X_train[config.FEATURES], y_train)

    # save the model
    
    _logger.info("saving model version")
    save_pipeline(pipeline_to_persist = pipeline.price_pipe )
    _logger.info("------model successfully saved----")






if __name__ == "__main__":
    # pdb.set_trace()
    run_training()