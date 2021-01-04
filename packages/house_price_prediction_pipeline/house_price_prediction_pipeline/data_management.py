import pandas as pd
import joblib
import config
import os
import typing as t
# import __version__ as _version
import logging


_logger = logging.getLogger(__name__)

def load_dataset(*,file_name : str) ->pd.DataFrame:
    data = pd.read_csv(config.DATASET_DIR +'/' + file_name)
    return data

def save_pipeline(*,pipeline_to_persist) ->None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """
    VERSION_PATH = config.PACKAGE_ROOT +'/' + 'VERSION'
    
    with open(VERSION_PATH, 'r') as version_file:
        __version__ = version_file.read().strip()

    print('version:',__version__)


    # prepare versioned save file name
    save_file_name = f"{config.PIPELINE_SAVE_FILE}{__version__}.pkl"
    save_path = config.TRAINED_MODEL_DIR +'/' + save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)
    _logger.info(f"saved pipeline: {save_file_name}")
    # print('done')

def remove_old_pipelines(*, files_to_keep:t.List[str]) -> None:
    """
    Remove old model pipelines.

    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    However, we do also include the immediate previous
    pipeline version for differential testing purposes.
    """

    do_not_delete = files_to_keep + [ "__init__.py"]
    dir = config.TRAINED_MODEL_DIR
    # print('dir:',dir)
    # print()
    files = os.listdir(dir)
    for model_file in files:
        if model_file not in do_not_delete:
            file = dir +'/' + model_file
            os.remove(file)
            print('succesfully deleted file:',model_file)
            # model_file.unlink()
            




