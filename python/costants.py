# Description: This file contains the constants used in the project

#constants used for audio preprocessing
LOWCUT = 100
HIGHCUT = 8000
ORDER = 5
RP =  5
TARGET_SR = 48000

#constants used for file paths
DATASETS_ROOT = 'datasets_raw/'
DATASETS_PATHS = ['datasets_raw/EMOVO/', 'datasets_raw/RAVDESS/']
FEATURES_PATH = 'features/'

#constants used for classification
PARAMS_BASE_PATH = 'params/base_classifiers/'
PARAMS_LOO_PATH = 'params/loo_classifiers/'