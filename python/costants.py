# Description: This file contains the constants used in the project
import random as ran
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
NORMALIZE_MATRIX = 'true'  #can be literal true or None, used to normalize cf matrix
USE_PARAMS = 'base' #can be 'base' or 'loso', used to select the parameters to use
PARAMS_BASE_PATH = 'params/base_classifiers/'
PARAMS_LOSO_PATH = 'params/loso_classifiers/'

REPORTS_BASE_PATH = 'reports/base_classifiers/'
REPORTS_CM_PATH = 'reports/confusion_matrices/'
REPORTS_LOSO_PATH = 'reports/loso_classifiers/'
SINGLE_ACTOR_REPORTS_PATH = 'reports/single_actor_reports/'
LABELS_MAP = {1: 'neu', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'ang', 6: 'fear', 7: 'disgust', 8: 'surprise'}

#RANDOM_STATE = ran.randint(0, 1000) #use this to randomize training and testing data
RANDOM_STATE = 42   #for reproducibility