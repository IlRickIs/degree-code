import numpy as np
import os
import pandas as pd
import costants as c
from audio_preprocessing import *
from feature_extraction import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dataset_paths = ['datasets_raw/EMOVO/', 'datasets_raw/RAVDESS/']
    features_dir = 'features/'
    
    for dataset_path in dataset_paths:
        os.makedirs(features_dir, exist_ok=True)
        print(f'----------WORKING WITH {dataset_path}----------\n')
        dataset_name = dataset_path.split('/')[-2]

        if not os.path.exists(features_dir + dataset_name + '.csv'):
            files = get_dataset_files(dataset_path)
            processed_signals = preprocess_audio_files(files, c.TARGET_SR)

            print(f'Processed {len(processed_signals)}')
            
            
            print("now starting to extract features")

            # Extract features
            features = extract_features(processed_signals, c.TARGET_SR)
            print(features.shape)
            features = features.copy()
            #TODO: parse filenames and add to features
            features.to_csv(features_dir + dataset_name + '.csv', index=False)
            print(f'Features for {dataset_name} extracted\n')
        else:
            print(f'Features for {dataset_name} already extracted\n')
        
        files = get_dataset_files(dataset_path)
        print(files)

        # Classification task
        print("now starting the classification task")
        features = pd.read_csv(features_dir + dataset_name + '.csv')

        
