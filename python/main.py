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
        print(f'----------WORKING WITH {dataset_path}----------')
        dataset_name = dataset_path.split('/')[-2]

        if not os.path.exists(features_dir + dataset_name + '.csv'):
            files = get_dataset_files(dataset_path)
            processed_signals = preprocess_audio_files(files, c.TARGET_SR)

            print(f'Processed {len(processed_signals)}')
            test = []
            test.append(processed_signals[0])
            print('first signal: ' + str(test))
            
            print("now starting to extract features")

            # Extract features
            # features = extract_features(processed_signals, c.TARGET_SR)
            # print(type(features))
            # print(features.shape)
            # features = features.copy()
            # features.to_csv(features_dir + dataset_path.split('/')[-2] + '.csv', index=False)

            # Extract MFCCs
            mfccs = extract_mfccs(test, c.TARGET_SR)
            print('Done!')
        else:
            print(f'Features for {dataset_name} already extracted')