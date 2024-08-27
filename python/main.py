import numpy as np
import os
import pandas as pd
import costants as c
import preprocessing.audio_preprocessing as preprocess
import features_script.feature_extraction as features_m
import matplotlib.pyplot as plt

def produce_features_dataframes(dataset_paths, features_dir):
    """Produce features dataframes for each dataset in dataset_paths"""
    if not os.path.exists(c.DATASETS_ROOT):
        print('Datasets root folder not found, please read the README.md file before running the script')
        exit(1)
    
    for dataset_path in dataset_paths:
        os.makedirs(features_dir, exist_ok=True)
        print(f'----------WORKING WITH {dataset_path}----------\n')
        dataset_name = dataset_path.split('/')[-2]

        if not os.path.exists(features_dir + dataset_name + '.csv'):
            files = preprocess.get_dataset_files(dataset_path)
            processed_signals = preprocess.preprocess_audio_files(files, c.TARGET_SR)
            print(f'Processed {len(processed_signals)}')
            
            print("now starting to extract features")
            # Extract features
            features = features_m.extract_features(processed_signals, c.TARGET_SR)
            print(features.shape)
            features = features.copy()
            #TODO: parse filenames and add to features
            parsed_df = features_m.parse_filenames(files, dataset_paths.index(dataset_path))
            features = pd.concat([parsed_df, features], axis=1)

            col = features.pop('actor')
            features.insert(0, 'actor', col)

            col = features.pop('emotion')
            features.insert(len(features.columns), 'emotion', col)

            features.to_csv(features_dir + dataset_name + '.csv', index=False)
            print(f'Features for {dataset_name} extracted\n')
        else:
            print(f'Features for {dataset_name} already extracted\n')

if __name__ == '__main__':
    # Produce features dataframes
    produce_features_dataframes(c.DATASETS_PATHS, c.FEATURES_PATH)

        
