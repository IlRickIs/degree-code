#this file is created to apply the feature extraction pipeline to emozionalmente dataset
#the pipeline is the same as the one in main.py but it changes slightly to adapt to the different dataset

import costants as c
from functions import *
import os
import preprocessing.audio_preprocessing as preprocess
import features_script.feature_extraction as features_m

dataset_path = "datasets_raw/EMOZIONALMENTE/audio/"
features_dir = "features/"

def get_dataset_filenames(path):
    """Get the filenames of the dataset"""
    files = []
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files

#given the features df with the firtst column being the filename, this function adds the actor and emotion columns
def add_actor_emotion_columns(features_df, metadata):
    """Add the actor and emotion columns to the dataframe"""
    actor = []
    emotion = []
    for file in features_df['filename']:
        filename = os.path.basename(file)
        actor.append(metadata[metadata['file_name'] == filename]['actor'].values[0])
        emotion.append(metadata[metadata['file_name'] == filename]['emotion_expressed'].values[0])
    features_df['actor'] = actor
    features_df['emotion'] = emotion
    return features_df
    

def add_filename_column(files):
    """Add a column with the filename to the dataframe"""
    filename_df = pd.DataFrame(columns=['filename'])
    filename_df['filename'] = files.split('/')[-1]
    return filename_df

    

if __name__ == '__main__':
    #this is to extract the features from the emozionalmente dataset
    # os.makedirs(features_dir, exist_ok=True)
    dataset_name = dataset_path.split('/')[-3]
    # print(f'----------FEATURE EXTRACTION AND PREPROCESSING OF {dataset_name}----------\n')
    # files = get_dataset_filenames(dataset_path)
    # processed_signals = preprocess.preprocess_audio_files(files, c.TARGET_SR)
    # print(f'Processed {len(processed_signals)}')

    # print("now starting to extract features")
    # # Extract features
    # features = features_m.extract_features(processed_signals, c.TARGET_SR)
    # print(features.shape)
    # features = features.copy()

    # filename_df = add_filename_column(files)
    # features = pd.concat([filename_df, features], axis=1)
    # features.to_csv(features_dir + dataset_name + '.csv', index=False)

    features = pd.read_csv('features/EMOZIONALMENTE.csv')
    #rename all the filenames as the last part of the path
    features['filename'] = features['filename'].apply(lambda x: x.split('/')[-1])

    metadata = pd.read_csv('datasets_raw/EMOZIONALMENTE/metadata/samples.csv')
    parsed_df = add_actor_emotion_columns(features, metadata)

    col = features.pop('filename')
    print(col)
    features.insert(0, 'filename', col)

    col = features.pop('actor')
    features.insert(1, 'actor', col)

    col = features.pop('emotion')
    features.insert(2, 'emotion', col)

    features.to_csv(features_dir + dataset_name + '.csv', index=False)
    print(f'Features for {dataset_name} extracted\n')



