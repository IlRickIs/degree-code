#functions for the main program
import numpy as np
import os
import pandas as pd
import costants as c
import preprocessing.audio_preprocessing as preprocess
import features_script.feature_extraction as features_m

def produce_features_dataframes(dataset_paths, features_dir):
    """Produce features dataframes for each dataset in dataset_paths"""
    if not os.path.exists(c.DATASETS_ROOT):
        print('Datasets root folder not found, please read the README.md file before running the script')
        exit(1)
    
    for dataset_path in dataset_paths:
        os.makedirs(features_dir, exist_ok=True)
        print(f'----------FEATURE EXTRACTION AND PREPROCESSING OF {dataset_path}----------\n')
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

def classify_task_base_classifier():
    """Classify task using the base classifier., the base classifier
    uses three different classifiers: SVM, Decision Tree and Linear discriminant analysis\n
    is called base classifier because it doesn't use LOSO validation approach"""
    import classification.BaseClassifier as base_classifier
    
    for dataset_path in c.DATASETS_PATHS:
        dataset_name = dataset_path.split('/')[-2]
        print(f'----------Base classification of {dataset_name}----------\n')

        #load the dataset and filter out the emotions that are not in the reference file
        df = pd.read_csv(c.FEATURES_PATH + dataset_name + '.csv')
        df = df[~df['emotion'].isin([2, 7, 8, 6])]

        # Load features and target 
        features = df.drop(columns=['actor', 'emotion'])
        target = df['emotion']

        classifier = base_classifier.BaseClassifier(features, target, dataset_name)
        print('SVM classifier')
        classifier.svm_classifier()
        print()

        print('Decision Tree classifier')
        classifier.decision_tree_classifier()
        print()

        print('Linear Discriminant Analysis classifier')
        classifier.lda_classifier()
        print()

        print(f'Base classification of {dataset_name} completed\n') 
        print()
    
def classify_task_loso_classifier():
    """Classify task using the LOSO (Leave One Subject Out) validation approach"""
    import classification.LosoClassifier as loso_classifier
    for dataset_path in c.DATASETS_PATHS:
        dataset_name = dataset_path.split('/')[-2]
        print(f'----------LOSO classification of {dataset_name}----------\n')

        # Load the dataset and filter out the emotions that are not in the reference file
        df = pd.read_csv(c.FEATURES_PATH + dataset_name + '.csv')
        if dataset_name == "EMOVO":
            df = df[~df['emotion'].isin([2, 7, 8, 6])]
        else:
            df = df[~df['emotion'].isin([1, 7, 8, 6])]

        # Load features and target
        features = df.drop(columns=['emotion'])
        target = df['emotion']

        classifier = loso_classifier.LosoClassifier(features, target, dataset_name)
        
        print('SVM classifier')
        classifier.svm_classifier()
        print()

        print('Decision Tree classifier')
        classifier.decision_tree_classifier()
        print()

        print('Linear Discriminant Analysis classifier')
        classifier.lda_classifier()
        print()
        
        print(f'LOSO classification of {dataset_name} completed\n')
        print()

def classify_task_cross_corpus_classifier(train_dataset, test_dataset):
    """Classify task using cross corpus training approach"""
    import classification.CrossCorpusClassifier as cross_corpus_classifier
    print(f'----------Using {train_dataset} as training dataset and {test_dataset} as testing dataset----------\n')

    # Load the training dataset and filter out the emotions that are not in the reference file
    df_train = pd.read_csv(c.FEATURES_PATH + train_dataset + '.csv')
    df_train = df_train[~df_train['emotion'].isin([2, 7, 8, 6])]

    # Load the testing dataset and filter out the emotions that are not in the reference file
    df_test = pd.read_csv(c.FEATURES_PATH + test_dataset + '.csv')
    df_test = df_test[~df_test['emotion'].isin([2, 7, 8, 6])]

    # Load features and target
    features_train = df_train.drop(columns=['emotion', 'actor'])
    target_train = df_train['emotion']
    features_test = df_test.drop(columns=['emotion', 'actor'])
    target_test = df_test['emotion']

    classifier = cross_corpus_classifier.CrossCorpusClassifier(features_train, target_train, features_test, target_test, train_dataset)

    print('SVM classifier')
    classifier.svm_classifier()
    print()

    print('Decision Tree classifier')
    classifier.decision_tree_classifier()
    print()

    print('Linear Discriminant Analysis classifier')
    classifier.lda_classifier()
    print()

def produce_combined_dataset(dataset_paths):
    """Produce a combined dataset from the datasets in dataset_paths"""
    combined_df = pd.DataFrame()
    for dataset_path in dataset_paths:
        dataset_name = dataset_path.split('/')[-2]
        df = pd.read_csv(c.FEATURES_PATH + dataset_name + '.csv')
        # modify the column actor to avoid duplicates
        df['actor'] = df['actor'].apply(lambda x: str(x) + dataset_name)
        combined_df = pd.concat([combined_df, df], axis=0)

    combined_df.to_csv(c.FEATURES_PATH + "COMBINED"+".csv", index=False)
    print('Combined dataset produced\n')

def classify_task_base_classifier_multisource():
    """Classify task using the base classifier on the multisource dataset"""
    import classification.BaseClassifier as base_classifier
    print('----------Base classification of multisource dataset----------\n')

    # Load the dataset and filter out the emotions that are not in the reference file
    df = pd.read_csv(c.FEATURES_PATH + "COMBINED" + '.csv')
    df = df[~df['emotion'].isin([2, 7, 8, 6])]

    # Load features and target
    features = df.drop(columns=['actor', 'emotion'])
    target = df['emotion']

    classifier = base_classifier.BaseClassifier(features, target, "COMBINED")
    print('SVM classifier')
    classifier.svm_classifier()
    print()

    print('Decision Tree classifier')
    classifier.decision_tree_classifier()
    print()

    print('Linear Discriminant Analysis classifier')
    classifier.lda_classifier()
    print()

    print('Base classification of multisource dataset completed\n')
    print()

def classify_task_loso_classifier_multisource():
    """Classify task using the LOSO (Leave One Subject Out) validation approach with multisource dataset"""
    import classification.LosoClassifier as loso_classifier
    print('----------LOSO classification of multisource dataset----------\n')

    # Load the dataset and filter out the emotions that are not in the reference file
    df = pd.read_csv(c.FEATURES_PATH + "COMBINED" + '.csv')
    df = df[~df['emotion'].isin([2, 7, 8, 6])]

    # Load features and target
    features = df.drop(columns=['emotion'])
    target = df['emotion']

    #define the weights for the samples
    n_emovo = len(df[df['actor'].str.contains('EMOVO')])
    n_ravdess = len(df[df['actor'].str.contains('RAVDESS')])
    total_samples = n_emovo + n_ravdess

    emovo_weights = [(total_samples / 2) / n_emovo] * n_emovo
    ravdess_weights = [(total_samples / 2) / n_ravdess] * n_ravdess

    sample_weights = emovo_weights + ravdess_weights

    classifier = loso_classifier.LosoClassifier(features, target, "COMBINED", sample_weights, optimization=True)
    # classifier = loso_classifier.LosoClassifier(features, target, "COMBINED")
    print('SVM classifier')
    classifier.svm_classifier()
    print()

    print('Decision Tree classifier')
    classifier.decision_tree_classifier()
    print()

    print('Linear Discriminant Analysis classifier')
    classifier.lda_classifier()
    print()

    print('LOSO classification of multisource dataset completed\n')
    print()