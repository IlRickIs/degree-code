from functions import *

if __name__ == '__main__':
    #Produce features dataframes
    produce_features_dataframes(c.DATASETS_PATHS, c.FEATURES_PATH)

    #Classify task using the base classifier
    classify_task_base_classifier()

    #classify task using the LOSO (Leave One Subject Out) validation approach
    classify_task_loso_classifier()

    #classify task using cross corpus training approach
    classify_task_cross_corpus_classifier('EMOVO', 'RAVDESS')
    classify_task_cross_corpus_classifier('RAVDESS', 'EMOVO')

    #make a multisource dataset
    produce_combined_dataset(c.DATASETS_PATHS)

    #classify task using the multisource dataset
    classify_task_base_classifier_multisource()

    #classify task using the LOSO (Leave One Subject Out) validation approach with multisource dataset
    classify_task_loso_classifier_multisource()
