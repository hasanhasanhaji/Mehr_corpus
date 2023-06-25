import Preprocess.Mehr_Corpus_Preprocess
import Preprocess.Feature_Vector
from pandas import DataFrame

# region create feature vector for all documents Create full feature vector once and save it to csv

mehr_feature_vector = []
for i in range(1, 358):
    print(i)
    # region Import Files
    # Convert conll file to structured File
    mehr_structuredFile = Preprocess.Mehr_Corpus_Preprocess.mehr_conll_to_structured(i)

    # Import Np file
    mehr_np_file = Preprocess.Mehr_Corpus_Preprocess.mehr_np_file(i)

    # Import Ne file
    mehr_ne_file = Preprocess.Mehr_Corpus_Preprocess.mehr_ne_file(i)

    # Import preprocess file
    mehr_pre_process_file = Preprocess.Mehr_Corpus_Preprocess.mehr_pre_process_file(i)

    # Extract Sentences from structured file
    mehr_input_sentences = Preprocess.Mehr_Corpus_Preprocess.extract_sentences(mehr_structuredFile)

    # Extract Coref Chain from Structured file and save them in new structured file and Also extract pronouns from
    # structured file
    mehr_chain_file = Preprocess.Mehr_Corpus_Preprocess.mehr_chain_file(mehr_structuredFile)
    # endregion

    # region Create positive and negative samples
    # for only pronouns
    mehr_perfect_sample_creation = Preprocess.Mehr_Corpus_Preprocess.mehr_perfect_sample_creation(mehr_chain_file,
                                                                                                  mehr_np_file,
                                                                                                  mehr_ne_file)

    # endregion

    # region Create feature vector
    # Create feature vector
    mehr_perfect_feature_vector_temp = Preprocess.Feature_Vector.mention_pair_setting1_mehr(
        mehr_perfect_sample_creation,
        mehr_pre_process_file)

    # endregion
    # add current feature vector to complete feature vector
    for items in mehr_perfect_feature_vector_temp:
        mehr_feature_vector.append(items)
dataset = DataFrame.from_records(mehr_feature_vector)
dataset.to_csv('full_train_fv.csv', index=False, encoding="utf-8")

# endregion
