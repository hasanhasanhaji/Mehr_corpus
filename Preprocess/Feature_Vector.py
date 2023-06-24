import Preprocess.Nlp_Preprocess
import numpy as np
from pandas import DataFrame
from scipy import spatial
import math

# region Average feature vector for sentence
def avg_feature_vector(sentence, model, num_features, index2word_set):
    words = sentence.split()
    feature_vec = np.zeros((num_features,), dtype='float32')
    n_words = 0
    for word in words:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if n_words > 0:
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec
# endregion

# region Mention-Pair feature vector
def mention_pair_setting1_mehr(mehr_sample_creation, mehr_pre_process_file):
    feature_vectors = []

    for samples in mehr_sample_creation:
        pr_token = samples[0]
        pr_role = mehr_pre_process_file[int(samples[1])][4]
        np_tokens = str(samples[2])
        sbj_agreement = []
        obj_agreement = []
        num_agreement = []
        feature_vector = []

        # region Features of pronouns
        feature_vector.append(pr_token)
        feature_vector.append(np_tokens)
        feature_vector.append(Preprocess.Nlp_Preprocess.is_personal_pronoun(pr_token))
        feature_vector.append(Preprocess.Nlp_Preprocess.is_demonstartive_pronoun(pr_token))
        feature_vector.append(Preprocess.Nlp_Preprocess.is_reflexive_pronoun(pr_token))
        feature_vector.append(Preprocess.Nlp_Preprocess.is_third_person(pr_token))
        feature_vector.append(Preprocess.Nlp_Preprocess.is_speech(pr_token))
        # region Calculate pos whole of Antecedent

        if int(samples[1]) > 0:
            np_pos = mehr_pre_process_file[int(samples[1]) - 1][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[1]) > 1:
            np_pos = mehr_pre_process_file[int(samples[1]) - 2][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[1]) > 2:
            np_pos = mehr_pre_process_file[int(samples[1]) - 3][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[1]) < len(mehr_pre_process_file) - 1:
            np_pos = mehr_pre_process_file[int(samples[1]) + 1][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[1]) < len(mehr_pre_process_file) - 2:
            np_pos = mehr_pre_process_file[int(samples[1]) + 2][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[1]) < len(mehr_pre_process_file) - 3:
            np_pos = mehr_pre_process_file[int(samples[1]) + 3][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos(np_pos))
        else:
            feature_vector.append("0")

        # endregion
        sbj_agreement.append(Preprocess.Nlp_Preprocess.is_subject(samples[1], samples[1], mehr_pre_process_file))
        feature_vector.append(sbj_agreement[0])
        obj_agreement.append(Preprocess.Nlp_Preprocess.is_object(samples[1], samples[1], mehr_pre_process_file))
        feature_vector.append(obj_agreement[0])
        num_agreement.append(Preprocess.Nlp_Preprocess.pronoun_number(pr_token))
        feature_vector.append(num_agreement[0])

        # endregion

        # region Features of antecedents
        feature_vector.append(Preprocess.Nlp_Preprocess.np_tokens_count(np_tokens))
        if Preprocess.Nlp_Preprocess.is_pronoun(np_tokens) == "T":
            feature_vector.append("1")
        else:
            feature_vector.append("0")
        feature_vector.append(Preprocess.Nlp_Preprocess.is_demonstrative(np_tokens))
        # region Calculate pos whole of Antecedent
        if int(samples[3]) > 0:
            np_pos = mehr_pre_process_file[int(samples[3]) - 1][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[3]) > 1:
            np_pos = mehr_pre_process_file[int(samples[3]) - 2][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[3]) > 2:
            np_pos = mehr_pre_process_file[int(samples[3]) - 3][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[4]) < len(mehr_pre_process_file) - 1:
            np_pos = mehr_pre_process_file[int(samples[4]) + 1][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[4]) < len(mehr_pre_process_file) - 2:
            np_pos = mehr_pre_process_file[int(samples[4]) + 2][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[4]) < len(mehr_pre_process_file) - 3:
            np_pos = mehr_pre_process_file[int(samples[4]) + 3][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos(np_pos))
        else:
            feature_vector.append("0")

        # endregion
        num_agreement.append(
            Preprocess.Nlp_Preprocess.calculate_number(np_tokens, samples[3], samples[4], mehr_pre_process_file))
        feature_vector.append(num_agreement[1])
        sbj_agreement.append(Preprocess.Nlp_Preprocess.is_subject(samples[3], samples[4], mehr_pre_process_file))
        feature_vector.append(sbj_agreement[1])
        obj_agreement.append(Preprocess.Nlp_Preprocess.is_object(samples[3], samples[4], mehr_pre_process_file))
        feature_vector.append(obj_agreement[1])
        # endregion

        # region relational features
        sentence_distance = Preprocess.Nlp_Preprocess.sentence_distance(samples, mehr_pre_process_file)
        feature_vector.append(sentence_distance)
        feature_vector.append(Preprocess.Nlp_Preprocess.token_distance(samples))
        feature_vector.append(Preprocess.Nlp_Preprocess.number_agreement(num_agreement[0], num_agreement[1]))
        feature_vector.append(Preprocess.Nlp_Preprocess.subject_agreement(sbj_agreement[0], sbj_agreement[1]))
        feature_vector.append(Preprocess.Nlp_Preprocess.object_agreement(obj_agreement[0], obj_agreement[1]))
        feature_vector.append(Preprocess.Nlp_Preprocess.string_match(pr_token, np_tokens))
        feature_vector.append(Preprocess.Nlp_Preprocess.three_distance(sentence_distance))
        feature_vector.append(Preprocess.Nlp_Preprocess.same_distance(sentence_distance))

        # endregion

        # region class label
        if len(samples) == 6:
            if samples[5] == "p":
                feature_vector.append(1)
            else:
                feature_vector.append(0)

        # endregion

        # region information
        # feature_vector.append(pr_token)  # This feature adds only for observation
        # feature_vector.append(samples[1])  # This feature adds only for observation pronoun document ID
        # feature_vector.append(np_tokens.lstrip())  # This feature adds only for observation antecedent token
        # feature_vector.append(samples[3])  # This feature adds only for observation antecedent token start id
        # feature_vector.append(samples[4])  # This feature adds only for observation antecedent token end
        # endregion

        feature_vectors.append(feature_vector)

    return feature_vectors

def mention_pair_setting1_rcdat(rcdat_sample_creation, rcdat_pre_process_file):
    feature_vectors = []

    for samples in rcdat_sample_creation:
        pr_token = samples[0]
        pr_role = rcdat_pre_process_file[int(samples[1])][4]
        np_tokens = str(samples[2])
        sbj_agreement = []
        obj_agreement = []
        num_agreement = []
        feature_vector = []

        # region Features of pronouns
        feature_vector.append(pr_token)
        feature_vector.append(np_tokens)
        feature_vector.append(Preprocess.Nlp_Preprocess.is_personal_pronoun(pr_token))
        feature_vector.append(Preprocess.Nlp_Preprocess.is_demonstartive_pronoun(pr_token))
        feature_vector.append(Preprocess.Nlp_Preprocess.is_reflexive_pronoun(pr_token))
        feature_vector.append(Preprocess.Nlp_Preprocess.is_third_person(pr_token))
        feature_vector.append(Preprocess.Nlp_Preprocess.is_speech(pr_token))
        # region Calculate pos whole of Antecedent

        if int(samples[1]) > 0:
            np_pos = rcdat_pre_process_file[int(samples[1]) - 1][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos_rcdat(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[1]) > 1:
            np_pos = rcdat_pre_process_file[int(samples[1]) - 2][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos_rcdat(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[1]) > 2:
            np_pos = rcdat_pre_process_file[int(samples[1]) - 3][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos_rcdat(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[1]) < len(rcdat_pre_process_file) - 1:
            np_pos = rcdat_pre_process_file[int(samples[1]) + 1][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos_rcdat(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[1]) < len(rcdat_pre_process_file) - 2:
            np_pos = rcdat_pre_process_file[int(samples[1]) + 2][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos_rcdat(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[1]) < len(rcdat_pre_process_file) - 3:
            np_pos = rcdat_pre_process_file[int(samples[1]) + 3][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos_rcdat(np_pos))
        else:
            feature_vector.append("0")

        # endregion
        sbj_agreement.append(Preprocess.Nlp_Preprocess.is_subject_rcdat(samples[1], samples[1], rcdat_pre_process_file))
        feature_vector.append(sbj_agreement[0])
        obj_agreement.append(Preprocess.Nlp_Preprocess.is_object_rcdat(samples[1], samples[1], rcdat_pre_process_file))
        feature_vector.append(obj_agreement[0])
        num_agreement.append(Preprocess.Nlp_Preprocess.pronoun_number(pr_token))
        feature_vector.append(num_agreement[0])

        # endregion

        # region Features of antecedents
        feature_vector.append(Preprocess.Nlp_Preprocess.np_tokens_count(np_tokens))
        if Preprocess.Nlp_Preprocess.is_pronoun(np_tokens) == "T":
            feature_vector.append("1")
        else:
            feature_vector.append("0")
        feature_vector.append(Preprocess.Nlp_Preprocess.is_demonstrative(np_tokens))
        # region Calculate pos whole of Antecedent
        if int(samples[3]) > 0:
            np_pos = rcdat_pre_process_file[int(samples[3]) - 1][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos_rcdat(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[3]) > 1:
            np_pos = rcdat_pre_process_file[int(samples[3]) - 2][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos_rcdat(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[3]) > 2:
            np_pos = rcdat_pre_process_file[int(samples[3]) - 3][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos_rcdat(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[4]) < len(rcdat_pre_process_file) - 1:
            np_pos = rcdat_pre_process_file[int(samples[4]) + 1][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos_rcdat(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[4]) < len(rcdat_pre_process_file) - 2:
            np_pos = rcdat_pre_process_file[int(samples[4]) + 2][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos_rcdat(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[4]) < len(rcdat_pre_process_file) - 3:
            np_pos = rcdat_pre_process_file[int(samples[4]) + 3][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos_rcdat(np_pos))
        else:
            feature_vector.append("0")

        # endregion
        num_agreement.append(
            Preprocess.Nlp_Preprocess.calculate_number_rcdat(np_tokens, samples[3], samples[4], rcdat_pre_process_file))
        feature_vector.append(num_agreement[1])

        
        sbj_agreement.append(Preprocess.Nlp_Preprocess.is_subject_rcdat(samples[3], samples[4], rcdat_pre_process_file))
        feature_vector.append(sbj_agreement[1])
        obj_agreement.append(Preprocess.Nlp_Preprocess.is_object_rcdat(samples[3], samples[4], rcdat_pre_process_file))
        feature_vector.append(obj_agreement[1])
        # endregion

        # region relational features
        sentence_distance = Preprocess.Nlp_Preprocess.sentence_distance(samples, rcdat_pre_process_file)
        feature_vector.append(sentence_distance)
        feature_vector.append(Preprocess.Nlp_Preprocess.token_distance(samples))
        feature_vector.append(Preprocess.Nlp_Preprocess.number_agreement(num_agreement[0], num_agreement[1]))
        feature_vector.append(Preprocess.Nlp_Preprocess.subject_agreement(sbj_agreement[0], sbj_agreement[1]))
        feature_vector.append(Preprocess.Nlp_Preprocess.object_agreement(obj_agreement[0], obj_agreement[1]))
        feature_vector.append(Preprocess.Nlp_Preprocess.string_match(pr_token, np_tokens))
        feature_vector.append(Preprocess.Nlp_Preprocess.three_distance(sentence_distance))
        feature_vector.append(Preprocess.Nlp_Preprocess.same_distance(sentence_distance))


        # endregion

        # region class label
        if len(samples) == 6:
            if samples[5] == "p":
                feature_vector.append(1)
            else:
                feature_vector.append(0)

        # endregion

        # region information
        # feature_vector.append(pr_token)  # This feature adds only for observation
        # feature_vector.append(samples[1])  # This feature adds only for observation pronoun document ID
        # feature_vector.append(np_tokens.lstrip())  # This feature adds only for observation antecedent token
        # feature_vector.append(samples[3])  # This feature adds only for observation antecedent token start id
        # feature_vector.append(samples[4])  # This feature adds only for observation antecedent token end
        # endregion

        feature_vectors.append(feature_vector)

    return feature_vectors
# endregion

# region Hybrid feature vector

def hybrid_setting1_mehr(mehr_sample_creation, mehr_pre_process_file, mehr_chain_file, mehr_ne_file, animacy_file,
                    persian_names, model, mehr_input_sentences):

    index2word_set = model.index_to_key
    feature_vectors = []

    for samples in mehr_sample_creation:
        pr_token = samples[0]
        pr_role = mehr_pre_process_file[int(samples[1])][4]
        np_tokens = str(samples[2])
        sbj_agreement = []
        obj_agreement = []
        num_agreement = []
        feature_vector = []

        # pr_chain = [i for i in mehr_chain_file if i[6] == samples[1]][0][5]
        np_chain_num = [i for i in mehr_chain_file if i[6] == samples[3]]
        if len(np_chain_num) != 0:
            np_chain_num = np_chain_num[0][5]
            np_chain = np.array([i for i in mehr_chain_file if i[5] == np_chain_num and i[0] != "T"])



        # information
        feature_vector.append(pr_token)
        feature_vector.append(np_tokens)
        # region Pronoun features
        feature_vector.append(Preprocess.Nlp_Preprocess.is_personal_pronoun(pr_token))
        feature_vector.append(Preprocess.Nlp_Preprocess.is_demonstartive_pronoun(pr_token))
        feature_vector.append(Preprocess.Nlp_Preprocess.is_reflexive_pronoun(pr_token))
        feature_vector.append(Preprocess.Nlp_Preprocess.is_third_person(pr_token))
        feature_vector.append(Preprocess.Nlp_Preprocess.is_speech(pr_token))
        # region Calculate pos whole of Antecedent

        if int(samples[1]) > 0:
            np_pos = mehr_pre_process_file[int(samples[1]) - 1][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[1]) > 1:
            np_pos = mehr_pre_process_file[int(samples[1]) - 2][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[1]) > 2:
            np_pos = mehr_pre_process_file[int(samples[1]) - 3][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[1]) < len(mehr_pre_process_file) - 1:
            np_pos = mehr_pre_process_file[int(samples[1]) + 1][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[1]) < len(mehr_pre_process_file) - 2:
            np_pos = mehr_pre_process_file[int(samples[1]) + 2][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[1]) < len(mehr_pre_process_file) - 3:
            np_pos = mehr_pre_process_file[int(samples[1]) + 3][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos(np_pos))
        else:
            feature_vector.append("0")

        # endregion
        sbj_agreement.append(Preprocess.Nlp_Preprocess.is_subject(samples[1], samples[1], mehr_pre_process_file))
        feature_vector.append(sbj_agreement[0])
        obj_agreement.append(Preprocess.Nlp_Preprocess.is_object(samples[1], samples[1], mehr_pre_process_file))
        feature_vector.append(obj_agreement[0])
        num_agreement.append(Preprocess.Nlp_Preprocess.pronoun_number(pr_token))
        feature_vector.append(num_agreement[0])

        anaphor_animacy = Preprocess.Nlp_Preprocess.pronoun_animacy(samples[0])
        feature_vector.append(anaphor_animacy)

        anaphora_person = Preprocess.Nlp_Preprocess.person_num(samples[0])
        feature_vector.append(anaphora_person)


        # endregion

        # region Features of antecedents
        feature_vector.append(Preprocess.Nlp_Preprocess.np_tokens_count(np_tokens))
        if Preprocess.Nlp_Preprocess.is_pronoun(np_tokens) == "T":
            feature_vector.append("1")
        else:
            feature_vector.append("0")
        feature_vector.append(Preprocess.Nlp_Preprocess.is_demonstrative(np_tokens))
        # region Calculate pos whole of Antecedent
        if int(samples[3]) > 0:
            np_pos = mehr_pre_process_file[int(samples[3]) - 1][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[3]) > 1:
            np_pos = mehr_pre_process_file[int(samples[3]) - 2][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[3]) > 2:
            np_pos = mehr_pre_process_file[int(samples[3]) - 3][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[4]) < len(mehr_pre_process_file) - 1:
            np_pos = mehr_pre_process_file[int(samples[4]) + 1][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[4]) < len(mehr_pre_process_file) - 2:
            np_pos = mehr_pre_process_file[int(samples[4]) + 2][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[4]) < len(mehr_pre_process_file) - 3:
            np_pos = mehr_pre_process_file[int(samples[4]) + 3][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos(np_pos))
        else:
            feature_vector.append("0")

        # endregion
        num_agreement.append(
            Preprocess.Nlp_Preprocess.calculate_number(np_tokens, samples[3], samples[4], mehr_pre_process_file))
        feature_vector.append(num_agreement[1])
        sbj_agreement.append(Preprocess.Nlp_Preprocess.is_subject(samples[3], samples[4], mehr_pre_process_file))
        feature_vector.append(sbj_agreement[1])
        obj_agreement.append(Preprocess.Nlp_Preprocess.is_object(samples[3], samples[4], mehr_pre_process_file))
        feature_vector.append(obj_agreement[1])

        # E How many mentions are currently in the (antecedent / anaphor)’s cluster??
        if len(np_chain_num) == 0:
            feature_vector.append(1)
        else:
            feature_vector.append(len(np_chain))


        # M the antecedent is Reflexive pronoun?
        feature_vector.append(Preprocess.Nlp_Preprocess.is_reflexive_pronoun(samples[2]))

        # M  (antecedent / anaphor) mention type?  (0 = pronoun 1= proper 2 = common)
        feature_vector.append(Preprocess.Nlp_Preprocess.mention_type(samples[3], samples[4], samples[2],
                                                                          mehr_ne_file))

        # E The sentence # that includes the first mention in the (antecedent / anaphor ) cluster
        if len(np_chain_num) == 0 or len(np_chain) == 0:
            np_sentence = int([i for i in mehr_pre_process_file if i[0] == samples[3]][0][1])
            feature_vector.append( np_sentence)
        else:
            feature_vector.append(int(np_chain[0][1]))

        # Animacy
        antecedent_animacy = Preprocess.Nlp_Preprocess.animacy_detection(samples[2], samples[3], samples[4],
                                                                         mehr_pre_process_file,
                                                                         mehr_ne_file,
                                                                         animacy_file,
                                                                         persian_names)
        feature_vector.append(antecedent_animacy)

        # M antecedent person
        ante_person = Preprocess.Nlp_Preprocess.person_num(samples[2])
        feature_vector.append( ante_person)

        # M named_entity_type of antecedent
        en_type = Preprocess.Nlp_Preprocess.entity_Type(samples[3], samples[4], mehr_ne_file)
        feature_vector.append(en_type)


        # E the number of antecedent cluster ( First mention of cluster)
        if len(np_chain_num) == 0 or len(np_chain) == 0:
            cluster_num = Preprocess.Nlp_Preprocess.calculate_number(samples[2], samples[3], samples[4],
                                                                     mehr_pre_process_file)
            feature_vector.append( cluster_num)
        else:
            cluster_num = Preprocess.Nlp_Preprocess.calculate_number(np_chain[0][4], np_chain[0][6], np_chain[0][7],
                                                                     mehr_pre_process_file)
            feature_vector.append(cluster_num)




        # E antecedent  first cluster animcay
        if len(np_chain_num) == 0 or len(np_chain) == 0:
            cluster_animacy = Preprocess.Nlp_Preprocess.animacy_detection(samples[2], samples[3], samples[4],
                                                                          mehr_pre_process_file,
                                                                          mehr_ne_file,
                                                                          animacy_file,
                                                                          persian_names)
            feature_vector.append(cluster_animacy)
        else:
            cluster_animacy = Preprocess.Nlp_Preprocess.animacy_detection(np_chain[0][4], np_chain[0][6],
                                                                          np_chain[0][7],
                                                                          mehr_pre_process_file,
                                                                          mehr_ne_file,
                                                                          animacy_file,
                                                                          persian_names)
            feature_vector.append(cluster_animacy)



        # endregion

        # region relational features
        sentence_distance = Preprocess.Nlp_Preprocess.sentence_distance(samples, mehr_pre_process_file)
        feature_vector.append(sentence_distance)
        feature_vector.append(Preprocess.Nlp_Preprocess.token_distance(samples))
        feature_vector.append(Preprocess.Nlp_Preprocess.number_agreement(num_agreement[0], num_agreement[1]))
        feature_vector.append(Preprocess.Nlp_Preprocess.subject_agreement(sbj_agreement[0], sbj_agreement[1]))
        feature_vector.append(Preprocess.Nlp_Preprocess.object_agreement(obj_agreement[0], obj_agreement[1]))
        feature_vector.append(Preprocess.Nlp_Preprocess.string_match(pr_token, np_tokens))
        feature_vector.append(Preprocess.Nlp_Preprocess.three_distance(sentence_distance))
        feature_vector.append(Preprocess.Nlp_Preprocess.same_distance(sentence_distance))

        # E Minimum sentence distance between any two mentions from each cluster.
        if len(np_chain_num) == 0 or len(np_chain) == 0:
            feature_vector.append( sentence_distance)
        else:
            pr_sentence = int([i for i in mehr_pre_process_file if i[0] == samples[1]][0][1])
            min_distance = min(abs(np.asarray(np_chain[:, 1], dtype=np.int) - pr_sentence))
            feature_vector.append( int(min_distance))

        # M Is length of antecedent  longer than the length of the anaphor mention ?
        if len(samples[2].split()) > len(samples[0].split()):
            feature_vector.append( 1)
        else:
            feature_vector.append( 0)
        # M animacy agreement
        feature_vector.append(Preprocess.Nlp_Preprocess.animacy_agreement(anaphor_animacy, antecedent_animacy))


        # M person agreement
        feature_vector.append(Preprocess.Nlp_Preprocess.person_agreement(ante_person, anaphora_person))





        # M If anaphor is object and antecedent is subject and both are in same sentence
        is_anphor_object = Preprocess.Nlp_Preprocess.is_object(samples[1], samples[1], mehr_pre_process_file)
        is_antecedent_subject = Preprocess.Nlp_Preprocess.is_subject(samples[3], samples[4], mehr_pre_process_file)

        if is_anphor_object == "1" and is_antecedent_subject == "1" and sentence_distance == 0:
            feature_vector.append( "1")
        else:
            feature_vector.append( "0")


        # M The anaphor mention is a reflexive pronoun and the antecedent is its subject
        if Preprocess.Nlp_Preprocess.is_subject(samples[3], samples[4], mehr_pre_process_file) == "1" \
            and Preprocess.Nlp_Preprocess.is_reflexive_pronoun(pr_token) == "1" and sentence_distance == 0:
            feature_vector.append( "1")
        else:
            feature_vector.append( "0")


        # endregion

        # region word vector features
        # M The euclidean distance between the tow headwords in thier corresponding word vectors
        qh = [i for i in mehr_pre_process_file if int(samples[3]) <= int(i[0]) <= int(samples[4]) and (i[10] == "N" or i[10] == "Ne")]
        if len(qh) > 0:
            m2_head = qh[0][7]
        else:
            m2_head = samples[2].split(' ')[0]

        h1_afv = avg_feature_vector(samples[0], model=model, num_features=50, index2word_set=index2word_set)
        h2_afv = avg_feature_vector(m2_head, model=model, num_features=50, index2word_set=index2word_set)
        dist = float("{:.2f}".format(1 - spatial.distance.cosine(h1_afv, h2_afv)))
        if math.isnan(dist):
            dist = 0.0
        feature_vector.append( dist)


        # M the distance between the two noun phrases average
        m1_afv = avg_feature_vector(samples[0], model=model, num_features=50, index2word_set=index2word_set)
        m2_afv = avg_feature_vector(samples[2], model=model, num_features=50, index2word_set=index2word_set)
        dist = float("{:.2f}".format(1 - spatial.distance.cosine(m1_afv, m2_afv)))
        if math.isnan(dist):
            dist = 0.0
        feature_vector.append( dist)

        #   M calculate similarity between candidate and the sentence where anaphora is in.
        qs = [i for i in mehr_pre_process_file if int(i[0]) == int(samples[1])][0][1]
        anaphora_sentence = mehr_input_sentences[int(qs)]
        m2_afv = avg_feature_vector(samples[2], model=model, num_features=50, index2word_set=index2word_set)
        s2_afv = avg_feature_vector(anaphora_sentence, model=model, num_features=50, index2word_set=index2word_set)
        dist = float("{:.2f}".format(1 - spatial.distance.cosine(m2_afv, s2_afv)))
        if math.isnan(dist):
            dist = 0.0
        feature_vector.append( dist)

        # first and second token
        m3_afv = avg_feature_vector(samples[0], model=model, num_features=50, index2word_set=index2word_set).tolist()
        feature_vector.extend(m3_afv)

        m4_afv = avg_feature_vector(samples[2], model=model, num_features=50, index2word_set=index2word_set).tolist()
        feature_vector.extend(m4_afv)

        # endregion

        # region class label
        if len(samples) == 6:
            if samples[5] == "p":
                feature_vector.append( 1)
            else:
                feature_vector.append( 0)

        # endregion
        feature_vectors.append(feature_vector)
    return feature_vectors

def hybrid_setting2_mehr(mehr_sample_creation, mehr_pre_process_file, mehr_mentions_dict, mehr_ne_file, animacy_file,
                    persian_names, model, mehr_input_sentences):

    index2word_set = model.index_to_key
    feature_vectors = []

    for samples in mehr_sample_creation:
        pr_token = samples[0]
        pr_role = mehr_pre_process_file[int(samples[1])][4]
        np_tokens = str(samples[2])
        sbj_agreement = []
        obj_agreement = []
        num_agreement = []
        feature_vector = []

        # pr_chain = [i for i in mehr_chain_file if i[6] == samples[1]][0][5]
        np_chain_number = [v for k, v in mehr_mentions_dict.items() if v[0][6] == samples[3] and v[2] != -1]

        if len(np_chain_number) != 0:
            np_chain_num = np_chain_number[0][2]
            np_chain = np.array([v[0] for k, v in mehr_mentions_dict.items() if v[2] == np_chain_num])




        # information
        feature_vector.append(pr_token)
        feature_vector.append(np_tokens)
        # region Pronoun features
        feature_vector.append(Preprocess.Nlp_Preprocess.is_personal_pronoun(pr_token))
        feature_vector.append(Preprocess.Nlp_Preprocess.is_demonstartive_pronoun(pr_token))
        feature_vector.append(Preprocess.Nlp_Preprocess.is_reflexive_pronoun(pr_token))
        feature_vector.append(Preprocess.Nlp_Preprocess.is_third_person(pr_token))
        feature_vector.append(Preprocess.Nlp_Preprocess.is_speech(pr_token))
        # region Calculate pos whole of Antecedent

        if int(samples[1]) > 0:
            np_pos = mehr_pre_process_file[int(samples[1]) - 1][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[1]) > 1:
            np_pos = mehr_pre_process_file[int(samples[1]) - 2][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[1]) > 2:
            np_pos = mehr_pre_process_file[int(samples[1]) - 3][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[1]) < len(mehr_pre_process_file) - 1:
            np_pos = mehr_pre_process_file[int(samples[1]) + 1][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[1]) < len(mehr_pre_process_file) - 2:
            np_pos = mehr_pre_process_file[int(samples[1]) + 2][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[1]) < len(mehr_pre_process_file) - 3:
            np_pos = mehr_pre_process_file[int(samples[1]) + 3][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos(np_pos))
        else:
            feature_vector.append("0")

        # endregion
        sbj_agreement.append(Preprocess.Nlp_Preprocess.is_subject(samples[1], samples[1], mehr_pre_process_file))
        feature_vector.append(sbj_agreement[0])
        obj_agreement.append(Preprocess.Nlp_Preprocess.is_object(samples[1], samples[1], mehr_pre_process_file))
        feature_vector.append(obj_agreement[0])
        num_agreement.append(Preprocess.Nlp_Preprocess.pronoun_number(pr_token))
        feature_vector.append(num_agreement[0])

        anaphor_animacy = Preprocess.Nlp_Preprocess.pronoun_animacy(samples[0])
        feature_vector.append(anaphor_animacy)

        anaphora_person = Preprocess.Nlp_Preprocess.person_num(samples[0])
        feature_vector.append(anaphora_person)


        # endregion

        # region Features of antecedents
        feature_vector.append(Preprocess.Nlp_Preprocess.np_tokens_count(np_tokens))
        if Preprocess.Nlp_Preprocess.is_pronoun(np_tokens) == "T":
            feature_vector.append("1")
        else:
            feature_vector.append("0")
        feature_vector.append(Preprocess.Nlp_Preprocess.is_demonstrative(np_tokens))
        # region Calculate pos whole of Antecedent
        if int(samples[3]) > 0:
            np_pos = mehr_pre_process_file[int(samples[3]) - 1][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[3]) > 1:
            np_pos = mehr_pre_process_file[int(samples[3]) - 2][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[3]) > 2:
            np_pos = mehr_pre_process_file[int(samples[3]) - 3][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[4]) < len(mehr_pre_process_file) - 1:
            np_pos = mehr_pre_process_file[int(samples[4]) + 1][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[4]) < len(mehr_pre_process_file) - 2:
            np_pos = mehr_pre_process_file[int(samples[4]) + 2][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[4]) < len(mehr_pre_process_file) - 3:
            np_pos = mehr_pre_process_file[int(samples[4]) + 3][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos(np_pos))
        else:
            feature_vector.append("0")

        # endregion
        num_agreement.append(
            Preprocess.Nlp_Preprocess.calculate_number(np_tokens, samples[3], samples[4], mehr_pre_process_file))
        feature_vector.append(num_agreement[1])
        sbj_agreement.append(Preprocess.Nlp_Preprocess.is_subject(samples[3], samples[4], mehr_pre_process_file))
        feature_vector.append(sbj_agreement[1])
        obj_agreement.append(Preprocess.Nlp_Preprocess.is_object(samples[3], samples[4], mehr_pre_process_file))
        feature_vector.append(obj_agreement[1])

        # E How many mentions are currently in the (antecedent / anaphor)’s cluster??
        if len(np_chain_number) == 0:
            feature_vector.append(1)
        else:
            feature_vector.append(len(np_chain))


        # M the antecedent is Reflexive pronoun?
        feature_vector.append(Preprocess.Nlp_Preprocess.is_reflexive_pronoun(samples[2]))

        # M  (antecedent / anaphor) mention type?  (0 = pronoun 1= proper 2 = common)
        feature_vector.append(Preprocess.Nlp_Preprocess.mention_type(samples[3], samples[4], samples[2],
                                                                          mehr_ne_file))

        # E The sentence # that includes the first mention in the (antecedent / anaphor ) cluster
        if len(np_chain_number) == 0 or len(np_chain) == 0:
            np_sentence = int([i for i in mehr_pre_process_file if i[0] == samples[3]][0][1])
            feature_vector.append( np_sentence)
        else:
            feature_vector.append(int(np_chain[0][1]))

        # Animacy
        antecedent_animacy = Preprocess.Nlp_Preprocess.animacy_detection(samples[2], samples[3], samples[4],
                                                                         mehr_pre_process_file,
                                                                         mehr_ne_file,
                                                                         animacy_file,
                                                                         persian_names)
        feature_vector.append(antecedent_animacy)

        # M antecedent person
        ante_person = Preprocess.Nlp_Preprocess.person_num(samples[2])
        feature_vector.append( ante_person)

        # M named_entity_type of antecedent
        en_type = Preprocess.Nlp_Preprocess.entity_Type(samples[3], samples[4], mehr_ne_file)
        feature_vector.append(en_type)


        # E the number of antecedent cluster ( First mention of cluster)
        if len(np_chain_number) == 0 or len(np_chain) == 0:
            cluster_num = Preprocess.Nlp_Preprocess.calculate_number(samples[2], samples[3], samples[4],
                                                                     mehr_pre_process_file)
            feature_vector.append( cluster_num)
        else:
            cluster_num = Preprocess.Nlp_Preprocess.calculate_number(np_chain[0][4], np_chain[0][6], np_chain[0][7],
                                                                     mehr_pre_process_file)
            feature_vector.append(cluster_num)


        # E antecedent  first cluster animcay
        if len(np_chain_number) == 0 or len(np_chain) == 0:
            cluster_animacy = Preprocess.Nlp_Preprocess.animacy_detection(samples[2], samples[3], samples[4],
                                                                          mehr_pre_process_file,
                                                                          mehr_ne_file,
                                                                          animacy_file,
                                                                          persian_names)
            feature_vector.append(cluster_animacy)
        else:
            cluster_animacy = Preprocess.Nlp_Preprocess.animacy_detection(np_chain[0][4], np_chain[0][6],
                                                                          np_chain[0][7],
                                                                          mehr_pre_process_file,
                                                                          mehr_ne_file,
                                                                          animacy_file,
                                                                          persian_names)
            feature_vector.append(cluster_animacy)



        # endregion

        # region relational features
        sentence_distance = Preprocess.Nlp_Preprocess.sentence_distance(samples, mehr_pre_process_file)
        feature_vector.append(sentence_distance)
        feature_vector.append(Preprocess.Nlp_Preprocess.token_distance(samples))
        feature_vector.append(Preprocess.Nlp_Preprocess.number_agreement(num_agreement[0], num_agreement[1]))
        feature_vector.append(Preprocess.Nlp_Preprocess.subject_agreement(sbj_agreement[0], sbj_agreement[1]))
        feature_vector.append(Preprocess.Nlp_Preprocess.object_agreement(obj_agreement[0], obj_agreement[1]))
        feature_vector.append(Preprocess.Nlp_Preprocess.string_match(pr_token, np_tokens))
        feature_vector.append(Preprocess.Nlp_Preprocess.three_distance(sentence_distance))
        feature_vector.append(Preprocess.Nlp_Preprocess.same_distance(sentence_distance))

        # E Minimum sentence distance between any two mentions from each cluster.
        if len(np_chain_number) == 0 or len(np_chain) == 0:
            feature_vector.append( sentence_distance)
        else:
            pr_sentence = int([i for i in mehr_pre_process_file if i[0] == samples[1]][0][1])
            min_distance = min(abs(np.asarray(np_chain[:, 1], dtype=np.int) - pr_sentence))
            feature_vector.append( int(min_distance))

        # M Is length of antecedent  longer than the length of the anaphor mention ?
        if len(samples[2].split()) > len(samples[0].split()):
            feature_vector.append( 1)
        else:
            feature_vector.append( 0)
        # M animacy agreement
        feature_vector.append(Preprocess.Nlp_Preprocess.animacy_agreement(anaphor_animacy, antecedent_animacy))


        # M person agreement
        feature_vector.append(Preprocess.Nlp_Preprocess.person_agreement(ante_person, anaphora_person))



        # M If anaphor is object and antecedent is subject and both are in same sentence
        is_anphor_object = Preprocess.Nlp_Preprocess.is_object(samples[1], samples[1], mehr_pre_process_file)
        is_antecedent_subject = Preprocess.Nlp_Preprocess.is_subject(samples[3], samples[4], mehr_pre_process_file)

        if is_anphor_object == "1" and is_antecedent_subject == "1" and sentence_distance == 0:
            feature_vector.append( "1")
        else:
            feature_vector.append( "0")


        # M The anaphor mention is a reflexive pronoun and the antecedent is its subject
        if Preprocess.Nlp_Preprocess.is_subject(samples[3], samples[4], mehr_pre_process_file) == "1" \
            and Preprocess.Nlp_Preprocess.is_reflexive_pronoun(pr_token) == "1" and sentence_distance == 0:
            feature_vector.append( "1")
        else:
            feature_vector.append( "0")


        # endregion

        # region word vector features
        # M The euclidean distance between the tow headwords in thier corresponding word vectors
        qh = [i for i in mehr_pre_process_file if int(samples[3]) <= int(i[0]) <= int(samples[4]) and (i[10] == "N" or i[10] == "Ne")]
        if len(qh) > 0:
            m2_head = qh[0][7]
        else:
            m2_head = samples[2].split(' ')[0]

        h1_afv = avg_feature_vector(samples[0], model=model, num_features=50, index2word_set=index2word_set)
        h2_afv = avg_feature_vector(m2_head, model=model, num_features=50, index2word_set=index2word_set)
        dist = float("{:.2f}".format(1 - spatial.distance.cosine(h1_afv, h2_afv)))
        if math.isnan(dist):
            dist = 0.0
        feature_vector.append( dist)


        # M the distance between the two noun phrases average
        m1_afv = avg_feature_vector(samples[0], model=model, num_features=50, index2word_set=index2word_set)
        m2_afv = avg_feature_vector(samples[2], model=model, num_features=50, index2word_set=index2word_set)
        dist = float("{:.2f}".format(1 - spatial.distance.cosine(m1_afv, m2_afv)))
        if math.isnan(dist):
            dist = 0.0
        feature_vector.append( dist)

        #   M calculate similarity between candidate and the sentence where anaphora is in.
        qs = [i for i in mehr_pre_process_file if int(i[0]) == int(samples[1])][0][1]
        anaphora_sentence = mehr_input_sentences[int(qs)]
        m2_afv = avg_feature_vector(samples[2], model=model, num_features=50, index2word_set=index2word_set)
        s2_afv = avg_feature_vector(anaphora_sentence, model=model, num_features=50, index2word_set=index2word_set)
        dist = float("{:.2f}".format(1 - spatial.distance.cosine(m2_afv, s2_afv)))
        if math.isnan(dist):
            dist = 0.0
        feature_vector.append( dist)

        # first and second token
        #m3_afv = avg_feature_vector(samples[0], model=model, num_features=50, index2word_set=index2word_set).tolist()
        #feature_vector.extend(m3_afv)

        #m4_afv = avg_feature_vector(samples[2], model=model, num_features=50, index2word_set=index2word_set).tolist()
        #feature_vector.extend(m4_afv)

        # endregion

        # region class label
        if len(samples) == 6:
            if samples[5] == "p":
                feature_vector.append( 1)
            else:
                feature_vector.append( 0)

        # endregion
        feature_vectors.append(feature_vector)
    return feature_vectors

def hybrid_setting1_rcdat(rcdat_sample_creation, rcdat_pre_process_file, rcdat_chain_file, rcdat_ne_file, animacy_file,
                    persian_names, model, rcdat_input_sentences):

    index2word_set = model.index_to_key
    feature_vectors = []
    for samples in rcdat_sample_creation:
        pr_token = samples[0]
        pr_role = rcdat_pre_process_file[int(samples[1])][4]
        np_tokens = str(samples[2])
        sbj_agreement = []
        obj_agreement = []
        num_agreement = []
        feature_vector = []

        # pr_chain = [i for i in mehr_chain_file if i[6] == samples[1]][0][5]
        np_chain_num = [i for i in rcdat_chain_file if i[6] == samples[3]]
        if len(np_chain_num) != 0:
            np_chain_num = np_chain_num[0][5]
            np_chain = np.array([i for i in rcdat_chain_file if i[5] == np_chain_num and i[0] != "T"])



        # information
        feature_vector.append(pr_token)
        feature_vector.append(np_tokens)
        # region Pronoun features
        feature_vector.append(Preprocess.Nlp_Preprocess.is_personal_pronoun(pr_token))
        feature_vector.append(Preprocess.Nlp_Preprocess.is_demonstartive_pronoun(pr_token))
        feature_vector.append(Preprocess.Nlp_Preprocess.is_reflexive_pronoun(pr_token))
        feature_vector.append(Preprocess.Nlp_Preprocess.is_third_person(pr_token))
        feature_vector.append(Preprocess.Nlp_Preprocess.is_speech(pr_token))
        # region Calculate pos whole of Antecedent

        if int(samples[1]) > 0:
            np_pos = rcdat_pre_process_file[int(samples[1]) - 1][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos_rcdat(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[1]) > 1:
            np_pos = rcdat_pre_process_file[int(samples[1]) - 2][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos_rcdat(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[1]) > 2:
            np_pos = rcdat_pre_process_file[int(samples[1]) - 3][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos_rcdat(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[1]) < len(rcdat_pre_process_file) - 1:
            np_pos = rcdat_pre_process_file[int(samples[1]) + 1][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos_rcdat(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[1]) < len(rcdat_pre_process_file) - 2:
            np_pos = rcdat_pre_process_file[int(samples[1]) + 2][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos_rcdat(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[1]) < len(rcdat_pre_process_file) - 3:
            np_pos = rcdat_pre_process_file[int(samples[1]) + 3][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos_rcdat(np_pos))
        else:
            feature_vector.append("0")

        # endregion
        sbj_agreement.append(Preprocess.Nlp_Preprocess.is_subject_rcdat(samples[1], samples[1], rcdat_pre_process_file))
        feature_vector.append(sbj_agreement[0])
        obj_agreement.append(Preprocess.Nlp_Preprocess.is_object_rcdat(samples[1], samples[1], rcdat_pre_process_file))
        feature_vector.append(obj_agreement[0])
        num_agreement.append(Preprocess.Nlp_Preprocess.pronoun_number(pr_token))
        feature_vector.append(num_agreement[0])

        anaphor_animacy = Preprocess.Nlp_Preprocess.pronoun_animacy(samples[0])
        feature_vector.append(anaphor_animacy)

        anaphora_person = Preprocess.Nlp_Preprocess.person_num(samples[0])
        feature_vector.append(anaphora_person)


        # endregion

        # region Features of antecedents
        feature_vector.append(Preprocess.Nlp_Preprocess.np_tokens_count(np_tokens))
        if Preprocess.Nlp_Preprocess.is_pronoun(np_tokens) == "T":
            feature_vector.append("1")
        else:
            feature_vector.append("0")
        feature_vector.append(Preprocess.Nlp_Preprocess.is_demonstrative(np_tokens))
        # region Calculate pos whole of Antecedent
        if int(samples[3]) > 0:
            np_pos = rcdat_pre_process_file[int(samples[3]) - 1][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos_rcdat(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[3]) > 1:
            np_pos = rcdat_pre_process_file[int(samples[3]) - 2][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos_rcdat(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[3]) > 2:
            np_pos = rcdat_pre_process_file[int(samples[3]) - 3][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos_rcdat(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[4]) < len(rcdat_pre_process_file) - 1:
            np_pos = rcdat_pre_process_file[int(samples[4]) + 1][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos_rcdat(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[4]) < len(rcdat_pre_process_file) - 2:
            np_pos = rcdat_pre_process_file[int(samples[4]) + 2][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos_rcdat(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[4]) < len(rcdat_pre_process_file) - 3:
            np_pos = rcdat_pre_process_file[int(samples[4]) + 3][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos_rcdat(np_pos))
        else:
            feature_vector.append("0")

        # endregion
        num_agreement.append(
            Preprocess.Nlp_Preprocess.calculate_number_rcdat(np_tokens, samples[3], samples[4], rcdat_pre_process_file))
        feature_vector.append(num_agreement[1])
        sbj_agreement.append(Preprocess.Nlp_Preprocess.is_subject_rcdat(samples[3], samples[4], rcdat_pre_process_file))
        feature_vector.append(sbj_agreement[1])
        obj_agreement.append(Preprocess.Nlp_Preprocess.is_object_rcdat(samples[3], samples[4], rcdat_pre_process_file))
        feature_vector.append(obj_agreement[1])

        # E How many mentions are currently in the (antecedent / anaphor)’s cluster??
        if len(np_chain_num) == 0:
            feature_vector.append(1)
        else:
            feature_vector.append(len(np_chain))


        # M the antecedent is Reflexive pronoun?
        feature_vector.append(Preprocess.Nlp_Preprocess.is_reflexive_pronoun(samples[2]))

        # M  (antecedent / anaphor) mention type?  (0 = pronoun 1= proper 2 = common)
        feature_vector.append(Preprocess.Nlp_Preprocess.mention_type(samples[3], samples[4], samples[2],
                                                                          rcdat_ne_file))

        # E The sentence # that includes the first mention in the (antecedent / anaphor ) cluster
        if len(np_chain_num) == 0 or len(np_chain) == 0:
            np_sentence = int([i for i in rcdat_pre_process_file if i[0] == samples[3]][0][1])
            feature_vector.append( np_sentence)
        else:
            feature_vector.append(int(np_chain[0][1]))

        # Animacy
        antecedent_animacy = Preprocess.Nlp_Preprocess.animacy_detection_rcdat(samples[2], samples[3], samples[4],
                                                                         rcdat_pre_process_file,
                                                                         rcdat_ne_file,
                                                                         animacy_file,
                                                                         persian_names)
        feature_vector.append(antecedent_animacy)

        # M antecedent person
        ante_person = Preprocess.Nlp_Preprocess.person_num(samples[2])
        feature_vector.append( ante_person)

        # M named_entity_type of antecedent
        en_type = Preprocess.Nlp_Preprocess.entity_Type(samples[3], samples[4], rcdat_ne_file)
        feature_vector.append(en_type)


        # E the number of antecedent cluster ( First mention of cluster)
        if len(np_chain_num) == 0 or len(np_chain) == 0:
            cluster_num = Preprocess.Nlp_Preprocess.calculate_number(samples[2], samples[3], samples[4],
                                                                     rcdat_pre_process_file)
            feature_vector.append( cluster_num)
        else:
            cluster_num = Preprocess.Nlp_Preprocess.calculate_number(np_chain[0][4], np_chain[0][6], np_chain[0][7],
                                                                     rcdat_pre_process_file)
            feature_vector.append(cluster_num)




        # E antecedent  first cluster animcay
        if len(np_chain_num) == 0 or len(np_chain) == 0:
            cluster_animacy = Preprocess.Nlp_Preprocess.animacy_detection_rcdat(samples[2], samples[3], samples[4],
                                                                          rcdat_pre_process_file,
                                                                          rcdat_ne_file,
                                                                          animacy_file,
                                                                          persian_names)
            feature_vector.append(cluster_animacy)
        else:
            cluster_animacy = Preprocess.Nlp_Preprocess.animacy_detection_rcdat(np_chain[0][4], np_chain[0][6],
                                                                          np_chain[0][7],
                                                                          rcdat_pre_process_file,
                                                                          rcdat_ne_file,
                                                                          animacy_file,
                                                                          persian_names)
            feature_vector.append(cluster_animacy)



        # endregion

        # region relational features
        sentence_distance = Preprocess.Nlp_Preprocess.sentence_distance(samples, rcdat_pre_process_file)
        feature_vector.append(sentence_distance)
        feature_vector.append(Preprocess.Nlp_Preprocess.token_distance(samples))
        feature_vector.append(Preprocess.Nlp_Preprocess.number_agreement(num_agreement[0], num_agreement[1]))
        feature_vector.append(Preprocess.Nlp_Preprocess.subject_agreement(sbj_agreement[0], sbj_agreement[1]))
        feature_vector.append(Preprocess.Nlp_Preprocess.object_agreement(obj_agreement[0], obj_agreement[1]))
        feature_vector.append(Preprocess.Nlp_Preprocess.string_match(pr_token, np_tokens))
        feature_vector.append(Preprocess.Nlp_Preprocess.three_distance(sentence_distance))
        feature_vector.append(Preprocess.Nlp_Preprocess.same_distance(sentence_distance))

        # E Minimum sentence distance between any two mentions from each cluster.
        if len(np_chain_num) == 0 or len(np_chain) == 0:
            feature_vector.append( sentence_distance)
        else:
            pr_sentence = int([i for i in rcdat_pre_process_file if i[0] == samples[1]][0][1])
            min_distance = min(abs(np.asarray(np_chain[:, 1], dtype=np.int) - pr_sentence))
            feature_vector.append( int(min_distance))

        # M Is length of antecedent  longer than the length of the anaphor mention ?
        if len(samples[2].split()) > len(samples[0].split()):
            feature_vector.append( 1)
        else:
            feature_vector.append( 0)
        # M animacy agreement
        feature_vector.append(Preprocess.Nlp_Preprocess.animacy_agreement(anaphor_animacy, antecedent_animacy))


        # M person agreement
        feature_vector.append(Preprocess.Nlp_Preprocess.person_agreement(ante_person, anaphora_person))





        # M If anaphor is object and antecedent is subject and both are in same sentence
        is_anphor_object = Preprocess.Nlp_Preprocess.is_object_rcdat(samples[1], samples[1], rcdat_pre_process_file)
        is_antecedent_subject = Preprocess.Nlp_Preprocess.is_subject_rcdat(samples[3], samples[4], rcdat_pre_process_file)

        if is_anphor_object == "1" and is_antecedent_subject == "1" and sentence_distance == 0:
            feature_vector.append( "1")
        else:
            feature_vector.append( "0")


        # M The anaphor mention is a reflexive pronoun and the antecedent is its subject
        if Preprocess.Nlp_Preprocess.is_subject_rcdat(samples[3], samples[4], rcdat_pre_process_file) == "1" \
            and Preprocess.Nlp_Preprocess.is_reflexive_pronoun(pr_token) == "1" and sentence_distance == 0:
            feature_vector.append( "1")
        else:
            feature_vector.append( "0")


        # endregion

        # region word vector features
        # M The euclidean distance between the tow headwords in thier corresponding word vectors
        qh = [i for i in rcdat_pre_process_file if int(samples[3]) <= int(i[0]) <= int(samples[4]) and (i[10] == "NOUN" or i[10] == "PROPN")]
        if len(qh) > 0:
            m2_head = qh[0][7]
        else:
            m2_head = samples[2].split(' ')[0]

        h1_afv = avg_feature_vector(samples[0], model=model, num_features=50, index2word_set=index2word_set)
        h2_afv = avg_feature_vector(m2_head, model=model, num_features=50, index2word_set=index2word_set)
        dist = float("{:.2f}".format(1 - spatial.distance.cosine(h1_afv, h2_afv)))
        if math.isnan(dist):
            dist = 0.0
        feature_vector.append( dist)


        # M the distance between the two noun phrases average
        m1_afv = avg_feature_vector(samples[0], model=model, num_features=50, index2word_set=index2word_set)
        m2_afv = avg_feature_vector(samples[2], model=model, num_features=50, index2word_set=index2word_set)
        dist = float("{:.2f}".format(1 - spatial.distance.cosine(m1_afv, m2_afv)))
        if math.isnan(dist):
            dist = 0.0
        feature_vector.append( dist)

        #   M calculate similarity between candidate and the sentence where anaphora is in.
        qs = [i for i in rcdat_pre_process_file if int(i[0]) == int(samples[1])][0][1]
        anaphora_sentence = rcdat_input_sentences[int(qs)]
        m2_afv = avg_feature_vector(samples[2], model=model, num_features=50, index2word_set=index2word_set)
        s2_afv = avg_feature_vector(anaphora_sentence, model=model, num_features=50, index2word_set=index2word_set)
        dist = float("{:.2f}".format(1 - spatial.distance.cosine(m2_afv, s2_afv)))
        if math.isnan(dist):
            dist = 0.0
        feature_vector.append( dist)

        # first and second token
        m3_afv = avg_feature_vector(samples[0], model=model, num_features=50, index2word_set=index2word_set).tolist()
        feature_vector.extend(m3_afv)

        m4_afv = avg_feature_vector(samples[2], model=model, num_features=50, index2word_set=index2word_set).tolist()
        feature_vector.extend(m4_afv)

        # endregion

        # region class label
        if len(samples) == 6:
            if samples[5] == "p":
                feature_vector.append( 1)
            else:
                feature_vector.append( 0)

        # endregion
        feature_vectors.append(feature_vector)
    return feature_vectors

def hybrid_setting2_rcdat(rcdat_sample_creation, rcdat_pre_process_file, mehr_mentions_dict, rcdat_ne_file, animacy_file,
                    persian_names, model, rcdat_input_sentences):

    index2word_set = model.index_to_key
    feature_vectors = []

    for samples in rcdat_sample_creation:
        pr_token = samples[0]
        pr_role = rcdat_pre_process_file[int(samples[1])][4]
        np_tokens = str(samples[2])
        sbj_agreement = []
        obj_agreement = []
        num_agreement = []
        feature_vector = []

        # pr_chain = [i for i in mehr_chain_file if i[6] == samples[1]][0][5]
        np_chain_number = [v for k, v in mehr_mentions_dict.items() if v[0][6] == samples[3] and v[2] != -1]

        if len(np_chain_number) != 0:
            np_chain_num = np_chain_number[0][2]
            np_chain = np.array([v[0] for k, v in mehr_mentions_dict.items() if v[2] == np_chain_num])



        # information
        feature_vector.append(pr_token)
        feature_vector.append(np_tokens)
        # region Pronoun features
        feature_vector.append(Preprocess.Nlp_Preprocess.is_personal_pronoun(pr_token))
        feature_vector.append(Preprocess.Nlp_Preprocess.is_demonstartive_pronoun(pr_token))
        feature_vector.append(Preprocess.Nlp_Preprocess.is_reflexive_pronoun(pr_token))
        feature_vector.append(Preprocess.Nlp_Preprocess.is_third_person(pr_token))
        feature_vector.append(Preprocess.Nlp_Preprocess.is_speech(pr_token))
        # region Calculate pos whole of Antecedent

        if int(samples[1]) > 0:
            np_pos = rcdat_pre_process_file[int(samples[1]) - 1][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos_rcdat(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[1]) > 1:
            np_pos = rcdat_pre_process_file[int(samples[1]) - 2][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos_rcdat(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[1]) > 2:
            np_pos = rcdat_pre_process_file[int(samples[1]) - 3][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos_rcdat(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[1]) < len(rcdat_pre_process_file) - 1:
            np_pos = rcdat_pre_process_file[int(samples[1]) + 1][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos_rcdat(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[1]) < len(rcdat_pre_process_file) - 2:
            np_pos = rcdat_pre_process_file[int(samples[1]) + 2][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos_rcdat(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[1]) < len(rcdat_pre_process_file) - 3:
            np_pos = rcdat_pre_process_file[int(samples[1]) + 3][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos_rcdat(np_pos))
        else:
            feature_vector.append("0")

        # endregion
        sbj_agreement.append(Preprocess.Nlp_Preprocess.is_subject_rcdat(samples[1], samples[1], rcdat_pre_process_file))
        feature_vector.append(sbj_agreement[0])
        obj_agreement.append(Preprocess.Nlp_Preprocess.is_object_rcdat(samples[1], samples[1], rcdat_pre_process_file))
        feature_vector.append(obj_agreement[0])
        num_agreement.append(Preprocess.Nlp_Preprocess.pronoun_number(pr_token))
        feature_vector.append(num_agreement[0])

        anaphor_animacy = Preprocess.Nlp_Preprocess.pronoun_animacy(samples[0])
        feature_vector.append(anaphor_animacy)

        anaphora_person = Preprocess.Nlp_Preprocess.person_num(samples[0])
        feature_vector.append(anaphora_person)


        # endregion

        # region Features of antecedents
        feature_vector.append(Preprocess.Nlp_Preprocess.np_tokens_count(np_tokens))
        if Preprocess.Nlp_Preprocess.is_pronoun(np_tokens) == "T":
            feature_vector.append("1")
        else:
            feature_vector.append("0")
        feature_vector.append(Preprocess.Nlp_Preprocess.is_demonstrative(np_tokens))
        # region Calculate pos whole of Antecedent
        if int(samples[3]) > 0:
            np_pos = rcdat_pre_process_file[int(samples[3]) - 1][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos_rcdat(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[3]) > 1:
            np_pos = rcdat_pre_process_file[int(samples[3]) - 2][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos_rcdat(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[3]) > 2:
            np_pos = rcdat_pre_process_file[int(samples[3]) - 3][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos_rcdat(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[4]) < len(rcdat_pre_process_file) - 1:
            np_pos = rcdat_pre_process_file[int(samples[4]) + 1][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos_rcdat(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[4]) < len(rcdat_pre_process_file) - 2:
            np_pos = rcdat_pre_process_file[int(samples[4]) + 2][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos_rcdat(np_pos))
        else:
            feature_vector.append("0")

        if int(samples[4]) < len(rcdat_pre_process_file) - 3:
            np_pos = rcdat_pre_process_file[int(samples[4]) + 3][6]
            feature_vector.append(Preprocess.Nlp_Preprocess.calculate_pos_rcdat(np_pos))
        else:
            feature_vector.append("0")

        # endregion
        num_agreement.append(
            Preprocess.Nlp_Preprocess.calculate_number_rcdat(np_tokens, samples[3], samples[4], rcdat_pre_process_file))
        feature_vector.append(num_agreement[1])
        sbj_agreement.append(Preprocess.Nlp_Preprocess.is_subject_rcdat(samples[3], samples[4], rcdat_pre_process_file))
        feature_vector.append(sbj_agreement[1])
        obj_agreement.append(Preprocess.Nlp_Preprocess.is_object_rcdat(samples[3], samples[4], rcdat_pre_process_file))
        feature_vector.append(obj_agreement[1])

        # E How many mentions are currently in the (antecedent / anaphor)’s cluster??
        if len(np_chain_number) == 0:
            feature_vector.append(1)
        else:
            feature_vector.append(len(np_chain))


        # M the antecedent is Reflexive pronoun?
        feature_vector.append(Preprocess.Nlp_Preprocess.is_reflexive_pronoun(samples[2]))

        # M  (antecedent / anaphor) mention type?  (0 = pronoun 1= proper 2 = common)
        feature_vector.append(Preprocess.Nlp_Preprocess.mention_type(samples[3], samples[4], samples[2],
                                                                     rcdat_ne_file))

        # E The sentence # that includes the first mention in the (antecedent / anaphor ) cluster
        if len(np_chain_number) == 0 or len(np_chain) == 0:
            np_sentence = int([i for i in rcdat_pre_process_file if i[0] == samples[3]][0][1])
            feature_vector.append( np_sentence)
        else:
            feature_vector.append(int(np_chain[0][1]))

        # Animacy
        antecedent_animacy = Preprocess.Nlp_Preprocess.animacy_detection_rcdat(samples[2], samples[3], samples[4],
                                                                         rcdat_pre_process_file,
                                                                         rcdat_ne_file,
                                                                         animacy_file,
                                                                         persian_names)
        feature_vector.append(antecedent_animacy)

        # M antecedent person
        ante_person = Preprocess.Nlp_Preprocess.person_num(samples[2])
        feature_vector.append( ante_person)

        # M named_entity_type of antecedent
        en_type = Preprocess.Nlp_Preprocess.entity_Type(samples[3], samples[4], rcdat_ne_file)
        feature_vector.append(en_type)


        # E the number of antecedent cluster ( First mention of cluster)
        if len(np_chain_number) == 0 or len(np_chain) == 0:
            cluster_num = Preprocess.Nlp_Preprocess.calculate_number_rcdat(samples[2], samples[3], samples[4],
                                                                     rcdat_pre_process_file)
            feature_vector.append( cluster_num)
        else:
            cluster_num = Preprocess.Nlp_Preprocess.calculate_number_rcdat(np_chain[0][4], np_chain[0][6], np_chain[0][7],
                                                                     rcdat_pre_process_file)
            feature_vector.append(cluster_num)


        # E antecedent  first cluster animcay
        if len(np_chain_number) == 0 or len(np_chain) == 0:
            cluster_animacy = Preprocess.Nlp_Preprocess.animacy_detection_rcdat(samples[2], samples[3], samples[4],
                                                                                rcdat_pre_process_file,
                                                                                rcdat_ne_file,
                                                                          animacy_file,
                                                                          persian_names)
            feature_vector.append(cluster_animacy)
        else:
            cluster_animacy = Preprocess.Nlp_Preprocess.animacy_detection_rcdat(np_chain[0][4], np_chain[0][6],
                                                                          np_chain[0][7],
                                                                                rcdat_pre_process_file,
                                                                                rcdat_ne_file,
                                                                          animacy_file,
                                                                          persian_names)
            feature_vector.append(cluster_animacy)



        # endregion

        # region relational features
        sentence_distance = Preprocess.Nlp_Preprocess.sentence_distance(samples, rcdat_pre_process_file)
        feature_vector.append(sentence_distance)
        feature_vector.append(Preprocess.Nlp_Preprocess.token_distance(samples))
        feature_vector.append(Preprocess.Nlp_Preprocess.number_agreement(num_agreement[0], num_agreement[1]))
        feature_vector.append(Preprocess.Nlp_Preprocess.subject_agreement(sbj_agreement[0], sbj_agreement[1]))
        feature_vector.append(Preprocess.Nlp_Preprocess.object_agreement(obj_agreement[0], obj_agreement[1]))
        feature_vector.append(Preprocess.Nlp_Preprocess.string_match(pr_token, np_tokens))
        feature_vector.append(Preprocess.Nlp_Preprocess.three_distance(sentence_distance))
        feature_vector.append(Preprocess.Nlp_Preprocess.same_distance(sentence_distance))

        # E Minimum sentence distance between any two mentions from each cluster.
        if len(np_chain_number) == 0 or len(np_chain) == 0:
            feature_vector.append( sentence_distance)
        else:
            pr_sentence = int([i for i in rcdat_pre_process_file if i[0] == samples[1]][0][1])
            min_distance = min(abs(np.asarray(np_chain[:, 1], dtype=np.int) - pr_sentence))
            feature_vector.append( int(min_distance))

        # M Is length of antecedent  longer than the length of the anaphor mention ?
        if len(samples[2].split()) > len(samples[0].split()):
            feature_vector.append( 1)
        else:
            feature_vector.append( 0)
        # M animacy agreement
        feature_vector.append(Preprocess.Nlp_Preprocess.animacy_agreement(anaphor_animacy, antecedent_animacy))


        # M person agreement
        feature_vector.append(Preprocess.Nlp_Preprocess.person_agreement(ante_person, anaphora_person))



        # M If anaphor is object and antecedent is subject and both are in same sentence
        is_anphor_object = Preprocess.Nlp_Preprocess.is_object_rcdat(samples[1], samples[1], rcdat_pre_process_file)
        is_antecedent_subject = Preprocess.Nlp_Preprocess.is_subject_rcdat(samples[3], samples[4], rcdat_pre_process_file)

        if is_anphor_object == "1" and is_antecedent_subject == "1" and sentence_distance == 0:
            feature_vector.append( "1")
        else:
            feature_vector.append( "0")


        # M The anaphor mention is a reflexive pronoun and the antecedent is its subject
        if Preprocess.Nlp_Preprocess.is_subject_rcdat(samples[3], samples[4], rcdat_pre_process_file) == "1" \
            and Preprocess.Nlp_Preprocess.is_reflexive_pronoun(pr_token) == "1" and sentence_distance == 0:
            feature_vector.append( "1")
        else:
            feature_vector.append( "0")


        # endregion

        # region word vector features
        # M The euclidean distance between the tow headwords in thier corresponding word vectors
        qh = [i for i in rcdat_pre_process_file if int(samples[3]) <= int(i[0]) <= int(samples[4]) and (i[10] == "N" or i[10] == "Ne")]
        if len(qh) > 0:
            m2_head = qh[0][7]
        else:
            m2_head = samples[2].split(' ')[0]

        h1_afv = avg_feature_vector(samples[0], model=model, num_features=50, index2word_set=index2word_set)
        h2_afv = avg_feature_vector(m2_head, model=model, num_features=50, index2word_set=index2word_set)
        dist = float("{:.2f}".format(1 - spatial.distance.cosine(h1_afv, h2_afv)))
        if math.isnan(dist):
            dist = 0.0
        feature_vector.append( dist)


        # M the distance between the two noun phrases average
        m1_afv = avg_feature_vector(samples[0], model=model, num_features=50, index2word_set=index2word_set)
        m2_afv = avg_feature_vector(samples[2], model=model, num_features=50, index2word_set=index2word_set)
        dist = float("{:.2f}".format(1 - spatial.distance.cosine(m1_afv, m2_afv)))
        if math.isnan(dist):
            dist = 0.0
        feature_vector.append( dist)

        #   M calculate similarity between candidate and the sentence where anaphora is in.
        qs = [i for i in rcdat_pre_process_file if int(i[0]) == int(samples[1])][0][1]
        anaphora_sentence = rcdat_input_sentences[int(qs)]
        m2_afv = avg_feature_vector(samples[2], model=model, num_features=50, index2word_set=index2word_set)
        s2_afv = avg_feature_vector(anaphora_sentence, model=model, num_features=50, index2word_set=index2word_set)
        dist = float("{:.2f}".format(1 - spatial.distance.cosine(m2_afv, s2_afv)))
        if math.isnan(dist):
            dist = 0.0
        feature_vector.append( dist)

        # first and second token
        m3_afv = avg_feature_vector(samples[0], model=model, num_features=50, index2word_set=index2word_set).tolist()
        feature_vector.extend(m3_afv)

        m4_afv = avg_feature_vector(samples[2], model=model, num_features=50, index2word_set=index2word_set).tolist()
        feature_vector.extend(m4_afv)

        # endregion

        # region class label
        if len(samples) == 6:
            if samples[5] == "p":
                feature_vector.append( 1)
            else:
                feature_vector.append( 0)

        # endregion
        feature_vectors.append(feature_vector)
    return feature_vectors
# endregion



def sahlani_setting1(mehr_sample_creation, mehr_pre_process_file, mehr_chain_file, mehr_ne_file, animacy_file,
                    persian_names, model, mehr_input_sentences):
    feature_vectors = []
    index2word_set = model.index_to_key

    for samples in mehr_sample_creation:
        feature_vector = []
        first_mention = samples[0]
        feature_vector.append(first_mention)
        second_mention = samples[3]
        feature_vector.append(second_mention)

        # region hand crafted features
        token_distance = float(Preprocess.Nlp_Preprocess.token_distance(samples))
        feature_vector.append(token_distance)
        sentence_distance = float(Preprocess.Nlp_Preprocess.sentence_distance(samples, mehr_pre_process_file))
        feature_vector.append(sentence_distance)
        paragraph_distance = int(float(sentence_distance) / 3)
        feature_vector.append(paragraph_distance)
        mention_type_1 = Preprocess.Nlp_Preprocess.mention_type(samples[1], samples[2], samples[0],
                                                                mehr_ne_file)
        feature_vector.append(float(mention_type_1))
        mention_type_2 = Preprocess.Nlp_Preprocess.mention_type(samples[4], samples[5], samples[3],
                                                                mehr_ne_file)
        feature_vector.append(float(mention_type_2))


        head_match = Preprocess.Nlp_Preprocess.head_string_match(samples, mehr_pre_process_file)
        feature_vector.append(float(head_match))
        string_match = Preprocess.Nlp_Preprocess.string_match(samples[0], samples[3])
        feature_vector.append(float(string_match))
        partial_match = Preprocess.Nlp_Preprocess.partial_string_match(samples[0], samples[3])
        feature_vector.append(float(partial_match))
        same_speaker = Preprocess.Nlp_Preprocess.speaker_agreement(samples, mehr_pre_process_file)
        feature_vector.append(float(same_speaker))

        # num agreement
        num1 = Preprocess.Nlp_Preprocess.calculate_number(samples[0], samples[1], samples[2], mehr_pre_process_file)
        num2 = Preprocess.Nlp_Preprocess.calculate_number(samples[3], samples[4], samples[5], mehr_pre_process_file)
        number_agreement = Preprocess.Nlp_Preprocess.number_agreement(num1, num2)
        feature_vector.append(float(number_agreement))

        # animacy agreement

        if Preprocess.Nlp_Preprocess.is_pronoun(samples[0]) == "T":
            anim1 = Preprocess.Nlp_Preprocess.pronoun_animacy(samples[0])
        else:
            anim1 = Preprocess.Nlp_Preprocess.animacy_detection(samples[0], samples[1], samples[2],
                                                                mehr_pre_process_file, mehr_ne_file, animacy_file,
                                                                persian_names)
        if Preprocess.Nlp_Preprocess.is_pronoun(samples[3]) == "T":
            anim2 = Preprocess.Nlp_Preprocess.pronoun_animacy(samples[3])
        else:
            anim2 = Preprocess.Nlp_Preprocess.animacy_detection(samples[3], samples[4], samples[5],
                                                                mehr_pre_process_file, mehr_ne_file, animacy_file,
                                                                persian_names)
        animacy_agreement = Preprocess.Nlp_Preprocess.animacy_agreement(anim1, anim2)
        feature_vector.append(float(animacy_agreement))

        # gender agreement
        is_person1 = Preprocess.Nlp_Preprocess.entity_Type(samples[1], samples[2], mehr_ne_file)
        gen1 = Preprocess.Nlp_Preprocess.gender_detection(anim1, samples[0],
                                                          samples[1], samples[2],
                                                          mehr_pre_process_file, persian_names, is_person1)
        is_person2 = Preprocess.Nlp_Preprocess.entity_Type(samples[4], samples[5], mehr_ne_file)
        gen2 = Preprocess.Nlp_Preprocess.gender_detection(anim1, samples[3],
                                                          samples[4], samples[5],
                                                          mehr_pre_process_file, persian_names, is_person2)
        gender_agreement = Preprocess.Nlp_Preprocess.gender_agreement(gen1, gen2)
        feature_vector.append(float(gender_agreement))

        # endregion

        # region Word embedding feature

        # First calculate similarity between anaphora and candidate
        m1_afv = avg_feature_vector(samples[0], model=model, num_features=50, index2word_set=index2word_set)
        m2_afv = avg_feature_vector(samples[3], model=model, num_features=50, index2word_set=index2word_set)
        distance_ana_can = float("{:.2f}".format(1 - spatial.distance.cosine(m1_afv, m2_afv)))
        if math.isnan(distance_ana_can):
            distance_ana_can = 0.0
        feature_vector.append(distance_ana_can)

        # Second calculate similarity between candidate and the sentence where anaphora is in.
        qq = [i for i in mehr_pre_process_file if int(i[0]) == int(samples[1])][0][1]
        anaphora_sentence = mehr_input_sentences[int(qq)]
        m1_afv = avg_feature_vector(samples[3], model=model, num_features=50, index2word_set=index2word_set)
        s2_afv = avg_feature_vector(anaphora_sentence, model=model, num_features=50, index2word_set=index2word_set)
        distance_can_s = float("{:.2f}".format(1 - spatial.distance.cosine(m1_afv, s2_afv)))
        if math.isnan(distance_can_s):
            distance_can_s = 0.0
        feature_vector.append(distance_can_s)

        # token feature
        m3_afv = avg_feature_vector(samples[0], model=model, num_features=50, index2word_set=index2word_set).tolist()
        feature_vector.extend(m3_afv)

        m4_afv = avg_feature_vector(samples[3], model=model, num_features=50, index2word_set=index2word_set).tolist()
        feature_vector.extend(m4_afv)


        # endregion

        # region class label
        if len(samples) == 7:
            if samples[6] == "p":
                feature_vector.append(1)
            else:
                feature_vector.append(0)
        feature_vectors.append(feature_vector)

    return feature_vectors

def sahlani_setting1_rcdat(rcdat_sample_creation,
                                                                                        rcdat_pre_process_file,
                                                                                        rcdat_chain_file, rcdat_ne_file,
                                                                                        animacy_file,
                                                                                        persian_names, model,
                                                                                        rcdat_input_sentences):
    feature_vectors = []
    index2word_set = model.index_to_key

    for samples in rcdat_sample_creation:
        feature_vector = []
        first_mention = samples[0]
        feature_vector.append(first_mention)
        second_mention = samples[3]
        feature_vector.append(second_mention)

        # region hand crafted features
        token_distance = float(Preprocess.Nlp_Preprocess.token_distance(samples))
        feature_vector.append(token_distance)
        sentence_distance = float(Preprocess.Nlp_Preprocess.sentence_distance(samples, rcdat_pre_process_file))
        feature_vector.append(sentence_distance)
        paragraph_distance = int(float(sentence_distance) / 3)
        feature_vector.append(paragraph_distance)
        mention_type_1 = Preprocess.Nlp_Preprocess.mention_type(samples[1], samples[2], samples[0],
                                                                rcdat_ne_file)
        feature_vector.append(float(mention_type_1))
        mention_type_2 = Preprocess.Nlp_Preprocess.mention_type(samples[4], samples[5], samples[3],
                                                                rcdat_ne_file)
        feature_vector.append(float(mention_type_2))

        head_match = Preprocess.Nlp_Preprocess.head_string_match_rcdat(samples, rcdat_pre_process_file)
        feature_vector.append(float(head_match))
        string_match = Preprocess.Nlp_Preprocess.string_match(samples[0], samples[3])
        feature_vector.append(float(string_match))
        partial_match = Preprocess.Nlp_Preprocess.partial_string_match(samples[0], samples[3])
        feature_vector.append(float(partial_match))
        same_speaker = Preprocess.Nlp_Preprocess.speaker_agreement_rcdat(samples, rcdat_pre_process_file)
        feature_vector.append(float(same_speaker))

        # num agreement
        num1 = Preprocess.Nlp_Preprocess.calculate_number_rcdat(samples[0], samples[1], samples[2], rcdat_pre_process_file)
        num2 = Preprocess.Nlp_Preprocess.calculate_number_rcdat(samples[3], samples[4], samples[5], rcdat_pre_process_file)
        number_agreement = Preprocess.Nlp_Preprocess.number_agreement(num1, num2)
        feature_vector.append(float(number_agreement))

        # animacy agreement

        if Preprocess.Nlp_Preprocess.is_pronoun(samples[0]) == "T":
            anim1 = Preprocess.Nlp_Preprocess.pronoun_animacy(samples[0])
        else:
            anim1 = Preprocess.Nlp_Preprocess.animacy_detection_rcdat(samples[0], samples[1], samples[2],
                                                                      rcdat_pre_process_file, rcdat_ne_file, animacy_file,
                                                                persian_names)
        if Preprocess.Nlp_Preprocess.is_pronoun(samples[3]) == "T":
            anim2 = Preprocess.Nlp_Preprocess.pronoun_animacy(samples[3])
        else:
            anim2 = Preprocess.Nlp_Preprocess.animacy_detection_rcdat(samples[3], samples[4], samples[5],
                                                                      rcdat_pre_process_file, rcdat_ne_file, animacy_file,
                                                                persian_names)
        animacy_agreement = Preprocess.Nlp_Preprocess.animacy_agreement(anim1, anim2)
        feature_vector.append(float(animacy_agreement))

        # gender agreement
        is_person1 = Preprocess.Nlp_Preprocess.entity_Type(samples[1], samples[2], rcdat_ne_file)
        gen1 = Preprocess.Nlp_Preprocess.gender_detection_rcdat(anim1, samples[0],
                                                          samples[1], samples[2],
                                                                rcdat_pre_process_file, persian_names, is_person1)
        is_person2 = Preprocess.Nlp_Preprocess.entity_Type(samples[4], samples[5], rcdat_ne_file)
        gen2 = Preprocess.Nlp_Preprocess.gender_detection_rcdat(anim1, samples[3],
                                                          samples[4], samples[5],
                                                          rcdat_pre_process_file, persian_names, is_person2)
        gender_agreement = Preprocess.Nlp_Preprocess.gender_agreement(gen1, gen2)
        feature_vector.append(float(gender_agreement))

        # endregion

        # region Word embedding feature

        # First calculate similarity between anaphora and candidate
        m1_afv = avg_feature_vector(samples[0], model=model, num_features=50, index2word_set=index2word_set)
        m2_afv = avg_feature_vector(samples[3], model=model, num_features=50, index2word_set=index2word_set)
        distance_ana_can = float("{:.2f}".format(1 - spatial.distance.cosine(m1_afv, m2_afv)))
        if math.isnan(distance_ana_can):
            distance_ana_can = 0.0
        feature_vector.append(distance_ana_can)

        # Second calculate similarity between candidate and the sentence where anaphora is in.
        qq = [i for i in rcdat_pre_process_file if int(i[0]) == int(samples[1])][0][1]
        anaphora_sentence = rcdat_input_sentences[int(qq)]
        m1_afv = avg_feature_vector(samples[3], model=model, num_features=50, index2word_set=index2word_set)
        s2_afv = avg_feature_vector(anaphora_sentence, model=model, num_features=50, index2word_set=index2word_set)
        distance_can_s = float("{:.2f}".format(1 - spatial.distance.cosine(m1_afv, s2_afv)))
        if math.isnan(distance_can_s):
            distance_can_s = 0.0
        feature_vector.append(distance_can_s)

        # token feature
        m3_afv = avg_feature_vector(samples[0], model=model, num_features=50, index2word_set=index2word_set).tolist()
        feature_vector.extend(m3_afv)

        m4_afv = avg_feature_vector(samples[3], model=model, num_features=50, index2word_set=index2word_set).tolist()
        feature_vector.extend(m4_afv)

        # endregion

        # region class label
        if len(samples) == 7:
            if samples[6] == "p":
                feature_vector.append(1)
            else:
                feature_vector.append(0)
        feature_vectors.append(feature_vector)

    return feature_vectors


def sahlani_setting2(mehr_sample_creation, mehr_pre_process_file, mehr_chain_file, mehr_ne_file, animacy_file,
                    persian_names, model, mehr_input_sentences):
    feature_vectors = []
    index2word_set = model.index_to_key

    for samples in mehr_sample_creation:
        feature_vector =np.array([])
        first_mention = samples[0]
        feature_vector = np.append(feature_vector,first_mention)
        second_mention = samples[3]
        feature_vector = np.append(feature_vector, second_mention)

        # region hand crafted features
        token_distance = float(Preprocess.Nlp_Preprocess.token_distance(samples))
        feature_vector = np.append(feature_vector, token_distance)
        sentence_distance = float(Preprocess.Nlp_Preprocess.sentence_distance(samples, mehr_pre_process_file))
        feature_vector = np.append(feature_vector,sentence_distance )
        paragraph_distance = int(float(sentence_distance) / 3)
        feature_vector = np.append(feature_vector,paragraph_distance )
        mention_type_1 = Preprocess.Nlp_Preprocess.mention_type(samples[1], samples[2], samples[0],
                                                                mehr_ne_file)
        feature_vector = np.append(feature_vector,float(mention_type_1) )
        mention_type_2 = Preprocess.Nlp_Preprocess.mention_type(samples[4], samples[5], samples[3],
                                                                mehr_ne_file)
        feature_vector = np.append(feature_vector,float(mention_type_2) )


        head_match = Preprocess.Nlp_Preprocess.head_string_match(samples, mehr_pre_process_file)
        feature_vector = np.append(feature_vector,float(head_match) )
        string_match = Preprocess.Nlp_Preprocess.string_match(samples[0], samples[3])
        feature_vector = np.append(feature_vector,float(string_match) )
        partial_match = Preprocess.Nlp_Preprocess.partial_string_match(samples[0], samples[3])
        feature_vector = np.append(feature_vector,float(partial_match) )
        same_speaker = Preprocess.Nlp_Preprocess.speaker_agreement(samples, mehr_pre_process_file)
        feature_vector = np.append(feature_vector,float(same_speaker) )

        # num agreement
        num1 = Preprocess.Nlp_Preprocess.calculate_number(samples[0], samples[1], samples[2], mehr_pre_process_file)
        num2 = Preprocess.Nlp_Preprocess.calculate_number(samples[3], samples[4], samples[5], mehr_pre_process_file)
        number_agreement = Preprocess.Nlp_Preprocess.number_agreement(num1, num2)
        feature_vector = np.append(feature_vector, float(number_agreement))

        # animacy agreement

        if Preprocess.Nlp_Preprocess.is_pronoun(samples[0]) == "T":
            anim1 = Preprocess.Nlp_Preprocess.pronoun_animacy(samples[0])
        else:
            anim1 = Preprocess.Nlp_Preprocess.animacy_detection(samples[0], samples[1], samples[2],
                                                                mehr_pre_process_file, mehr_ne_file, animacy_file,
                                                                persian_names)
        if Preprocess.Nlp_Preprocess.is_pronoun(samples[3]) == "T":
            anim2 = Preprocess.Nlp_Preprocess.pronoun_animacy(samples[3])
        else:
            anim2 = Preprocess.Nlp_Preprocess.animacy_detection(samples[3], samples[4], samples[5],
                                                                mehr_pre_process_file, mehr_ne_file, animacy_file,
                                                                persian_names)
        animacy_agreement = Preprocess.Nlp_Preprocess.animacy_agreement(anim1, anim2)
        feature_vector = np.append(feature_vector, float(animacy_agreement))

        # gender agreement
        is_person1 = Preprocess.Nlp_Preprocess.entity_Type(samples[1], samples[2], mehr_ne_file)
        gen1 = Preprocess.Nlp_Preprocess.gender_detection(anim1, samples[0],
                                                          samples[1], samples[2],
                                                          mehr_pre_process_file, persian_names, is_person1)
        is_person2 = Preprocess.Nlp_Preprocess.entity_Type(samples[4], samples[5], mehr_ne_file)
        gen2 = Preprocess.Nlp_Preprocess.gender_detection(anim1, samples[3],
                                                          samples[4], samples[5],
                                                          mehr_pre_process_file, persian_names, is_person2)
        gender_agreement = Preprocess.Nlp_Preprocess.gender_agreement(gen1, gen2)
        feature_vector = np.append(feature_vector,float(gender_agreement) )

        # endregion

        # region Word embedding feature

        # First calculate similarity between anaphora and candidate
        m1_afv = avg_feature_vector(samples[0], model=model, num_features=50, index2word_set=index2word_set)
        m2_afv = avg_feature_vector(samples[3], model=model, num_features=50, index2word_set=index2word_set)
        distance_ana_can = float("{:.2f}".format(1 - spatial.distance.cosine(m1_afv, m2_afv)))
        if math.isnan(distance_ana_can):
            distance_ana_can = 0.0
        feature_vector = np.append(feature_vector, distance_ana_can)

        # Second calculate similarity between candidate and the sentence where anaphora is in.
        qq = [i for i in mehr_pre_process_file if int(i[0]) == int(samples[1])][0][1]
        anaphora_sentence = mehr_input_sentences[int(qq)]
        m1_afv = avg_feature_vector(samples[3], model=model, num_features=50, index2word_set=index2word_set)
        s2_afv = avg_feature_vector(anaphora_sentence, model=model, num_features=50, index2word_set=index2word_set)
        distance_can_s = float("{:.2f}".format(1 - spatial.distance.cosine(m1_afv, s2_afv)))
        if math.isnan(distance_can_s):
            distance_can_s = 0.0
        feature_vector = np.append(feature_vector, distance_can_s)
        # endregion

        # class
        if samples[6] == "p":
            feature_vector = np.append(feature_vector,1 )
        else:
            feature_vector = np.append(feature_vector,0 )
        feature_vectors.append(feature_vector)


    return feature_vectors

# region Convert Features to data frame
def convert_to_data_frame(feature_vector):
    df = DataFrame(feature_vector)
    return df

# endregion
