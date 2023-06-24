import Preprocess.Feature_Vector
import Preprocess.Mehr_Corpus_Preprocess
import pandas as pd
import itertools
import warnings
import statistics
from sklearn.ensemble import RandomForestClassifier
from pandas import DataFrame

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Experiment 10 time and produce results

prec_lst = []
recall_lst = []
f1_lst = []
for iteration in range(1,11):
    print("Iteration : " , iteration)
    # Train final optimal model and save it as file
    dataset = pd.read_csv('full_train_fv.csv')
    column_count = len(dataset.iloc[0])
    x_train = dataset.iloc[:, 2:column_count - 1]
    y_train = dataset.iloc[:, column_count - 1]
    clf = RandomForestClassifier(n_estimators=1000, max_depth=20)
    clf.fit(x_train, y_train)


    number_of_documents = 0
    number_of_pr = 0
    number_of_resolved = 0
    number_of_correct = 0


    for i in range(358, 401):
        number_of_documents += 1
        print("document" + str(i))

        # region Import Conll, NP, NE, Preprocess and extract sentence and coref chains
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

        # region Test model
        final_pair = []
        sentence_window = 3  # only match 3 sentence candidates before pronoun

        # Load trained model

        # model for train + test
        # clf = load('./Models/Mehr_Perfect_Random_Forest.joblib')


        # Extracting pronouns manually
        pronouns = [i for i in mehr_pre_process_file if i[10] == "PRO"]
        pr_gold = [i for i in mehr_chain_file if i[0] == "T"]
        pr_system = []
        for prs in pronouns:
            pr_srch = [i for i in pr_gold if prs[0] == i[6]]
            if len(pr_srch) > 0:
                pr_system.append(pr_srch[0])
            else:
                continue

        for pr in pr_system:
            base_sample = []

            # Add all np's , ne's, pronoun's  backward in w windows

            np_candidates = [i for i in mehr_np_file if
                             int(i[1]) < int(pr[6]) and int(pr[1]) - int(i[2]) <= sentence_window]

            if len(np_candidates) > 0:
                for np in np_candidates:
                    temp_sample = [pr[4], pr[6], np[5], np[0], np[1]]
                    base_sample.append(temp_sample)

            pr_candidates = [i for i in pr_system if
                              int(pr[6]) > int(i[6]) and int(pr[1]) - int(i[1]) <= sentence_window]

            if len(pr_candidates) > 0:
                for pr_c in pr_candidates:
                    # Check if np candidate boundaries match the np candidate boundaries
                    # np_match = [i for i in np_file if i[0] == pr_c[6] and i[1] == pr_c[7]]
                    # if len(np_match) == 0:
                    temp_sample = [pr[4], pr[6], pr_c[4], pr_c[6], pr_c[7]]
                    base_sample.append(temp_sample)

            # For each Named Entity between pronoun and its real antecedent create a negative exam
            ne_candidates = [i for i in mehr_ne_file if
                                 int(i[1]) < int(pr[6]) and int(pr[1]) - int(i[2]) <= sentence_window]

            if len(ne_candidates) > 0:
                for ne in ne_candidates:
                    # Check if ne candidate boundaries match the np candidate boundaries
                    # np_match = [i for i in np_file if i[0] == ne[0] and i[1] == ne[1]]

                    # if len(np_match) == 0:
                    temp_sample = [pr[4], pr[6], ne[5], ne[0], ne[1]]
                    base_sample.append(temp_sample)
            # endregion

            # sort candidates backward and remove duplicates
            base_sample = sorted(base_sample, key=lambda x: int(x[3]), reverse=True)
            base_sample = list(base_sample for base_sample, _ in itertools.groupby(base_sample))

            # Create feature vector for each candidates for current pronoun
            temp_feature_vec = Preprocess.Feature_Vector.mention_pair_setting1_mehr(base_sample, mehr_pre_process_file)
            # drop first two np tokens that arent feature

            # temp_feature_vec = temp_feature_vec[0][0:len(temp_feature_vec[0]) ]
            dataset = DataFrame.from_records(temp_feature_vec)
            column_count = len(dataset.iloc[0])
            temp_feature_vec = dataset.iloc[:, 2:column_count]

            # Calculate Class and Probability for all feature vector for current pronoun
            predict_r = list(map(clf.predict, [temp_feature_vec]))[0]
            predict_p = list(map(clf.predict_proba, [temp_feature_vec]))[0]
            predict_p = [j for i, j in predict_p]
            predict_all = [[i, j] for i, j in zip(predict_r, predict_p)]
            predict_result = [i + j for i, j in zip(base_sample, predict_all)]

            # Best First Antecedent selection

            antes = [i for i in predict_result if i[5] == 1]

            if len(antes) > 0:  # found an antecedent for current pronoun
                max_antes = max(antes, key=lambda item: item[6])
                final_pair.append(max_antes)
            else:  # Didn't find an antecedent for current pronoun
                final_pair.append([pr[4], pr[6], "-", "-", "-", "-", "-"])

        # region Calculate precision recall and F_1 measure for this document
        number_pr_in_doc = len(pr_gold)
        number_pr_resolved_in_doc = len(final_pair)
        number_pr_correct_resolved_in_doc = 0

        for result in final_pair:
            pr_chain_num = [i for i in mehr_chain_file if
                            result[1] == i[6]][0][5]
            real_antecedent = [i for i in mehr_chain_file if
                               i[5] == pr_chain_num and int(i[6]) < int(result[1])]

            if len(real_antecedent) > 0:
                if result[3] != "-":
                    antecedent_candidate_f_token = int(result[3])
                    antecedent_candidate_l_token = int(result[4])
                    check_antecedent_in_chain = [i for i in real_antecedent if int(
                        i[6]) <= antecedent_candidate_f_token and antecedent_candidate_l_token <= int(i[7])]
                    if (len(check_antecedent_in_chain)) > 0:
                        number_pr_correct_resolved_in_doc += 1
            else:
                if result[3] == "-":
                    number_pr_correct_resolved_in_doc += 1

        number_of_pr = number_of_pr + number_pr_in_doc
        number_of_resolved = number_of_resolved + number_pr_resolved_in_doc
        number_of_correct = number_of_correct + number_pr_correct_resolved_in_doc
        # endregion
        # endregion

    current_prec = number_of_correct / number_of_resolved
    prec_lst.append(current_prec)
    current_recall = number_of_correct / number_of_pr
    recall_lst.append(current_recall)
    current_f1 = (2 * (current_prec * current_recall)) / (current_prec + current_recall)
    f1_lst.append(current_f1)


print("Final Precision list is: " ,prec_lst )
print("Final Precision  is: " ,statistics.mean(prec_lst) )
print("Final Recall list is : " , recall_lst)
print("Final Recall  is: " ,statistics.mean(recall_lst) )
print("Final f1 list is : " , f1_lst)
print("Final f1 list  is: " ,statistics.mean(f1_lst) )