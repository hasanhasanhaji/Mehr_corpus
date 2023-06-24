import Preprocess.Nlp_Preprocess


# Mehr Convert Conll File To Structured File
def mehr_conll_to_structured(document_id):
    conll_file_p = "../Corpus/Mehr Corpus/Conll File/" + str(document_id) + ".conll"
    conll_file = []
    structured_file = []
    sentence_count = 0
    # Read line by line whole input text
    with open(conll_file_p, encoding="utf-8") as fp:
        while True:
            line = fp.readline()
            if "#begin document" in line:
                continue
            if "#end document" in line:
                break
            if line == "\n":
                sentence_count += 1
                continue
            temp_list = line.split()
            temp_list.append(str(sentence_count))
            conll_file.append(temp_list)

    # Convert conll file to structured file
    line_count = 0
    for lines in conll_file:
        temp_lst = [lines[12], lines[2], lines[3], lines[11], str(line_count)]
        line_count += 1
        structured_file.append(temp_lst)

    return structured_file


"""
def animacy2_file():
    animacy_file_p = "../Preprocess/" + "animacy" + ".conll"
    animacy_file = []
    structured_file = []
    # Read line by line whole input text
    with open(animacy_file_p, encoding="utf-8") as fp:
        while True:
            line = fp.readline()
            if line == "\n":
                continue
            if "#end" in line:
                break
            temp_list = line.split()
            animacy_file.append(temp_list)

    # Convert conll file to structured file
    for lines in animacy_file:
        temp_lst = [lines[1], lines[2], lines[4]]
        structured_file.append(temp_lst)
    return structured_file
"""


# Import animacy file
def animacy_file(file_num):
    animacy_file_p = "../Corpus/Mehr Corpus/animacy/" + str(file_num) + ".animacy"
    animacy_file = []
    structured_file = []

    with open(animacy_file_p, encoding="utf-8") as fp:
        while True:
            line = fp.readline()
            if line == "":
                break
            temp_list = line.split()
            animacy_file.append(temp_list)

    # Convert conll file to structured file
    for lines in animacy_file:
        temp_lst = [lines[0], lines[1], lines[2]]
        structured_file.append(temp_lst)
    return structured_file


# Import persian names file
def p_name_file():
    p_name_file_p = "../Preprocess/" + "persian_names" + ".txt"
    p_name_file = []
    structured_file = []
    # Read line by line whole input text
    with open(p_name_file_p, encoding="utf-8") as fp:
        while True:
            line = fp.readline()
            if line == "\n":
                continue
            if "#end" in line:
                break
            temp_list = line.split(",")
            p_name_file.append(temp_list)

    # Convert conll file to structured file
    for lines in p_name_file:
        temp_lst = [lines[0], lines[1]]
        structured_file.append(temp_lst)
    return structured_file


# Import Np file
def mehr_np_file(document_id):
    np_file_p = "../Corpus/Mehr Corpus/np/" + str(document_id) + ".np"
    np_file = []

    with open(np_file_p, encoding="utf-8") as fp:
        for line in fp:
            temp_list = line.split("\t")

            if "\n" in temp_list and len(temp_list) > 2:
                temp_list.remove("\n")
            else:
                continue
            np_file.append(temp_list)

    return np_file


# Import Ne file
def mehr_ne_file(document_id):
    ne_file_p = "../Corpus/Mehr Corpus/ne/" + str(document_id) + ".ne"
    ne_file = []

    with open(ne_file_p, encoding="utf-8") as fp:
        for line in fp:
            temp_list = line.split("\t")
            temp_list.remove("\n")
            ne_file.append(temp_list)

    return ne_file


# Import preprocess file
def mehr_pre_process_file(document_id):
    pre_file_p = "../Corpus/Mehr Corpus/preproc/" + str(document_id) + ".preproc"
    preprocess_file = []

    with open(pre_file_p, encoding="utf-8") as fp:
        for line in fp:
            temp_list = line.split("\t")
            temp_list.remove("\n")
            preprocess_file.append(temp_list)
    return preprocess_file


# Extract Sentences from structured file
def extract_sentences(structured_file):
    sentence_count = int(structured_file[len(structured_file) - 1][0])
    input_sentences = []

    for sentence in range(0, sentence_count + 1):
        sen_tokens = [i for i in structured_file if i[0] == str(sentence)]
        temp_string = ""

        for item in sen_tokens:
            temp_string += item[2] + " "
        input_sentences.append(temp_string.rstrip())

    return input_sentences


# Mehr create chain File

def mehr_chain_file(structured_file):
    """
    first column ==> Is Pronoun?
    second column ==> Sentence number
    third column ==> First token
    fourth column ==> Last token
    Fifth column ==> Noun phrase string
    6th column ==> Chain number
    7th column ==> File token start
    8th column ==> File token End
    :param structured_file: فایل ساختار یافته
    :return: لیست حاوی اطلاعات هم مرجع
    """

    chains = []
    token_count = 0

    try:
        for item in structured_file:
            temp_row = []

            if item[3] == "-":
                token_count += 1
                continue

            # For resolve tokens like this :  (1)
            elif "(" in item[3] and ")" in item[3] and "|" not in item[3]:

                temp_row.append(Preprocess.Nlp_Preprocess.is_pronoun(item[2]))
                temp_row.append(item[0])
                temp_row.append(item[1])
                temp_row.append(item[1])
                temp_row.append(item[2])

                temp_row.append(str(item[3].strip("()")))
                temp_row.append(str(token_count))
                temp_row.append(str(token_count))
                chains.append(temp_row)
                token_count += 1

            # For resolve tokens like this: (10
            elif "(" in item[3] and "|" not in item[3] and ")" not in item[3]:
                temp_row.append("F")
                temp_row.append(item[0])
                temp_row.append(item[1])
                # for second token we must find the whole noun phrase in current chain

                temp = [i for i in structured_file if
                        ")" in i[3] and item[3].strip("()") in i[3].replace(")", "|").replace("(", "|").split("|") and
                        int(i[4]) > int(
                            item[4])][0]

                temp_row.append(temp[1])

                # for Np string we must create string between first and last token of current chain
                np_string = ""
                for j in range(int(temp_row[2]), int(temp_row[3]) + 1):
                    temp = [i for i in structured_file if i[0] == item[0] and i[1] == str(j)][0]
                    np_string += temp[2] + " "
                temp_row.append(np_string.rstrip())
                # Append chain number
                temp_row.append(str(item[3].strip("()")))
                # Append First and last item Document token
                temp_row.append(str(token_count))
                last_item_token_count = token_count + (int(temp[1]) - int(item[1]))
                temp_row.append(str(last_item_token_count))

                chains.append(temp_row)
                token_count += 1
                # Write code for tokens like this in corpus 9)|(10) . we want extract (10)

            # For resolve tokens like this   (4|(5 or    (4| (5) or (4) |(5) or 4)|(5
            elif item[3].count("(") >= item[3].count(")") and "|" in item[3]:
                split_tokens = item[3].split("|")

                for spl_t in split_tokens:
                    temp_row = []
                    # for tokens like (5)
                    if "(" in spl_t and ")" in spl_t:
                        temp_row.append(Preprocess.Nlp_Preprocess.is_pronoun(item[2]))
                        temp_row.append(item[0])
                        temp_row.append(item[1])
                        temp_row.append(item[1])
                        temp_row.append(item[2])

                        temp_row.append(str(spl_t.strip("()")))
                        temp_row.append(str(token_count))
                        temp_row.append(str(token_count))
                        chains.append(temp_row)
                    # for tokens like (5
                    elif "(" in spl_t:
                        temp_row.append("F")
                        temp_row.append(item[0])
                        temp_row.append(item[1])
                        # for second token we must find the whole noun phrase in current chain

                        temp = [i for i in structured_file if
                                ")" in i[3] and spl_t.strip("()") in i[3].replace(")", "|").replace("(", "|").split(
                                    "|") and int(
                                    i[4]) > int(item[4])][0]
                        temp_row.append(temp[1])
                        # for Np string we must create string between first and last token of current chain
                        np_string = ""
                        for j in range(int(temp_row[2]), int(temp_row[3]) + 1):
                            temp = [i for i in structured_file if i[0] == item[0] and i[1] == str(j)][0]
                            np_string += temp[2] + " "
                        temp_row.append(np_string.rstrip())
                        # Append chain number
                        temp_row.append(str(spl_t.strip("()")))
                        # Append First and last item Document token
                        temp_row.append(str(token_count))
                        last_item_token_count = token_count + (int(temp[1]) - int(item[1]))
                        temp_row.append(str(last_item_token_count))
                        chains.append(temp_row)

                token_count += 1
            # For resolve tokens like this 4)||(5)
            else:
                split_tokens = item[3].split("|")

                for spl_t in split_tokens:
                    if "(" in spl_t and ")" in spl_t:
                        temp_row.append(Preprocess.Nlp_Preprocess.is_pronoun(item[2]))
                        temp_row.append(item[0])
                        temp_row.append(item[1])
                        temp_row.append(item[1])
                        temp_row.append(item[2])

                        temp_row.append(str(spl_t.strip("()")))
                        temp_row.append(str(token_count))
                        temp_row.append(str(token_count))
                        chains.append(temp_row)

                token_count += 1
    except:
        print(item)

    return chains


# Mehr Create perfect sample creation
def mehr_perfect_sample_creation(chain_file, np_file, ne_file):
    base_sample = []

    # Search for pronouns
    pronouns = [i for i in chain_file if i[0] == "T"]

    #  Create samples for all pronouns in the current document
    for pr in pronouns:

        # Find   Antecedents from corpus for current pronoun
        antecedents = [i for i in chain_file if i[5] == pr[5] and int(i[6]) < int(pr[6])]
        # if pronoun have at least on antecedent
        if len(antecedents) > 0:
            # Select Near Antecedent and Choose it for Creating feature vector
            antecedent = antecedents[-1]

            # Create positive example for current pronoun
            temp_sample = [pr[4], pr[6], antecedent[4], antecedent[6], antecedent[7], "p"]
            base_sample.append(temp_sample)

            # region Create negative examples for current pronoun

            # For each Noun phrase between pronoun and its real antecedent create a negative exam
            np_candidates = [i for i in np_file if int(i[1]) < int(pr[6]) and int(i[0]) > int(antecedent[7])]
            if len(np_candidates) > 0:
                for np in np_candidates:
                    temp_sample = [pr[4], pr[6], np[5], np[0], np[1], "n"]
                    base_sample.append(temp_sample)

            # For Each Pronoun between pronoun and its real antecedent that isn't find in previous block
            # create a negative example
            pr_candidates = [i for i in chain_file if i[0] == "T" and int(pr[6]) > int(i[6]) > int(antecedent[7])]
            if len(pr_candidates) > 0:
                for pr_c in pr_candidates:
                    # Check if np candidate boundaries match the np candidate boundaries
                    np_match = [i for i in np_file if i[0] == pr_c[6] and i[1] == pr_c[7]]

                    if len(np_match) == 0:
                        temp_sample = [pr[4], pr[6], pr_c[4], pr_c[6], pr_c[7], "n"]
                        base_sample.append(temp_sample)

            # For each Named Entity between pronoun and its real antecedent create a negative exam
            ne_candidates = [i for i in ne_file if int(i[1]) < int(pr[6]) and int(i[0]) > int(antecedent[7])]

            if len(ne_candidates) > 0:
                for ne in ne_candidates:
                    # Check if ne candidate boundaries match the np candidate boundaries
                    np_match = [i for i in np_file if i[0] == ne[0] and i[1] == ne[1]]

                    if len(np_match) == 0:
                        temp_sample = [pr[4], pr[6], ne[5], ne[0], ne[1], "n"]
                        base_sample.append(temp_sample)

            # endregion
    return base_sample


# Mehr Create perfect sample creation for coreference resolution
def mehr_perfect_sample_creation_coref(chain_file, np_file, ne_file):
    base_sample = []

    # Search for pronouns
    anaphors = chain_file
    #  Create samples for all pronouns in the current document
    for pr in anaphors:

        # Find   Antecedents from corpus for current pronoun
        antecedents = [i for i in chain_file if i[5] == pr[5] and int(i[6]) < int(pr[6])]
        # if pronoun have at least on antecedent
        if len(antecedents) > 0:
            # Select Near Antecedent and Choose it for Creating feature vector
            antecedent = antecedents[-1]

            # Create positive example for current pronoun
            temp_sample = [pr[4], pr[6], pr[7], antecedent[4], antecedent[6], antecedent[7], "p"]
            base_sample.append(temp_sample)

            # region Create negative examples for current pronoun

            # For each Noun phrase between pronoun and its real antecedent create a negative exam
            np_candidates = [i for i in np_file if int(i[1]) < int(pr[6]) and int(i[0]) > int(antecedent[7])]
            if len(np_candidates) > 0:
                for np in np_candidates:
                    temp_sample = [pr[4], pr[6], pr[7], np[5], np[0], np[1], "n"]
                    base_sample.append(temp_sample)

            # For Each Pronoun between pronoun and its real antecedent that isn't find in previous block
            # create a negative example
            pr_candidates = [i for i in chain_file if i[0] == "T" and int(pr[6]) > int(i[6]) > int(antecedent[7])]
            if len(pr_candidates) > 0:
                for pr_c in pr_candidates:
                    # Check if np candidate boundaries match the np candidate boundaries
                    np_match = [i for i in np_file if i[0] == pr_c[6] and i[1] == pr_c[7]]

                    if len(np_match) == 0:
                        temp_sample = [pr[4], pr[6], pr[7], pr_c[4], pr_c[6], pr_c[7], "n"]
                        base_sample.append(temp_sample)

            # For each Named Entity between pronoun and its real antecedent create a negative exam
            ne_candidates = [i for i in ne_file if int(i[1]) < int(pr[6]) and int(i[0]) > int(antecedent[7])]

            if len(ne_candidates) > 0:
                for ne in ne_candidates:
                    # Check if ne candidate boundaries match the np candidate boundaries
                    np_match = [i for i in np_file if i[0] == ne[0] and i[1] == ne[1]]

                    if len(np_match) == 0:
                        temp_sample = [pr[4], pr[6], pr[7], ne[5], ne[0], ne[1], "n"]
                        base_sample.append(temp_sample)

            # endregion
    return base_sample
