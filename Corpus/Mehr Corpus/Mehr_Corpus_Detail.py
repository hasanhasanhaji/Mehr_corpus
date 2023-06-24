def document_token_count(file_name):
    line_count = 0
    with open(file_name, encoding="utf-8") as fp:
        while True:
            line = fp.readline()
            if "#begin document" in line:
                continue
            elif "#end document" in line:
                break
            elif line == "\n":
                continue
            else:
                line_count += 1

    return line_count

def document_sentence_count(conll_file_p):
    with open(conll_file_p, encoding="utf-8") as fp:
        for line in fp:
            pass
        last_line = line
        sen_count = last_line.split() [1]
    return int(sen_count)

def corpus_sentence_count():
    sum = 0
    for i in range(358, 401):
        conll_file_p = "../Mehr Corpus/preproc/" + str(i) + ".preproc"
        doc_sen = document_sentence_count(conll_file_p)
        sum += doc_sen

    return sum

def corpus_token_count():
    sum = 0
    for i in range(1, 401):
        conll_file_p = "../Mehr Corpus/Conll File/" + str(i) + ".conll"
        doc_tokens = document_token_count(conll_file_p)
        sum += doc_tokens

    return sum


def test_split():
    sum = 0
    for i in range(1, 323):
        conll_file_p = "../Mehr Corpus/Conll File/" + str(i) + ".conll"
        doc_tokens = document_token_count(conll_file_p)
        sum += doc_tokens
    return sum


# region Corpus Details
#print("تعداد توکن های کل پیکره مهر:")
#print(corpus_token_count())
#print("تعداد توکن های بخش آموزش شامل سند 1 الی سند 357")
#print(test_split())

print("تعداد جملات کل پیکره مهر:")
print(corpus_sentence_count())
# endregion
