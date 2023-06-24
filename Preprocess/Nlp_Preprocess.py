import Preprocess.Nlp_Preprocess


# region pronoun's modules
def is_pronoun(v):
    """
    This module determine if the current Token from corpus chain Is Pronoun or not
    :param v: token Inserted
    :return: T if the token is Pronoun.
    """
    if v == "من" or v == "تو" or v == "او" or v == "وی" or v == "ما" or v == "شما" or v == "ایشان" or \
            v == "آنها" or v == "انها" or v == "آنان" or v == "انان" or v == "اینان" or v == "این" or v == "اینها" or \
            v == "آن" or v == "ان" or v == "همین" or v == "همان" or v == "خود" or v == "خودم" or v == "خودش" or \
            v == "خودت" or v == "خودمان" or v == "خودتان" or v == "خودشان" or v == "خویش":
        return "T"
    else:
        return "F"


def is_personal_pronoun(pr_token):
    if pr_token == "من" or pr_token == "تو" or pr_token == "او" or pr_token == \
            "وی" or pr_token == "ما" or pr_token == "شما" or pr_token == "ایشان" or pr_token == \
            "آنها" or pr_token == "انها":
        return "1"
    else:
        return "0"


def is_demonstartive_pronoun(pr_token):
    if pr_token == "این" or pr_token == "آن" or pr_token == "ان" or pr_token == "اینان" or \
            pr_token == "انان" or pr_token == "آنان" or pr_token == "همین" or pr_token == \
            "همان" or pr_token == "اینها":
        return "1"
    else:
        return "0"


def is_third_person(pr_token):
    if pr_token == "او" or pr_token == "وی" or pr_token == "ایشان" or pr_token == "آنها" or \
            pr_token == "انها" or pr_token == "خودش" or pr_token == "خودشان" or pr_token == \
            "آنان" or pr_token == "انان":
        return "1"
    else:
        return "0"


def is_reflexive_pronoun(pr_token):
    if pr_token == "خودم" or pr_token == "خودت" or pr_token == "خودش" or pr_token == \
            "خودمان" or pr_token == "خودتان" or pr_token == "خودشان" or pr_token == \
            "خود" or pr_token == "خویش":
        return "1"
    else:
        return "0"


def is_speech(pr_token):
    if pr_token == "من" or pr_token == "تو" or pr_token == "ما" or pr_token == \
            "شما" or pr_token == "خودم" or pr_token == "خودت" or pr_token == "خودمان" or pr_token == "خودتان":
        return "1"
    else:
        return "0"


def person_num(pr_token):
    if pr_token == "من" or pr_token == "ما" or pr_token == "خودم" or pr_token == "خودمان":
        return "1"
    elif pr_token == "تو" or pr_token == "شما" or pr_token == "خودت" or pr_token == "خودتان":
        return "2"
    elif pr_token == "او" or pr_token == "وی" or pr_token == "ایشان" or pr_token == "آنها" or \
            pr_token == "انها" or pr_token == "خودش" or pr_token == "خودشان" or pr_token == \
            "آنان" or pr_token == "انان":
        return "3"
    else:
        return "4"


def pronoun_number(pr_token):
    if pr_token == "من" or pr_token == "تو" or pr_token == "او" or \
            pr_token == "وی" or pr_token == "این" or pr_token == "آن" or pr_token == \
            "ان" or pr_token == "همین" or pr_token == "همان" or pr_token == "خودم" or \
            pr_token == "خودت" or pr_token == "خودش" or pr_token == "خویش":
        return "0"
    elif pr_token == "ما" or pr_token == "شما" or pr_token == "ایشان" or \
            pr_token == "آنها" or pr_token == "انها" or pr_token == "اینها" or pr_token == "همینها" or \
            pr_token == "همانها" or pr_token == "خودمان" or pr_token == "خودتان" or pr_token == "آنان" or \
            pr_token == "خودشان" or pr_token == "اینان" or pr_token == "انان":
        return "1"
    else:
        return "2"


def pronoun_animacy(v):
    if v == "من" or v == "تو" or v == "او" or v == "وی" or v == "ما" or v == "شما" or v == "ایشان" or \
            v == "خودم" or v == "خودش" or \
            v == "خودت" or v == "خودمان" or v == "خودتان" or v == "خودشان" or v == "خویش":
        return "1"
    elif v == "آن" or v == "این" or v == "آن" or v == "ان" or v == "همین" or v == "همان":
        return "0"
    else:
        return "2"


# endregion

# region Np's modules

def calculate_pos(np_pos):
    if np_pos == "N_SING" or np_pos == "N_PL":
        return "1"  # اسم
    elif np_pos == "CLITIC":
        return "2"  # را
    elif "ADJ" in np_pos:
        return "3"  # صفت
    elif np_pos == "NUM":
        return "4"  # عدد
    elif "V_" in np_pos:
        if "ADV_" in np_pos:
            return "7"  # قید
        else:
            return "5"  # فعل
    elif np_pos == "DELM":
        return "6"  # علایم نگارشی
    elif "ADV" in np_pos:
        return "7"  # قید
    elif np_pos == "CON":
        return "8"  # نقش نما-حرف ربط
    elif np_pos == "PRO":
        return "9"  # ضمیر
    elif np_pos == "DET":
        return "10"  # صفت اشاره
    else:
        return "11"  # حرف اضافه

def calculate_pos_rcdat(np_pos):
    if np_pos == "NOUN" :
        return "1"  # اسم
    elif np_pos == "ADP":
        return "2"  # حزف اضافه
    elif "ADJ" == np_pos:
        return "3"  # صفت
    elif np_pos == "AUX":
        return "4"  # فعل کمکی
    elif np_pos == "PROPN":
        return "5" # اسم خاص
    elif "VERB" == np_pos:
        return "6"  #  فعل
    elif np_pos == "PUNCT":
        return "7"  # علایم نگارشی
    elif np_pos =="NUM":
        return "8"  #عدد
    elif "CONJ" in np_pos:
        return "9"  # نقش نما-حرف ربط
    elif np_pos == "PRON":
        return "10"  # ضمیر
    elif np_pos == "DET":
        return "11"  # صفت اشاره
    elif np_pos == "INTJ":
        return "12"  #
    elif np_pos == "ADV":
        return "13"  #قید
    elif np_pos == "PART":
        return "14"  # آیا
    else:
        return "15"  #

def is_subject(f_token, l_token, preprocessed_file):
    roles = [i for i in preprocessed_file if int(f_token) <= int(i[0]) <= int(l_token) and "SBJ" in [i[4]]]

    if len(roles) > 0:
        return "1"
    else:
        return "0"

def is_subject_rcdat(f_token, l_token, preprocessed_file):
    roles = [i for i in preprocessed_file if int(f_token) <= int(i[0]) <= int(l_token) and "nsubj" in [i[4]]]

    if len(roles) > 0:
        return "1"
    else:
        return "0"

def is_object(first_token, last_token, preprocessed_file):
    obj_1 = [i for i in preprocessed_file if int(first_token) <= int(i[0]) <= int(last_token) and i[4] == "OBJ"]
    if len(obj_1) > 0:
        return "1"

    else:
        if int(last_token) != len(preprocessed_file) - 1:
            obj_2 = [i for i in preprocessed_file if
                     int(i[0]) == int(last_token) + 1 and i[4] == "OBJ" and i[7] == "را"]

            if len(obj_2) > 0:
                return "1"
            else:
                return "0"
        else:
            return "0"


def is_object_rcdat(first_token, last_token, preprocessed_file):
    obj_1 = [i for i in preprocessed_file if int(first_token) <= int(i[0]) <= int(last_token) and i[4] == "obj"]
    if len(obj_1) > 0:
        return "1"

    else:
        if int(last_token) != len(preprocessed_file) - 1:
            obj_2 = [i for i in preprocessed_file if
                     int(i[0]) == int(last_token) + 1 and i[4] == "OBJ" and i[7] == "را"]

            if len(obj_2) > 0:
                return "1"
            else:
                return "0"
        else:
            return "0"

def np_tokens_count(np_tokens):
    tokens = str(np_tokens).split(' ')
    return len(tokens)


def is_demonstrative(np_tokens):
    tokens = str(np_tokens).split(' ')
    if len(tokens) > 1:
        if tokens[0] == "این" or tokens[0] == "آن" or tokens[0] == "ان" or tokens[0] == "همین" or tokens[0] == "همان":
            return "1"
        else:
            return "0"
    else:
        return "0"


def calculate_number(np_tokens, ante_f_token, ante_l_token, m_preprocessed_file):
    tokens = str(np_tokens).split(' ')
    pos = [i[6] for i in m_preprocessed_file if int(ante_f_token) <= int(i[0]) <= int(ante_l_token)]

    if len(pos) == 0 :
        return "0"

    if len(tokens) == 1:
        if is_pronoun(np_tokens) == "T":
            return pronoun_number(tokens[0])
        # if antecedent is Number
        elif pos[0] == "NUM":
            if "یک" in tokens[0]:
                return "0"
            else:
                return "1"
        # if antecedent is adjective
        elif pos[0] == "CLITIC" or "ADJ" in pos[0]:
            return "0"
        elif "SING" in pos[0]:
            return "0"
        elif "PL" in pos[0]:
            return "1"
        else:
            return "2"
    else:
        if pos[0] == "NUM":
            if "یک" in tokens[0]:
                return "0"
            else:
                return "1"
        elif pos[0] == "CLITIC" or "ADJ" in pos[0] or pos[0] == "DET" or pos[0] == "DELM":
            if "SING" in pos[1]:
                return "0"
            elif "PL" in pos[1]:
                return "1"
            else:
                return "2"
        else:
            if "SING" in pos[0]:
                return "0"
            elif "PL" in pos[0]:
                return "1"
            else:
                return "2"


def calculate_number_rcdat(np_tokens, ante_f_token, ante_l_token, m_preprocessed_file):
    tokens = str(np_tokens).split(' ')
    pos = [i[6] for i in m_preprocessed_file if int(ante_f_token) <= int(i[0]) <= int(ante_l_token)]
    number = [i[11] for i in m_preprocessed_file if int(ante_f_token) <= int(i[0]) <= int(ante_l_token)]

    if len(pos) == 0 :
        return "0"

    if len(tokens) == 1:
        if is_pronoun(np_tokens) == "T":
            return pronoun_number(tokens[0])
        # if antecedent is Number
        elif pos[0] == "NUM":
            if "یک" in tokens[0]:
                return "0"
            else:
                return "1"
        # if antecedent is adjective
        elif pos[0] == "ADP" or "ADJ" in pos[0]:
            return "0"
        elif "Sing" in number[0]:
            return "0"
        elif "Plur" in number[0]:
            return "1"
        else:
            return "2"
    else:
        if pos[0] == "NUM":
            if "یک" in tokens[0]:
                return "0"
            else:
                return "1"
        elif pos[0] == "ADP" or "ADJ" in pos[0] or pos[0] == "DET" or pos[0] == "PUNCT":
            if "Sing" in number[1]:
                return "0"
            elif "Plur" in number[1]:
                return "1"
            else:
                return "2"
        else:
            if "Sing" in number[0]:
                return "0"
            elif "Plur" in number[0]:
                return "1"
            else:
                return "2"

def animacy_detection(np_tokens, f_token, l_token, m_preprocessed_file, mehr_ne, animacy_file, persian_names):
    title_lst = ["مدیرکل", "مدیر", "رییس", "رئیس‌ کل", "رئیس", "مقام", "عضو", "دبیر", "ریاست", "رهبر", "مهندس", "سپهبد",
                 "سرتیپ", "سردار",
                 "معاونت", "معاون", "منشی", "نائب", "دکتر", "روان شناس", "استاد", "سخنگوی", "نماینده", "سرگرد",
                 "سرلشگر", "سروان",
                 "روزنامه", "فعال", "تحلیلگر", "بازیکن", "گزارشگر", "قاضی", "خطیب", "سرهنگ", "میرزا", "شاه", "سلطان",
                 "تیمسار", "افسر",
                 "امام", "وزیر", "فرماندار", "استاندار", "بخشدار", "مدیر", "قائم", "معاونت", "کارشناس", "پزشک",
                 "حسابدار", "مربی", "مامور",
                 "کارمند", "دفتردار", "موسس", "مؤسس", "پژوهشگر", "دانشمند", "سردار", "سرهنگ", "فرمانده", "مترجم",
                 "متصدی", "شهردار",
                 "جانشین", "بنیان گذار", "خلبان", "امیر", "دریادار", "ناخدا", "سرتیپ", "شهید", "استاندار", "دریادار",
                 "معلم", "مجاهد", "مجری", "نویسنده", "مربی", "شاعر", "فیلسوف", "هنرمند", "صنعتگر", "کارگزار", "نقاش ",
                 "بازیگر", "قائم", "سرمربی",
                 "مسئول", "قهرمان"
                 ]
    heads = [s for s in m_preprocessed_file if int(f_token) <= int(s[0]) <= int(l_token)
             and (s[10] == "N" or s[10] == "Ne")]
    if len(heads) == 0:
        return "2"
    else:
        head = heads[0][7]
        anim = [i for i in animacy_file if i[0] == heads[0][0]]

        if anim[0][2] == "ANM":
            return "1"
        elif anim[0][2] == "IANM":
            return "0"
        else:
            is_person = [i for i in mehr_ne if
                         int(i[0]) <= int(f_token) <= int(i[1]) and int(i[0]) <= int(l_token) <= int(i[1]) and i[
                             6] == "PER"]
            pr_name = [i for i in persian_names if i[0] == head]
            if head in title_lst or len(is_person) > 0 or len(pr_name) > 0:
                return "1"
            else:
                return "0"


def animacy_detection_rcdat(np_tokens, f_token, l_token, m_preprocessed_file, mehr_ne, animacy_file, persian_names):
    title_lst = ["مدیرکل", "مدیر", "رییس", "رئیس‌ کل", "رئیس", "مقام", "عضو", "دبیر", "ریاست", "رهبر", "مهندس", "سپهبد",
                 "سرتیپ", "سردار",
                 "معاونت", "معاون", "منشی", "نائب", "دکتر", "روان شناس", "استاد", "سخنگوی", "نماینده", "سرگرد",
                 "سرلشگر", "سروان",
                 "روزنامه", "فعال", "تحلیلگر", "بازیکن", "گزارشگر", "قاضی", "خطیب", "سرهنگ", "میرزا", "شاه", "سلطان",
                 "تیمسار", "افسر",
                 "امام", "وزیر", "فرماندار", "استاندار", "بخشدار", "مدیر", "قائم", "معاونت", "کارشناس", "پزشک",
                 "حسابدار", "مربی", "مامور",
                 "کارمند", "دفتردار", "موسس", "مؤسس", "پژوهشگر", "دانشمند", "سردار", "سرهنگ", "فرمانده", "مترجم",
                 "متصدی", "شهردار",
                 "جانشین", "بنیان گذار", "خلبان", "امیر", "دریادار", "ناخدا", "سرتیپ", "شهید", "استاندار", "دریادار",
                 "معلم", "مجاهد", "مجری", "نویسنده", "مربی", "شاعر", "فیلسوف", "هنرمند", "صنعتگر", "کارگزار", "نقاش ",
                 "بازیگر", "قائم", "سرمربی",
                 "مسئول", "قهرمان"
                 ]
    heads = [s for s in m_preprocessed_file if int(f_token) <= int(s[0]) <= int(l_token)
             and (s[10] == "NOUN" or s[10] == "PROPN")]
    if len(heads) == 0:
        return "2"
    else:
        head = heads[0][7]
        anim = [i for i in animacy_file if i[0] == heads[0][0]]

        if anim[0][2] == "ANM":
            return "1"
        elif anim[0][2] == "IANM":
            return "0"
        else:
            is_person = [i for i in mehr_ne if
                         int(i[0]) <= int(f_token) <= int(i[1]) and int(i[0]) <= int(l_token) <= int(i[1]) and i[
                             6] == "PER"]
            pr_name = [i for i in persian_names if i[0] == head]
            if head in title_lst or len(is_person) > 0 or len(pr_name) > 0:
                return "1"
            else:
                return "0"


def gender_detection(animacy, np_tokens, f_token, l_token, m_preprocessed_file, persian_names, is_person):
    if Preprocess.Nlp_Preprocess.is_pronoun(np_tokens) == "T":
        if np_tokens == "من" or np_tokens == "تو" or np_tokens == "او" or np_tokens == "وی" or np_tokens == "ما" or np_tokens == "شما" or np_tokens == "ایشان" or \
                np_tokens == "خودم" or np_tokens == "خودش" or \
                np_tokens == "خودت" or np_tokens == "خودمان" or np_tokens == "خودتان" or np_tokens == "خودشان" or np_tokens == "خویش":
            return "2"
        elif np_tokens == "آن" or np_tokens == "این" or np_tokens == "آن" or np_tokens == "ان" or np_tokens == "خود" or np_tokens == "همین" or np_tokens == "همان":
            return "3"
    elif animacy == "0" or animacy == "2":
        return "3"
    # The Noun phrase is Animate
    else:
        # if np is animate and have title
        heads = [s for s in m_preprocessed_file if int(f_token) <= int(s[0]) <= int(l_token)
                 and (s[10] == "N" or s[10] == "Ne")]
        if len(heads) > 0:
            head = heads[0][7]
            title_lst = ["مدیرکل", "مدیر", "رییس", "رئیس‌ کل", "رئیس", "مقام", "عضو", "دبیر", "ریاست", "رهبر", "مهندس",
                         "سپهبد",
                         "سرتیپ", "سردار",
                         "معاونت", "معاون", "منشی", "نائب", "دکتر", "روان شناس", "استاد", "سخنگوی", "نماینده", "سرگرد",
                         "سرلشگر", "سروان",
                         "روزنامه", "فعال", "تحلیلگر", "بازیکن", "گزارشگر", "قاضی", "خطیب", "سرهنگ", "میرزا", "شاه",
                         "سلطان",
                         "تیمسار", "افسر",
                         "امام", "وزیر", "فرماندار", "استاندار", "بخشدار", "مدیر", "قائم", "معاونت", "کارشناس", "پزشک",
                         "حسابدار", "مربی", "مامور",
                         "کارمند", "دفتردار", "موسس", "مؤسس", "پژوهشگر", "دانشمند", "سردار", "سرهنگ", "فرمانده",
                         "مترجم",
                         "متصدی", "شهردار",
                         "جانشین", "بنیان گذار", "خلبان", "امیر", "دریادار", "ناخدا", "سرتیپ", "شهید", "استاندار",
                         "دریادار",
                         "معلم", "مجاهد", "مجری", "نویسنده", "مربی", "شاعر", "فیلسوف", "هنرمند", "صنعتگر", "کارگزار",
                         "نقاش ",
                         "بازیگر", "قائم", "سرمربی",
                         "مسئول", "قهرمان"
                         ]
            if head in title_lst:
                return "2"
            # if np is animate and not person
            elif is_person != "1":
                return "3"
            else:
                arr = np_tokens.split(' ')
                for item in arr:
                    q = [i for i in persian_names if i[0] == item]
                    if len(q) == 0:
                        continue
                    else:
                        if "male" in q[0][1]:
                            return "1"
                        else:
                            return "0"
                return "3"
        else:
            return "3"


def gender_detection_rcdat(animacy, np_tokens, f_token, l_token, m_preprocessed_file, persian_names, is_person):
    if Preprocess.Nlp_Preprocess.is_pronoun(np_tokens) == "T":
        if np_tokens == "من" or np_tokens == "تو" or np_tokens == "او" or np_tokens == "وی" or np_tokens == "ما" or np_tokens == "شما" or np_tokens == "ایشان" or \
                np_tokens == "خودم" or np_tokens == "خودش" or \
                np_tokens == "خودت" or np_tokens == "خودمان" or np_tokens == "خودتان" or np_tokens == "خودشان" or np_tokens == "خویش":
            return "2"
        elif np_tokens == "آن" or np_tokens == "این" or np_tokens == "آن" or np_tokens == "ان" or np_tokens == "خود" or np_tokens == "همین" or np_tokens == "همان":
            return "3"
    elif animacy == "0" or animacy == "2":
        return "3"
    # The Noun phrase is Animate
    else:
        # if np is animate and have title
        heads = [s for s in m_preprocessed_file if int(f_token) <= int(s[0]) <= int(l_token)
                 and (s[10] == "NOUN" or s[10] == "PROPN")]
        if len(heads) > 0:
            head = heads[0][7]
            title_lst = ["مدیرکل", "مدیر", "رییس", "رئیس‌ کل", "رئیس", "مقام", "عضو", "دبیر", "ریاست", "رهبر", "مهندس",
                         "سپهبد",
                         "سرتیپ", "سردار",
                         "معاونت", "معاون", "منشی", "نائب", "دکتر", "روان شناس", "استاد", "سخنگوی", "نماینده", "سرگرد",
                         "سرلشگر", "سروان",
                         "روزنامه", "فعال", "تحلیلگر", "بازیکن", "گزارشگر", "قاضی", "خطیب", "سرهنگ", "میرزا", "شاه",
                         "سلطان",
                         "تیمسار", "افسر",
                         "امام", "وزیر", "فرماندار", "استاندار", "بخشدار", "مدیر", "قائم", "معاونت", "کارشناس", "پزشک",
                         "حسابدار", "مربی", "مامور",
                         "کارمند", "دفتردار", "موسس", "مؤسس", "پژوهشگر", "دانشمند", "سردار", "سرهنگ", "فرمانده",
                         "مترجم",
                         "متصدی", "شهردار",
                         "جانشین", "بنیان گذار", "خلبان", "امیر", "دریادار", "ناخدا", "سرتیپ", "شهید", "استاندار",
                         "دریادار",
                         "معلم", "مجاهد", "مجری", "نویسنده", "مربی", "شاعر", "فیلسوف", "هنرمند", "صنعتگر", "کارگزار",
                         "نقاش ",
                         "بازیگر", "قائم", "سرمربی",
                         "مسئول", "قهرمان"
                         ]
            if head in title_lst:
                return "2"
            # if np is animate and not person
            elif is_person != "1":
                return "3"
            else:
                arr = np_tokens.split(' ')
                for item in arr:
                    q = [i for i in persian_names if i[0] == item]
                    if len(q) == 0:
                        continue
                    else:
                        if "male" in q[0][1]:
                            return "1"
                        else:
                            return "0"
                return "3"
        else:
            return "3"


def entity_Type(f_token, l_token, mehr_ne_file):
    entity = [i for i in mehr_ne_file if
              int(i[0]) <= int(f_token) <= int(i[1]) and int(i[0]) <= int(l_token) <= int(i[1])]

    if len(entity) > 0:
        en_type = entity[0][6]
        if en_type == "PER":
            return "1"
        elif en_type == "ORG":
            return "2"
        elif en_type == "LOC":
            return "3"
        elif en_type == "DAT":
            return "4"
        elif en_type == "EVE":
            return "5"
        else:
            return "6"
    else:
        return "0"


# endregion

# region relational's modules


def sentence_distance(samples, preprocessed_file):
    anaphora_sentence = int(preprocessed_file[int(samples[1])][1])
    if len(samples) == 7:
        antecedent_sentence = int(preprocessed_file[int(samples[4])][1])
    else:
        antecedent_sentence = int(preprocessed_file[int(samples[4])][1])
    return anaphora_sentence - antecedent_sentence


def token_distance(samples):
    if len(samples) == 7:
        return int(samples[1]) - int(samples[5]) - 1
    else:
        return int(samples[1]) - int(samples[4]) - 1


def number_agreement(first_num, second_num):
    if first_num == second_num == "1" or first_num == second_num == "0":
        return "1"
    else:
        return "0"


def animacy_agreement(first_anim, second_anim):
    if first_anim == second_anim == "1" or first_anim == second_anim == "0":
        return "1"
    else:
        return "0"


def gender_agreement(first_gen, second_gen):
    if first_gen == second_gen == "1" or first_gen == second_gen == "0" or first_gen == second_gen == "2" or\
            (first_gen == "1" and second_gen == "2") or \
            (first_gen == "0" and second_gen == "2") or \
            (first_gen == "2" and second_gen == "0") or \
            (first_gen == "2" and second_gen == "1"):
        return "1"
    else:
        return "0"


def person_agreement(first_per, second_per):
    if first_per == second_per == "1" or first_per == second_per == "2" or first_per == second_per == "3":
        return "1"
    else:
        return "0"


def subject_agreement(first_sbj, second_sbj):
    if first_sbj == second_sbj == "1" or first_sbj == second_sbj == "0":
        return "1"
    else:
        return "0"


def object_agreement(first_obj, second_obj):
    if first_obj == second_obj == "1" or first_obj == second_obj == "0":
        return "1"
    else:
        return "0"


def string_match(first_token, second_token):
    first_token = "".join(first_token.split())
    second_token = "".join(second_token.split())

    if first_token.rstrip() == second_token.rstrip():
        return "1"
    else:
        return "0"


def partial_string_match(first_token, second_token):
    x = is_pronoun(first_token)
    y = is_pronoun(second_token)
    if is_pronoun(first_token) == "T" and is_pronoun(second_token) == "T":
        if first_token.rstrip() == second_token.rstrip():
            return "1"
        else:
            return "0"
    elif is_pronoun(first_token) == "T" or is_pronoun(second_token) == "T":
        return "0"
    else:
        first_token = "".join(first_token.split())
        second_token = "".join(second_token.split())
        if (first_token.rstrip() in second_token.rstrip()) or (second_token.rstrip() in first_token.rstrip()):
            return "1"
        else:
            return "0"


def speaker_agreement(samples, preprocessed_file):
    lst_1 = ["گفت#گو", "#افزا", "نوشت#نویس", "داشت#دار", "داد#ده", "کرد#کن", "شد#شو", "نمود#نما", "برد#بر"]
    # lst_2 = ["اظهار", "ادامه", "بیان", "اضافه", "خاطرنشان", "تصریح", "توضیح", "شرح",
    #        "مدعی", "اعلام", "ابراز", "عنوان", "یادآور", "آغاز", "تاکید", "اشاره", "ادعا", "تشریح",
    #       "متذکر", "اذعان", "پایان", "خبر", "تفسیر", "تحلیل"]

    # region determine if   Anaphor is speaker:
    anaphor_is_subject = is_subject(samples[1], samples[2], preprocessed_file)

    if anaphor_is_subject == "0":
        anaphor_speaker = "F"
    else:
        s1_sentence = [i for i in preprocessed_file if int(samples[1]) == int(i[0])][0][1]
        q1 = [i for i in preprocessed_file if i[1] == s1_sentence and i[7] == ":" and int(i[0]) > int(samples[2])]
        if len(q1) == 0:
            anaphor_speaker = "F"
        else:
            q1_tok_idx = int(q1[0][0])
            q1_v1 = [i for i in preprocessed_file if int(i[0]) == q1_tok_idx - 1 and i[9] in lst_1]
            if len(q1_v1) > 0:
                anaphor_speaker = "T"
            else:
                anaphor_speaker = "F"
    # endregion

    # region determine if  antecedent is speaker:

    antecedent_is_subject = is_subject(samples[4], samples[5], preprocessed_file)
    if antecedent_is_subject == "0":
        antecedent_speaker = "F"
    else:
        s2_sentence = [i for i in preprocessed_file if int(samples[4]) == int(i[0])][0][1]
        q2 = [i for i in preprocessed_file if i[1] == s2_sentence and i[7] == ":" and int(i[0]) > int(samples[5])]
        if len(q2) == 0:
            antecedent_speaker = "F"
        else:
            q2_tok_idx = int(q2[0][0])
            q2_v1 = [i for i in preprocessed_file if int(i[0]) == q2_tok_idx - 1 and i[9] in lst_1]
            if len(q2_v1) > 0:
                antecedent_speaker = "T"
            else:
                antecedent_speaker = "T"
    # endregion

    if antecedent_speaker == "T" and anaphor_speaker == "T":
        return "1"
    else:
        return "0"

def speaker_agreement_rcdat(samples, preprocessed_file):
    lst_1 = ["گفت#گو", "#افزا", "نوشت#نویس", "داشت#دار", "داد#ده", "کرد#کن", "شد#شو", "نمود#نما", "برد#بر"]
    # lst_2 = ["اظهار", "ادامه", "بیان", "اضافه", "خاطرنشان", "تصریح", "توضیح", "شرح",
    #        "مدعی", "اعلام", "ابراز", "عنوان", "یادآور", "آغاز", "تاکید", "اشاره", "ادعا", "تشریح",
    #       "متذکر", "اذعان", "پایان", "خبر", "تفسیر", "تحلیل"]

    # region determine if   Anaphor is speaker:
    anaphor_is_subject = is_subject_rcdat(samples[1], samples[2], preprocessed_file)

    if anaphor_is_subject == "0":
        anaphor_speaker = "F"
    else:
        s1_sentence = [i for i in preprocessed_file if int(samples[1]) == int(i[0])][0][1]
        q1 = [i for i in preprocessed_file if i[1] == s1_sentence and i[7] == ":" and int(i[0]) > int(samples[2])]
        if len(q1) == 0:
            anaphor_speaker = "F"
        else:
            q1_tok_idx = int(q1[0][0])
            q1_v1 = [i for i in preprocessed_file if int(i[0]) == q1_tok_idx - 1 and i[9] in lst_1]
            if len(q1_v1) > 0:
                anaphor_speaker = "T"
            else:
                anaphor_speaker = "F"
    # endregion

    # region determine if  antecedent is speaker:

    antecedent_is_subject = is_subject_rcdat(samples[4], samples[5], preprocessed_file)
    if antecedent_is_subject == "0":
        antecedent_speaker = "F"
    else:
        s2_sentence = [i for i in preprocessed_file if int(samples[4]) == int(i[0])][0][1]
        q2 = [i for i in preprocessed_file if i[1] == s2_sentence and i[7] == ":" and int(i[0]) > int(samples[5])]
        if len(q2) == 0:
            antecedent_speaker = "F"
        else:
            q2_tok_idx = int(q2[0][0])
            q2_v1 = [i for i in preprocessed_file if int(i[0]) == q2_tok_idx - 1 and i[9] in lst_1]
            if len(q2_v1) > 0:
                antecedent_speaker = "T"
            else:
                antecedent_speaker = "T"
    # endregion

    if antecedent_speaker == "T" and anaphor_speaker == "T":
        return "1"
    else:
        return "0"

def three_distance(sen_distance):
    if (int(sen_distance)) < 3:
        return "1"
    else:
        return "0"


def same_distance(sen_distance):
    if (int(sen_distance)) == 0:
        return "1"
    else:
        return "0"


def head_string_match(samples, preprocessed_file):
        f_str_head = [j for j in preprocessed_file if int(samples[1]) <= int(j[0]) <= int(samples[2])
                      and (j[10] == "N" or j[10] == "Ne")]
        if len(f_str_head) > 0:
            first_str = f_str_head[0][7]
        else:
            first_str = "f_null"

        l_str_head = [i for i in preprocessed_file if int(samples[4]) <= int(i[0]) <= int(samples[5])
                      and (i[10] == "N" or i[10] == "Ne")]
        if len(l_str_head) > 0:
            last_str = l_str_head[0][7]
        else:
            last_str = "l_null"
        if last_str == first_str:
            return "1"
        else:
            return "0"

def head_string_match_rcdat(samples, preprocessed_file):
    f_str_head = [j for j in preprocessed_file if int(samples[1]) <= int(j[0]) <= int(samples[2])
                  and (j[10] == "NOUN" or j[10] == "PROPN")]
    if len(f_str_head) > 0:
        first_str = f_str_head[0][7]
    else:
        first_str = "f_null"

    l_str_head = [i for i in preprocessed_file if int(samples[4]) <= int(i[0]) <= int(samples[5])
                  and (i[10] == "NOUN" or i[10] == "PROPN")]
    if len(l_str_head) > 0:
        last_str = l_str_head[0][7]
    else:
        last_str = "l_null"
    if last_str == first_str:
        return "1"
    else:
        return "0"

# endregion

def mention_type(f_token, l_token, np_str, ne_file):
    is_proper = [i for i in ne_file if
                 int(i[0]) <= int(f_token) <= int(i[1]) and int(i[0]) <= int(l_token) <= int(i[1])]
    if is_pronoun(np_str) == "T":
        return "0"
    elif len(is_proper) > 0:
        return "1"
    else:
        return "2"

# endregion
