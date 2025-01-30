import numpy as np

from src.datautils.token_dataset_bressay import TOKEN_POSITION_CHAR_LEVEL_DICT, TOKEN_IS_CROSS_CHAR_LEVEL_DICT, \
    TOKEN_IS_READABLE_CHAR_LEVEL_DICT, EOS_STR_TOKEN, SOS_STR_TOKEN


def transcript_tranforme_base(dictionary, str_to_transform, use_delimiter_tokens=False):
    """
    """
    labels = []

    if use_delimiter_tokens:
        labels.append(dictionary.get("<sos>"))

    for c in str_to_transform:
        if c not in dictionary:
            print("Text unknow char in dictionnary : " + str(c))
            # return -1
        else:
            labels.append(dictionary.get(c))

    if use_delimiter_tokens:
        labels.append(dictionary.get("<eos>"))

    return labels


def append_preds(pg_preds, line_preds):
    for i, lp in enumerate(line_preds):
        if lp is not None:
            pg_preds[i].append(lp)
    return pg_preds


def convert_int_to_chars(indices, char_list, break_on_eos=True):
    """
    This function applies a int to character dictionary to every int in a sequence

    Parameters
    ----------
    indices: int array
        A sequence of integers to decode
    char_list: char array
        A list of characters
    break_on_eos: bool
        Whether or not (default is set to true) to stop conversion when encountering a <eos> token

    Returns
    -------
    char_sequence: char array
        The decoded sequence
    """

    chars_sequence = ""

    for char_index in indices:
        try:
            c = char_list[char_index]
            if c == "<eos>" and break_on_eos:
                break
            chars_sequence += c
        except Exception as e:
            chars_sequence += "Error char index"

    return chars_sequence


def ctc_best_path_one(top_prob, char_list, blank_index):
    """
    """
    # Remove the duplicated characters
    sequence_without_duplicates = []
    previous_index = -1
    for index in top_prob:
        if index != previous_index:
            sequence_without_duplicates.append(index)
            previous_index = index

    # Remove the blanks
    sequence = []
    for index in sequence_without_duplicates:
        if index != blank_index:
            sequence.append(index)

    # Convert to characters
    char_sequence = convert_int_to_chars(sequence, char_list)

    return char_sequence


def best_path(probabilities, char_list, eos_index, blank_index=0):
    """
    """

    max_len = len(probabilities)

    sequence_raw = []

    # Get the characters with the highest values
    for j in range(max_len):
        frame = probabilities[j]
        predicted_char = np.argmax(frame)
        if predicted_char == eos_index:  #"<eos>":
            break
        sequence_raw.append(predicted_char)

    # # Remove the blanks
    # sequence = []
    # for char in sequence_raw:
    #     if char != blank_index:
    #         sequence.append(char)

    # Convert to characters
    char_sequence = convert_int_to_chars(sequence_raw, char_list)

    return char_sequence


def read_all_txt(path_file, add_space_before_after=True):
    content_str = ""

    # , encoding="utf-8"
    #with open(path_file) as file:
    with open(path_file, encoding="utf-8") as file:
        all_lines = file.readlines()

        for one_l in all_lines:
            one_l = one_l.replace("\n", "")

            if add_space_before_after:
                one_l = " " + one_l + " "

            content_str += one_l

    return content_str


def compute_index_tag_for_seq2seq_v2_char_level(txt_input):

    # stats_special_txt_2d = {
    #     "total_cross": 0,
    #     "total_over": 0,
    #     "total_sub": 0,
    #     "total_nonreadable_cross": 0,
    #     "total_nonreadable": 0,
    # }
    # stats_special_txt_4d = {
    #     "total_cross": 0,
    #     "total_over": 0,
    #     "total_sub": 0,
    #     "total_nonreadable_cross": 0,
    #     "total_nonreadable": 0,
    # }

    # Initialisation with start of sequence token
    ind_pos = [TOKEN_POSITION_CHAR_LEVEL_DICT[SOS_STR_TOKEN]]
    ind_is_cross = [TOKEN_IS_CROSS_CHAR_LEVEL_DICT[SOS_STR_TOKEN]]
    ind_is_readable = [TOKEN_IS_READABLE_CHAR_LEVEL_DICT[SOS_STR_TOKEN]]

   #  ind_tag_all = [ALL_TOKENS_CHAR_LEVEL_DICT[SOS_STR_TOKEN]]

    index_char = 0
    for one_char in txt_input:

        if one_char == "#":
            ind_pos.append(TOKEN_POSITION_CHAR_LEVEL_DICT["over"])
            ind_is_cross.append(TOKEN_IS_CROSS_CHAR_LEVEL_DICT["no_cross"])
            ind_is_readable.append(TOKEN_IS_READABLE_CHAR_LEVEL_DICT["readable"])

            #ind_tag_all.append(ALL_TOKENS_CHAR_LEVEL_DICT["over"])

            # stats_special_txt_4d["total_over"] += 1
            # stats_special_txt_2d["total_over"] += 1
        elif one_char == "$":
            ind_pos.append(TOKEN_POSITION_CHAR_LEVEL_DICT["sub"])
            ind_is_cross.append(TOKEN_IS_CROSS_CHAR_LEVEL_DICT["no_cross"])
            ind_is_readable.append(TOKEN_IS_READABLE_CHAR_LEVEL_DICT["readable"])

            # ind_tag_all.append(ALL_TOKENS_CHAR_LEVEL_DICT["sub"])

            # stats_special_txt_4d["total_sub"] += 1
            # stats_special_txt_2d["total_sub"] += 1
        elif one_char == "-":
            # Check is double dash: --
            previous_index = index_char - 1
            previous_is_cross = False

            if previous_index > 0:
                if txt_input[previous_index] == "-":
                    previous_is_cross = True

            next_index = index_char + 1
            next_is_cross = False

            if next_index < len(txt_input):
                if txt_input[next_index] == "-":
                    next_is_cross = True

            ind_pos.append(TOKEN_POSITION_CHAR_LEVEL_DICT["main"])
            ind_is_readable.append(TOKEN_IS_READABLE_CHAR_LEVEL_DICT["readable"])

            # Is the cross tag
            if previous_is_cross or next_is_cross:
                ind_is_cross.append(TOKEN_IS_CROSS_CHAR_LEVEL_DICT["cross"])
                # ind_tag_all.append(ALL_TOKENS_CHAR_LEVEL_DICT["cross"])

                # stats_special_txt_4d["total_cross"] += 1
                # stats_special_txt_2d["total_cross"] += 1
            # Is the - character
            else:
                ind_is_cross.append(TOKEN_IS_CROSS_CHAR_LEVEL_DICT["no_cross"])
                # ind_tag_all.append(ALL_TOKENS_CHAR_LEVEL_DICT["text"])

        elif one_char == "x":
            # Check is neighbourg is x or -
            previous_index = index_char - 1
            previous_is_cross = False
            previous_is_x = False

            if previous_index > 0:
                if txt_input[previous_index] == "-":
                    previous_is_cross = True
                elif txt_input[previous_index] == "x":
                    previous_is_x = True

            next_index = index_char + 1
            next_is_cross = False
            next_is_x = False

            if next_index < len(txt_input):
                if txt_input[next_index] == "-":
                    next_is_cross = True
                elif txt_input[next_index] == "x":
                    next_is_x = True

            ind_pos.append(TOKEN_POSITION_CHAR_LEVEL_DICT["main"])
            ind_is_cross.append(TOKEN_IS_CROSS_CHAR_LEVEL_DICT["no_cross"])

            # 3 Cases for tag --xx---
            # -xx
            if previous_is_cross and next_is_x:
                ind_is_readable.append(TOKEN_IS_READABLE_CHAR_LEVEL_DICT["non_readable_cross"])
                # ind_tag_all.append(ALL_TOKENS_CHAR_LEVEL_DICT["non_readable_cross"])

                # stats_special_txt_4d["total_nonreadable_cross"] += 1
                # stats_special_txt_2d["total_nonreadable_cross"] += 1
            # xxx
            elif previous_is_x and next_is_x:
                ind_is_readable.append(TOKEN_IS_READABLE_CHAR_LEVEL_DICT["non_readable_cross"])
                # ind_tag_all.append(ALL_TOKENS_CHAR_LEVEL_DICT["non_readable_cross"])

                # stats_special_txt_4d["total_nonreadable_cross"] += 1
                # stats_special_txt_2d["total_nonreadable_cross"] += 1
            # xx-
            elif previous_is_x and next_is_cross:
                ind_is_readable.append(TOKEN_IS_READABLE_CHAR_LEVEL_DICT["non_readable_cross"])
                # ind_tag_all.append(ALL_TOKENS_CHAR_LEVEL_DICT["non_readable_cross"])

                # stats_special_txt_4d["total_nonreadable_cross"] += 1
                # stats_special_txt_2d["total_nonreadable_cross"] += 1
            # is x character
            else:
                ind_is_readable.append(TOKEN_IS_READABLE_CHAR_LEVEL_DICT["readable"])

                # ind_tag_all.append(ALL_TOKENS_CHAR_LEVEL_DICT["text"])

        elif one_char == "@" or one_char == "?":
            # Simple check is neighbourg is @ or ?
            previous_index = index_char - 1
            previous_is_non_readable_char = False

            if previous_index > 0:
                if txt_input[previous_index] == "@" or txt_input[previous_index] == "?":
                    previous_is_non_readable_char = True

            next_index = index_char + 1
            next_is_non_readable_char = False

            if next_index < len(txt_input):
                if txt_input[next_index] == "@" or txt_input[next_index] == "?":
                    next_is_non_readable_char = True

            ind_pos.append(TOKEN_POSITION_CHAR_LEVEL_DICT["main"])
            ind_is_cross.append(TOKEN_IS_CROSS_CHAR_LEVEL_DICT["no_cross"])

            # Is the non readable tag
            if previous_is_non_readable_char or next_is_non_readable_char:
                ind_is_readable.append(TOKEN_IS_READABLE_CHAR_LEVEL_DICT["non_readable"])
                # ind_tag_all.append(ALL_TOKENS_CHAR_LEVEL_DICT["non_readable"])

                # stats_special_txt_4d["total_nonreadable"] += 1
                # stats_special_txt_2d["total_nonreadable"] += 1
            # Is the @ or ? character
            else:
                ind_is_readable.append(TOKEN_IS_READABLE_CHAR_LEVEL_DICT["readable"])
                # ind_tag_all.append(ALL_TOKENS_CHAR_LEVEL_DICT["text"])
        # "normal" character
        else:
            ind_pos.append(TOKEN_POSITION_CHAR_LEVEL_DICT["main"])
            ind_is_cross.append(TOKEN_IS_CROSS_CHAR_LEVEL_DICT["no_cross"])
            ind_is_readable.append(TOKEN_IS_READABLE_CHAR_LEVEL_DICT["readable"])

            # ind_tag_all.append(ALL_TOKENS_CHAR_LEVEL_DICT["text"])

        index_char += 1

    # Add End Of Sequence
    # ind_tag_all.append(ALL_TOKENS_CHAR_LEVEL_DICT[EOS_STR_TOKEN])
    ind_pos.append(TOKEN_POSITION_CHAR_LEVEL_DICT[EOS_STR_TOKEN])
    ind_is_cross.append(TOKEN_IS_CROSS_CHAR_LEVEL_DICT[EOS_STR_TOKEN])
    ind_is_readable.append(TOKEN_IS_READABLE_CHAR_LEVEL_DICT[EOS_STR_TOKEN])

    gt_dict = {
        "ind_pos": ind_pos,
        "ind_is_cross": ind_is_cross,
        "ind_is_readable": ind_is_readable,
        # "ind_tag_all": ind_tag_all,
        # "stats_special_txt_2d": stats_special_txt_2d,
        # stats_special_txt_4d": stats_special_txt_4d,
    }

    return gt_dict


def gt_transform_seq2seq_txt_multidecoders_use_tag_in_txt_char_level(path_label, dictionary_label):
    label_str_raw = read_all_txt(path_label, add_space_before_after=True)

    # Tag are in encoder text and decoder text
    label_ind_enc = transcript_tranforme_base(dictionary_label, label_str_raw, use_delimiter_tokens=False)
    label_ind_dec = transcript_tranforme_base(dictionary_label, label_str_raw, use_delimiter_tokens=True)

    # Tags are in special decoder(s)
    index_format_dict = compute_index_tag_for_seq2seq_v2_char_level(label_str_raw)

    gt_dict = {
        "label_str_raw": label_str_raw,
        "label_ind_enc": label_ind_enc,
        "label_ind_dec": label_ind_dec,
        "ind_pos": index_format_dict["ind_pos"],
        "ind_is_cross": index_format_dict["ind_is_cross"],
        "ind_is_readable": index_format_dict["ind_is_readable"],
        # "ind_tag_all": index_format_dict["ind_tag_all"],
       #  "stats_special_txt_2d": index_format_dict["stats_special_txt_2d"],
        # "stats_special_txt_4d": index_format_dict["stats_special_txt_4d"]
    }

    return gt_dict
