# Constants values
TAG_OVER = "##"
TAG_SUB = "$$"
TAG_ALLCROSS = "--"
TAG_NONREADABLE = "@@???@@"
TAG_CROSS_NONREADABLE = "--xxx--"
TAG_OVER_CROSS_NONREADABLE = "##--xxx--##"
TAG_SUB_CROSS_NON_READABLE = "$$--xxx--$$"
TAG_OVER_NON_READABLE = "##@@???@@##"
TAG_SUB_NON_READABLE = "$$@@???@@$$"

MAX_LEN_LINES_BRESSAY = 200
# MAX_WORDS_LINE_BRESSAY = 30  # Max Nb words (split space) Bressay

PAD_STR_TOKEN = "<pad>"
SOS_STR_TOKEN = "<sos>"
EOS_STR_TOKEN = "<eos>"
BLANK_STR_TOKEN = "<BLANK>"

TOKEN_POSITION_CHAR_LEVEL_DICT = {
    PAD_STR_TOKEN: 0,
    SOS_STR_TOKEN: 1,
    "over": 2,
    "main": 3,
    "sub": 4,
    EOS_STR_TOKEN: 5
}

TOKEN_IS_CROSS_CHAR_LEVEL_DICT = {
    PAD_STR_TOKEN: 0,
    SOS_STR_TOKEN: 1,
    "cross": 2,
    "no_cross": 3,
    EOS_STR_TOKEN: 4
}

TOKEN_IS_READABLE_CHAR_LEVEL_DICT = {
    PAD_STR_TOKEN: 0,
    SOS_STR_TOKEN: 1,
    "readable": 2,
    "non_readable_cross": 3,    # --xxx--
    "non_readable": 4,          # @@???@@
    EOS_STR_TOKEN: 5
}
TOKEN_POSITION_CHAR_LEVEL_LIST = [PAD_STR_TOKEN, SOS_STR_TOKEN, "#", " ", "$", EOS_STR_TOKEN]

TOKEN_POSITION_DICT = {
    PAD_STR_TOKEN: 0,
    SOS_STR_TOKEN: 1,
    "over_begin": 2,
    "over_all": 3,
    "over_end": 4,
    "main": 5,
    "sub_begin": 6,
    "sub_all": 7,
    "sub_end": 8,
    EOS_STR_TOKEN: 9
}

TOKEN_POSITION_LIST = [PAD_STR_TOKEN, SOS_STR_TOKEN, "over_begin", "over_all", "over_end", "main",
                       "sub_begin", "sub_all", "sub_end", EOS_STR_TOKEN]


TOKEN_IS_CROSS_DICT_V1 = {
    PAD_STR_TOKEN: 0,
    SOS_STR_TOKEN: 1,
    "yes": 2,
    "no": 3,
    EOS_STR_TOKEN: 4
}
#
# TOKEN_IS_CROSS_LIST = [PAD_STR_TOKEN, SOS_STR_TOKEN, "yes", "no", EOS_STR_TOKEN]
TOKEN_IS_CROSS_CHAR_LEVEL_DICT = {
    PAD_STR_TOKEN: 0,
    SOS_STR_TOKEN: 1,
    "cross": 2,
    "no_cross": 3,
    EOS_STR_TOKEN: 4
}

TOKEN_IS_CROSS_CHAR_LEVEL_LIST = [PAD_STR_TOKEN, SOS_STR_TOKEN, "-", " ", EOS_STR_TOKEN]

TOKEN_IS_CROSS_DICT = {
    PAD_STR_TOKEN: 0,
    SOS_STR_TOKEN: 1,
    "begin_cross": 2,
    "all_cross": 3,
    "end_cross": 4,
    "no_cross": 5,
    EOS_STR_TOKEN: 6
}

TOKEN_IS_CROSS_LIST = [PAD_STR_TOKEN, SOS_STR_TOKEN, "begin_cross", "all_cross", "end_cross", "no_cross", EOS_STR_TOKEN]

TOKEN_IS_READABLE_CHAR_LEVEL_DICT = {
    PAD_STR_TOKEN: 0,
    SOS_STR_TOKEN: 1,
    "readable": 2,
    "non_readable_cross": 3,    # --xxx--
    "non_readable": 4,          # @@???@@
    EOS_STR_TOKEN: 5
}

TOKEN_IS_READABLE_CHAR_LEVEL_LIST = [PAD_STR_TOKEN, SOS_STR_TOKEN, " ", "x", "@", EOS_STR_TOKEN]

TOKEN_IS_READABLE_DICT = {
    PAD_STR_TOKEN: 0,
    SOS_STR_TOKEN: 1,
    "yes": 2,
    "no": 3,
    EOS_STR_TOKEN: 4
}

TOKEN_IS_READABLE_LIST = [PAD_STR_TOKEN, SOS_STR_TOKEN, "yes", "no", EOS_STR_TOKEN]
