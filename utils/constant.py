from enum import Enum, auto

class UPOS(Enum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return count + 3

    ADJ = auto()
    ADP = auto()
    ADV = auto()
    AUX = auto()
    CCONJ = auto()
    DET = auto()
    INTJ = auto()
    NOUN = auto()
    NUM = auto()
    PART = auto()
    PRON = auto()
    PROPN = auto()
    PUNCT = auto()
    SCONJ = auto()
    SYM = auto()
    VERB = auto()
    X = auto()