from typing import NamedTuple

import stanza
from torch import Tensor
from transformers import BatchEncoding

from utils.constant import UPOS


class InputEncoded(NamedTuple):
    content_encoded: BatchEncoding
    upos_encoded: Tensor
    char_encoded: Tensor


def upos_to_index(upos: UPOS):
    return upos.value

def index_to_upos(index: int) -> UPOS:
    for upos in UPOS:
        if upos.value == index:
            return upos
    raise ValueError(f"No UPOS corresponds to index {index}")


class UPOSEncoder:
    def __init__(self, processors="tokenize,pos"):
        self.pos_tagger = stanza.Pipeline('en', processors=processors)

    def extract_upos_tags(self, text: str):
        doc = self.pos_tagger(text)
        return [word.upos for sent in doc.sentences for word in sent.words]

    def extract_upos_ids(self, text: str):
        upos_tags = self.extract_upos_tags(text)
        return [UPOS[tag].value for tag in upos_tags if tag in UPOS.__members__]
