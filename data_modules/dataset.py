import torch
import string
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from polars.dataframe.frame import DataFrame
from sklearn.preprocessing import LabelEncoder
from transformers import RobertaTokenizer

from encoding.encoder import UPOSEncoder, InputEncoded
from utils.token import align_pos_ids_with_tokens, align_labels_with_tokens
from torch.nn.utils.rnn import pad_sequence


class CTINERDataset(Dataset):
    def __init__(self,
                 device,
                 tokenizer: RobertaTokenizer,
                 dataframe: DataFrame,
                 content_column="content",
                 label_column="labels"):
        self.device = device
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.special_tokens = {t:i for t,i in zip(self.tokenizer.all_special_tokens,self.tokenizer.all_special_ids)
                               if t != "<mask>"}
        self.char_vocab = defaultdict(lambda: 1,
                                      {char:idx for idx, char in
                                       enumerate(string.printable, start=max(self.special_tokens.values(), default=-1))})
        self.content_column = self.dataframe[content_column]
        self.label_column = self.dataframe[label_column]

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.label_list)

        self.upos_encoder = UPOSEncoder()

    def __len__(self):
        return len(self.content_column)

    def __getitem__(self, idx):
        content = self.content_column[idx]
        label = self.label_column[idx].to_list()
        return content, label

    def encode_labels(self, labels):
        return self.label_encoder.transform(labels).tolist()

    @property
    def label_list(self):
        unique_labels = self.dataframe.select(self.label_column.list.explode().unique())
        return unique_labels["labels"].to_list()

    def collate_fn(self, batch):
        contents, labels = zip(*batch)
        contents_encoded_padded, token_lists = self.encode_and_tokenize(contents)

        labels_encoded_padded = self.process_labels(token_lists, labels)
        upos_tags_encoded_padded = self.process_upos_tags(contents, token_lists)
        char_encoded_padded = self.process_char_encoding(token_lists)

        return (InputEncoded(content_encoded=contents_encoded_padded.to(self.device),
                             upos_encoded=upos_tags_encoded_padded.to(self.device),
                             char_encoded=char_encoded_padded.to(self.device)),
                labels_encoded_padded.to(self.device))

    def encode_and_tokenize(self, contents):
        encoded = self.tokenizer.batch_encode_plus(contents, padding=True, add_special_tokens=False, return_tensors="pt")
        token_lists = [self.tokenizer.tokenize(sentence) for sentence in contents]
        return encoded, token_lists

    def process_labels(self, token_lists, labels):
        aligned_labels = [align_labels_with_tokens(tokens, label) for tokens, label in zip(token_lists, labels)]
        labels_encoded = [self.encode_labels(lbl)for lbl in aligned_labels]
        label_tensors = [torch.tensor(lbl, dtype=torch.long) for lbl in labels_encoded]
        return pad_sequence(label_tensors, batch_first=True, padding_value=-100)

    def process_upos_tags(self, contents, token_lists):
        upos_tags_encoded = [self.upos_encoder.extract_upos_ids(sentence) for sentence in contents]
        aligned_upos_tags = [align_pos_ids_with_tokens(tokens, tag) for tokens, tag in
                             zip(token_lists, upos_tags_encoded)]
        upos_tensors = [torch.tensor(tag, dtype=torch.long) for tag in aligned_upos_tags]
        return pad_sequence(upos_tensors, batch_first=True, padding_value=self.special_tokens["<pad>"])

    def process_char_encoding(self, token_lists, max_word_len=15):
        char_sequences = [
            [[self.char_vocab.get(char, self.special_tokens["<unk>"]) for char in word][:max_word_len]
             for word in tokens]
            for tokens in token_lists
        ]
        char_sequences_padded = [
            [seq + [self.special_tokens["<pad>"]] * (max_word_len - len(seq)) for seq in sent]
            for sent in char_sequences
        ]

        char_tensors = [torch.tensor(sent, dtype=torch.long) for sent in char_sequences_padded]
        return pad_sequence(char_tensors, batch_first=True, padding_value=self.special_tokens["<pad>"])

    def to_dataloader(self,batch_size=16, shuffle=True):
        dataloader = DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)
        return dataloader
