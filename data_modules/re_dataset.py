from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset
from polars.dataframe.frame import DataFrame


class CTIREDataset(Dataset):
    def __init__(self,
                 device,
                 dataframe: DataFrame,
                 tokenizer: PreTrainedTokenizer,
                 content_column : str = "content",
                 token_column: str = "tokens",
                 re_label_column: str = "re_labels",
                 ner_label_column: str = "labels",
                 max_seq_length: int = 256):
        self.device = device
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.content_column = self.dataframe[content_column]
        self.token_column = self.dataframe[token_column]
        self.re_label_column = self.dataframe[re_label_column]
        self.ner_label_column = self.dataframe[ner_label_column]
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.content_column)

    def __getitem__(self, idx):
        text_tokens = self.token_column[idx]
        re_label = self.re_label_column[idx]
        ner_label = self.ner_label_column[idx]

        text_input = combine_token_ner(text_tokens, ner_label)

        encoded_input = self.tokenizer(text_input, padding="max_length",
                                       truncation=True, max_length=self.max_seq_length,
                                       return_tensors="pt")
        encoded_label = self.tokenizer(re_label, padding="max_length",
                                       truncation=True, max_length=self.max_seq_length,
                                       return_tensors="pt")
        returned_labels = encoded_label["input_ids"].clone()
        returned_labels[returned_labels == self.tokenizer.pad_token_id] = -100

        tokenized_input = self.tokenizer.tokenize(text_input)
        tokenized_label = self.tokenizer.tokenize(re_label)
        return {
            "text_input": text_input,
            "text_label": re_label,
            "tokenized_input": tokenized_input,
            "tokenized_label": tokenized_label,
            "input_ids": encoded_input["input_ids"].squeeze(0),
            "attention_mask": encoded_input["attention_mask"].squeeze(0),
            "decoder_input_ids": encoded_label["input_ids"].squeeze(0),
            "decoder_attention_mask": encoded_label["attention_mask"].squeeze(0),
            "labels": returned_labels.squeeze(0),
        }