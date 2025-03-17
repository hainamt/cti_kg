# class CTIREDataset(Dataset):
#     def __init__(self,
#                  device,
#                  dataframe: DataFrame,
#                  input_tokenizer: PreTrainedTokenizer,
#                  label_tokenizer: PreTrainedTokenizer,
#                  content_column : str = "content",
#                  re_label_column: str = "re_labels",
#                  ner_label_column: str = "labels"):
#         self.device = device
#         self.input_tokenizer = input_tokenizer
#         self.label_tokenizer = label_tokenizer
#         self.dataframe = dataframe
#         self.content_column = self.dataframe[content_column]
#         self.re_label_column = self.dataframe[re_label_column]
#         self.ner_label_column = self.dataframe[ner_label_column]

#     def __len__(self):
#         return len(self.content_column)

#     def __getitem__(self, idx):
#         content = self.content_column[idx]
#         re_label = self.re_label_column[idx]
#         ner_label = self.ner_label_column[idx]
#         return content, re_label, ner_label

#     @property
#     def ner_label_list(self):
#         unique_labels = self.dataframe.select(self.ner_label_column.list.explode().unique())
#         return unique_labels["labels"].to_list()

#     def process_re_labels(self, labels):
#         encoded_labels = self.label_tokenizer.batch_encode_plus(labels, padding=True,
#                                                                 add_special_tokens=False,
#                                                                 return_tensors="pt")
#         return encoded_labels

#     def process_input(self, tokens_list, ner_labels):
#         aligned_ner_labels = [align_ner_labels_with_tokens(token_lists, ner_labels)
#                               for token_lists, ner_labels in zip(tokens_list, ner_labels)]
#         text_input = [combine_token_ner(tokens, aligned_ner_label)
#                       for tokens, aligned_ner_label in zip(tokens_list, aligned_ner_labels)]
#         encoded_input = self.input_tokenizer.batch_encode_plus(text_input, padding=True,
#                                                                  return_tensors="pt")
#         return encoded_input

#     def collate_fn(self, batch):
#         contents, re_labels, ner_labels = zip(*batch)
#         token_lists = [self.input_tokenizer.tokenize(sentence) for sentence in contents]
#         returned_labels = self.process_re_labels(re_labels)
#         encoded_input = self.process_input(token_lists, ner_labels)
#         return (REInputEncoded(content_encoded=encoded_input.to(self.device)),
#                 returned_labels.to(self.device))

#     def to_dataloader(self, batch_size=16, shuffle=True):
#         dataloader = DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)
#         return dataloader
