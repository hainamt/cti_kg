from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from transformers import BatchEncoding
from encoding.encoder import NERInputEncoded


class BaseEmbedder(nn.Module, ABC):
    @abstractmethod
    def get_input(self, batch: NERInputEncoded):
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    @property
    def output_dim(self) -> int:
        pass


class SecBertEmbedder(BaseEmbedder):
    def __init__(self, emb_model):
        super().__init__()
        self.emb_model = emb_model
        for param in self.emb_model.parameters():
            param.requires_grad = False

    def get_input(self, batch: NERInputEncoded):
        return batch.content_encoded

    def forward(self, input_encoded: BatchEncoding):
        with torch.no_grad():
            return self.emb_model(
                input_ids=input_encoded["input_ids"],
                attention_mask=input_encoded["attention_mask"]
            ).last_hidden_state

    @property
    def output_dim(self) -> int:
        return self.emb_model.config.hidden_size


class UPOSEmbedder(BaseEmbedder):
    def __init__(self, upos_vocab, emb_dim):
        super().__init__()
        self.embedding = nn.Embedding(upos_vocab, emb_dim)
        self.emb_dim = emb_dim

    def get_input(self, batch: NERInputEncoded):
        return batch.upos_encoded

    def forward(self, upos_encoded: torch.Tensor):
        return self.embedding(upos_encoded)

    @property
    def output_dim(self) -> int:
        return self.emb_dim


class CharCNNEmbedder(BaseEmbedder):
    def __init__(self, char_vocab_size, char_emb_dim=32, num_char_filters=50,
                 kernel_size=3, max_word_len=15):
        super().__init__()
        self.max_word_len = max_word_len
        self.num_char_filters = num_char_filters

        self.char_embedding = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=0)
        self.char_cnn = nn.Conv1d(in_channels=char_emb_dim, out_channels=num_char_filters,
                                  kernel_size=kernel_size, padding=1)

    def get_input(self, batch: NERInputEncoded) -> torch.Tensor:
        return batch.char_encoded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, word_len = x.shape
        char_emb = self.char_embedding(x)
        char_emb = char_emb.view(-1, word_len, char_emb.size(-1)).permute(0, 2, 1)
        char_cnn_out = F.relu(self.char_cnn(char_emb))
        char_cnn_out, _ = torch.max(char_cnn_out, dim=-1)
        return char_cnn_out.view(batch_size, seq_len, -1)

    @property
    def output_dim(self) -> int:
        return self.num_char_filters


class CustomEmbedder(nn.Module):
    def __init__(self, embedders: List[BaseEmbedder]):
        super().__init__()
        self.embedders = nn.ModuleList(embedders)

    def forward(self, batch: NERInputEncoded) -> torch.Tensor:
        embeddings = [embedder(embedder.get_input(batch)) for embedder in self.embedders]
        return torch.cat(embeddings, dim=-1)

    @property
    def output_dim(self) -> int:
        return sum(embedder.output_dim for embedder in self.embedders)