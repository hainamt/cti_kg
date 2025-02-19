import torch
import torch.nn as nn

from encoding.encoder import UPOS
from utils.constant import InputEncoded


class SecureBERTwithUPOSEmbedding(nn.Module):
    def __init__(self, emb_model, pos_emb_dim=64, char_emb_dim=32, char_vocab_size=100):
        super(SecureBERTwithUPOSEmbedding, self).__init__()
        self.emb_model = emb_model
        self.pos_emb_dim = pos_emb_dim
        self.upos_embedding = nn.Embedding(len(UPOS) + 50265 + 2, pos_emb_dim)
        self.char_embedding = nn.Embedding()

        for param in self.emb_model.parameters():
            param.requires_grad = False

    @property
    def output_dim(self):
        return self.model_hidden_dim + self.pos_emb_dim

    @property
    def model_hidden_dim(self):
        return self.model.config.hidden_size
    
    def get_secbert_embedding(self, input_ids, attention_mask):
        with torch.no_grad():
            secbert_emb = self.emb_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                ).last_hidden_state__
        return secbert_emb


    def forward(self, batch: InputEncoded):
        contents_encoded = batch.content_encoded
        upos_encoded = batch.upos_encoded

        upos_emb = self.upos_embedding(upos_encoded)
        secbert_emb = self.get_secbert_embedding(contents_encoded["input_ids"], contents_encoded["attention_mask"])

        return torch.cat((secbert_emb, upos_emb), dim=-1)

