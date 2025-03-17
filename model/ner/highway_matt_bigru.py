import torch
import torch.nn as nn
from torchcrf import CRF

from embedding.embedder import CustomEmbedder
from encoding.encoder import NERInputEncoded
from model.custom_layer.highway import Highway
from utils.models import initialize_weights


class MultiHeadAttentionBiLSTMCRFNER(nn.Module):
    def __init__(self,
                 emb_layer: CustomEmbedder,
                 num_classes,
                 bilstm_hidden_dim=768,
                 linear_hidden_dim=768,
                 highway_num_layers=1,
                 bilstm_num_layers=2,
                 num_heads=8):
        super(MultiHeadAttentionBiLSTMCRFNER, self).__init__()
        self.emb_layer = emb_layer
        # self.layernorm = nn.LayerNorm(self.emb_layer.output_dim)
        self.highway = Highway(self.emb_layer.output_dim, highway_num_layers, torch.nn.functional.relu)

        self.multihead_attention = nn.MultiheadAttention(embed_dim=self.emb_layer.output_dim,
                                                         num_heads=num_heads, batch_first=True)
        self.bilstm = nn.LSTM(
            self.emb_layer.output_dim, bilstm_hidden_dim, bidirectional=True,
            batch_first=True, num_layers=bilstm_num_layers, dropout=0.2)

        self.fc = nn.Linear(2 * linear_hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.2)
        self.crf = CRF(num_classes, batch_first=True)

        self.apply(initialize_weights)


    def forward(self, input_encoded: NERInputEncoded, labels=None):
        embeddings = self.emb_layer(input_encoded)
        # embeddings = self.layernorm(embeddings)
        highway_out = self.highway(embeddings)

        attn_out, _ = self.multihead_attention(highway_out, highway_out, highway_out)

        lstm_out, _ = self.bilstm(attn_out)
        emissions = self.fc(lstm_out)

        if labels is not None:
            labels = labels.clone()
            labels[labels == -100] = 0
            crf_mask = (labels != -100).bool()
            loss = -self.crf(emissions, labels, mask=crf_mask)
        else:
            loss = None
        return loss, self.crf.decode(emissions)