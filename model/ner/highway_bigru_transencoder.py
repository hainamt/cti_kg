import torch
import torch.nn as nn
from torchcrf import CRF

from embedding.embedder import CustomEmbedder
from encoding.encoder import InputEncoded
from model.custom_layer.highway import Highway
from utils.models import initialize_weights


class BiGruTransformerCRFNER(nn.Module):
    def __init__(self,
                 emb_layer: CustomEmbedder,
                 num_classes,
                 bigru_hidden_dim=256,
                 linear_hidden_dim=256,
                 highway_num_layers=2,
                 bigru_num_layers=2,
                 transformer_num_layers=2,
                 num_heads=8):
        super(BiGruTransformerCRFNER, self).__init__()
        self.emb_layer = emb_layer
        self.layernorm = nn.LayerNorm(self.emb_layer.output_dim)
        self.highway = Highway(self.emb_layer.output_dim, highway_num_layers, torch.nn.functional.relu)
        self.bigru = nn.GRU(
            self.emb_layer.output_dim, bigru_hidden_dim, bidirectional=True,
            batch_first=True, num_layers=bigru_num_layers, dropout=0.3)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=2 * bigru_hidden_dim, nhead=num_heads, dim_feedforward=512, dropout=0.2),
            num_layers=transformer_num_layers)

        self.fc = nn.Linear(2 * linear_hidden_dim, num_classes)
        self.crf = CRF(num_classes, batch_first=True)
        self.dropout = nn.Dropout(0.3)

        self.apply(initialize_weights)

    def forward(self, input_encoded: InputEncoded, labels=None):
        embeddings = self.emb_layer(input_encoded)
        embeddings = self.layernorm(embeddings)
        highway_out = self.highway(embeddings)
        gru_out, _ = self.bigru(highway_out)
        transformer_out = self.transformer_encoder(gru_out)
        emissions = self.fc(transformer_out)

        if labels is not None:
            labels = labels.clone()
            labels[labels == -100] = 0
            crf_mask = (labels != -100).bool()
            loss = -self.crf(emissions, labels, mask=crf_mask)
        else:
            loss = None
        return loss, self.crf.decode(emissions)
