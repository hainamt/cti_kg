import torch
import torch.nn as nn
from torchcrf import CRF

from embedding.embedder import CustomEmbedder
from encoding.encoder import NERInputEncoded
from utils.models import initialize_weights


class TransformerBiGruAttentionCRFNER(nn.Module):
    def __init__(self,
                 emb_layer: CustomEmbedder,
                 num_classes,
                 trans_encoder_ffdim= 512,
                 bigru_hidden_dim=256,
                 linear_hidden_dim=256,
                 transformer_num_layers=2,
                 bigru_num_layers=2,
                 num_heads=8):
        super(TransformerBiGruAttentionCRFNER, self).__init__()
        self.emb_layer = emb_layer
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.emb_layer.output_dim, nhead=num_heads,batch_first=True,
                                       dim_feedforward=trans_encoder_ffdim, dropout=0.2),
            num_layers=transformer_num_layers)
        self.bigru = nn.GRU(
            self.emb_layer.output_dim, bigru_hidden_dim, bidirectional=True,
            batch_first=True, num_layers=bigru_num_layers, dropout=0.3)
        self.attention = nn.MultiheadAttention(embed_dim=2 * bigru_hidden_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(2 * linear_hidden_dim, num_classes)
        self.crf = CRF(num_classes, batch_first=True)
        self.dropout = nn.Dropout(0.3)

        self.apply(initialize_weights)

    def forward(self, input_encoded: NERInputEncoded, labels=None):
        embeddings = self.emb_layer(input_encoded)
        embeddings = self.layernorm(embeddings)
        transformer_out = self.transformer_encoder(embeddings)
        gru_out, _ = self.bigru(transformer_out)
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        emissions = self.fc(attn_out)

        if labels is not None:
            labels = labels.clone()
            labels[labels == -100] = 0
            crf_mask = (labels != -100).bool()
            loss = -self.crf(emissions, labels, mask=crf_mask)
        else:
            loss = None
        return loss, self.crf.decode(emissions)
