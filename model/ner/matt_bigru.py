import torch
import torch.nn as nn
from torchcrf import CRF

from embedding.embedder import CustomEmbedder
from encoding.encoder import NERInputEncoded
from utils.models import initialize_weights


class MultiHeadAttentionBiGruCRFNER(nn.Module):
    def __init__(self,
                 emb_layer: CustomEmbedder,
                 num_classes,
                 gru_hidden_dim=768,
                 linear_hidden_dim=768,
                 gru_num_layers=2,
                 num_heads=8):
        super(MultiHeadAttentionBiGruCRFNER, self).__init__()
        self.emb_layer = emb_layer
        self.multihead_attention = nn.MultiheadAttention(embed_dim=self.emb_layer.output_dim,
                                                         num_heads=num_heads, batch_first=True)
        self.gru = nn.GRU(
            self.emb_layer.output_dim, gru_hidden_dim, bidirectional=True,
            batch_first=True, num_layers=gru_num_layers, dropout=0.2)
        self.fc = nn.Linear(2 * linear_hidden_dim, num_classes)
        self.crf = CRF(num_classes, batch_first=True)
        self.apply(initialize_weights)

    def forward(self, input_encoded: NERInputEncoded, labels=None):
        embeddings = self.emb_layer(input_encoded)
        attn_out, _ = self.multihead_attention(embeddings, embeddings, embeddings)
        gru_out, _ = self.gru(attn_out)
        emissions = self.fc(gru_out)

        if labels is not None:
            labels = labels.clone()
            labels[labels == -100] = 0
            crf_mask = (labels != -100).bool()
            loss = -self.crf(emissions, labels, mask=crf_mask)
        else:
            loss = None
        return loss, self.crf.decode(emissions)