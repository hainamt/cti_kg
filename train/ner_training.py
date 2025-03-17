import torch
import polars as pl
from torch import optim
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from transformers import RobertaModel, RobertaTokenizer

from data_modules.ner_dataset import CTINERDataset
from embedding.embedder import CustomEmbedder, SecBertEmbedder, UPOSEmbedder, CharCNNEmbedder
from encoding.encoder import UPOSEncoder
from model.ner.matt_bigru import MultiHeadAttentionBiGruCRFNER
from utils.constant import UPOS
from utils.dataframe import to_df, train_val_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def align_predictions_and_labels(labels, predictions):
    align_preds = []
    align_labels = []
    for seq_labels, seq_preds in zip(labels, predictions):
        valid_labels = [l for l in seq_labels if l != -100]
        valid_preds = seq_preds[:len(valid_labels)]
        align_labels.extend(valid_labels)
        align_preds.extend(valid_preds)
    return align_preds, align_labels

def stepping(model, data_loader, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(is_train):
        for batch in tqdm(data_loader, desc="Training" if is_train else "Evaluating", leave=False, dynamic_ncols=True):
            input_encoded, labels = batch
            if is_train:
                optimizer.zero_grad()
            loss, predictions = model(input_encoded, labels=labels)
            total_loss += loss.item()
            if is_train:
                loss.backward()
                optimizer.step()

            all_preds.extend(predictions)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    aligned_preds, aligned_labels = align_predictions_and_labels(all_preds, all_labels)
    f1 = f1_score(aligned_labels, aligned_preds, average='macro')
    acc = accuracy_score(aligned_labels, aligned_preds)
    return avg_loss, f1, acc

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Loaded checkpoint from epoch {start_epoch}")
    return model, optimizer, scheduler, start_epoch

def train_model(model,
                train_loader, val_loader,
                device, optimizer, scheduler,
                num_epochs,
                save_per, save_path,
                model_name):
    for epoch in range(num_epochs):
        train_loss, train_f1, train_acc = stepping(model, train_loader, device, optimizer)
        val_loss, val_f1, val_acc = stepping(model, val_loader, device)
        print(f"Epoch [{epoch+1}/{num_epochs}]:")
        print(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}, Val Acc: {val_acc:.4f}")

        if (epoch + 1) % save_per == 0:
            torch.save(model.state_dict(), f"{save_path}/ner_{model_name}_epoch_{epoch+1}.pth")
            print(f"Model saved at epoch {epoch+1}")

    if scheduler is not None:
        scheduler.step()


if __name__ == "__main__":
    DATA_PATH_DNRTI = "/content/drive/MyDrive/Colab Notebooks/CTI-KG/datasets/dataset-TiKG/DNRTI.txt"
    dnrti_df = to_df(DATA_PATH_DNRTI)

    unique_labels = dnrti_df.select(pl.col("labels").list.explode().unique())
    unique_labels_list = unique_labels["labels"].to_list()
    unique_names_list = list(set([ner_label.split("-")[-1] for ner_label in unique_labels_list if ner_label != "O"]))

    upos_encoder = UPOSEncoder()

    secure_bert_model = RobertaModel.from_pretrained("ehsanaghaei/SecureBERT")
    secure_bert_tokenizer = RobertaTokenizer.from_pretrained("ehsanaghaei/SecureBERT")

    ner_train_df, ner_val_df, ner_test_df = train_val_test_split(dnrti_df)

    BATCH_SIZE = 32
    ner_train_ds = CTINERDataset(device, upos_encoder, secure_bert_tokenizer, ner_train_df)
    ner_val_ds = CTINERDataset(device, upos_encoder, secure_bert_tokenizer, ner_val_df)
    ner_train_loader = ner_train_ds.to_dataloader(batch_size=BATCH_SIZE, shuffle=True)
    ner_val_loader = ner_val_ds.to_dataloader(batch_size=BATCH_SIZE, shuffle=False)

    NUM_CLASSES = len(ner_train_ds.label_list)
    NUM_EPOCHS = 25
    LEARNING_RATE = 4e-4

    secbert_pos_char_embedder = CustomEmbedder([
        SecBertEmbedder(emb_model=secure_bert_model),
        UPOSEmbedder(upos_vocab=len(UPOS), emb_dim=256),
        CharCNNEmbedder(char_vocab_size=len(ner_train_ds.char_vocab),
                        char_emb_dim=64, num_char_filters=48,
                        kernel_size=3, max_word_len=15)
    ]).to(device)

    print(f"Embedding dim: {secbert_pos_char_embedder.output_dim}")

    model = MultiHeadAttentionBiGruCRFNER(
        emb_layer=secbert_pos_char_embedder,
        num_classes=NUM_CLASSES,
        gru_hidden_dim=512,
        linear_hidden_dim=256,
        gru_num_layers=2,
        num_heads=8).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    scheduler = None

    MODEL_SAVED_PATH = "/content/drive/MyDrive/Colab Notebooks/ner_model_checkpoint"
    print(f"TRAINING USING DEVICE: {device}")
    train_model(model,
                ner_train_loader, ner_val_loader,
                device, optimizer, scheduler,
                NUM_EPOCHS,
                save_per=5, save_path=MODEL_SAVED_PATH,
                model_name="MultiHeadAtt_BiGru")