import evaluate
import torch
import polars as pl
import json
import string
from nltk.corpus import stopwords
from itertools import groupby
from transformers import RobertaTokenizerFast, EncoderDecoderModel, EvalPrediction, Seq2SeqTrainer, \
    Seq2SeqTrainingArguments
from data_modules.re_dataset import CTIREDataset
from utils.dataframe import to_df, train_val_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open("/content/drive/MyDrive/Colab Notebooks/CTI-KG/dnrti_do.json", "r", encoding="utf-8") as f:
    dnrti_do = json.load(f)

other_punc = {"''"}
punc_list = set(string.punctuation).union(other_punc)
stop_words_en = set(stopwords.words('english'))
re_delimiter = "[<RE>]"
obj_delimiter = "[<OBJ>]"
special_tokens = {re_delimiter, obj_delimiter}


def combine_token_ner(tokens, ner_labels):
    result = []
    for idx, (token, label) in enumerate(zip(tokens, ner_labels)):
        result.append(token)

        if idx == len(ner_labels) - 1:
            if label != "O":
                result.append("[" + label.split("-")[-1] + "]")
        else:
            next_label = ner_labels[idx + 1]
            if label != "O" and next_label in ["B", "O"]:
                result.append("[" + label.split("-")[-1] + "]")
    return " ".join(result)

def rela_labeling(tokens, ner_labels):
    entities = {}
    triplets = set()

    i = 0
    while i < len(tokens):
        if ner_labels[i].startswith("B"):
            label = ner_labels[i].split("-")[-1]
            start = i
            i += 1
            while i < len(tokens) and ner_labels[i].startswith("I"):
                i += 1
            entities[start] = (" ".join(tokens[start:i]), label)
        else:
            i += 1

    for obj_idx, (obj_name, obj_label) in entities.items():
        if obj_label in dnrti_do:
            for subj_idx, (subj_name, subj_label) in entities.items():
                if subj_idx != obj_idx and subj_label in dnrti_do[obj_label]:
                    relation = dnrti_do[obj_label][subj_label]
                    triplets.add((obj_name, relation, subj_name))

    sorted_data = sorted(triplets, key=lambda x: (x[0], x[1]))
    grouped_data = [ f" Ġ{obj_delimiter} ".join((key[0], key[1], *(item[2] for item in group)))
                 for key, group in groupby(sorted_data, key=lambda x: (x[0], x[1])) ]

    return f" Ġ{re_delimiter} ".join(grouped_data)

rouge = evaluate.load("rouge")
def compute_metrics(pred: EvalPrediction):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_score = rouge.compute(predictions=pred_str,
                                references=label_str,
                                rouge_types=["rouge2"])
    return rouge_score

if __name__ == "__main__":
    DATA_PATH_DNRTI = "/content/drive/MyDrive/Colab Notebooks/CTI-KG/datasets/dataset-TiKG/DNRTI.txt"
    dnrti_df = to_df(DATA_PATH_DNRTI)

    dnrti_df = dnrti_df.with_columns(
        pl.struct(["tokens", "labels"]).map_elements(
            lambda row: rela_labeling(row["tokens"], row["labels"]), return_dtype=pl.datatypes.String).alias(
            "re_labels"))

    unique_labels = dnrti_df.select(pl.col("labels").list.explode().unique())
    unique_labels_list = unique_labels["labels"].to_list()
    unique_names_list = list(set([ner_label.split("-")[-1] for ner_label in unique_labels_list if ner_label != "O"]))
    print(unique_labels_list)
    print(unique_names_list)

    model_name = "ehsanaghaei/SecureBERT"
    tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
    tokenizer.add_tokens(["[" + item + "]" for item in unique_names_list])
    tokenizer.add_tokens(["Ġ" + item for item in special_tokens])

    shared_secBERT = EncoderDecoderModel.from_encoder_decoder_pretrained(
        model_name, model_name, tie_encoder_decoder=True)
    shared_secBERT.encoder.resize_token_embeddings(len(tokenizer))
    shared_secBERT.decoder.resize_token_embeddings(len(tokenizer))

    shared_secBERT.config.decoder_start_token_id = tokenizer.bos_token_id
    shared_secBERT.config.eos_token_id = tokenizer.eos_token_id
    shared_secBERT.config.pad_token_id = tokenizer.pad_token_id
    shared_secBERT.config.max_length = 100
    shared_secBERT.config.vocab_size = shared_secBERT.config.encoder.vocab_size

    shared_secBERT.config.early_stopping = True
    shared_secBERT.config.no_repeat_ngram_size = 2
    shared_secBERT.config.length_penalty = 1.5
    shared_secBERT.config.repetition_penalty = 3.0
    shared_secBERT.config.num_beams = 4

    re_train, re_val, re_test = train_val_test_split(dnrti_df)
    re_train_ds = CTIREDataset(device=device,
                               dataframe=re_train,
                               tokenizer=tokenizer,
                               content_column="content",
                               re_label_column="re_labels",
                               ner_label_column="labels",
                               max_seq_length=90)

    re_eval_ds = CTIREDataset(device=device,
                              dataframe=re_val,
                              tokenizer=tokenizer,
                              content_column="content",
                              re_label_column="re_labels",
                              ner_label_column="labels",
                              max_seq_length=90)

    re_test_ds = CTIREDataset(device=device,
                              dataframe=re_test,
                              tokenizer=tokenizer,
                              content_column="content",
                              re_label_column="re_labels",
                              ner_label_column="labels",
                              max_seq_length=90)

    # training_args = Seq2SeqTrainingArguments(
    #     output_dir="./re_model_runs",
    #     per_device_train_batch_size=16,
    #     per_device_eval_batch_size=16,
    #     predict_with_generate=True,
    #     evaluation_strategy="epoch",
    #     do_train=True,
    #     do_eval=True,
    #     logging_steps=1024,
    #     warmup_steps=1024,
    #     num_train_epochs = 15,
    #     overwrite_output_dir=True,
    #     save_total_limit=1,
    #     fp16=True,
    # )

    training_args = Seq2SeqTrainingArguments(
        output_dir="/content/drive/MyDrive/Colab Notebooks/re_model_checkpoint/remodel_output/checkpoint-2880",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        predict_with_generate=True,
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=200,
        do_train=True,
        do_eval=True,
        warmup_steps=512,
        num_train_epochs=10,
        save_strategy="epoch",
        overwrite_output_dir=True,
        save_total_limit=1,
        resume_from_checkpoint=True,
        fp16=True,
    )

    trainer = Seq2SeqTrainer(
        model=shared_secBERT,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=re_train_ds,
        eval_dataset=re_eval_ds,
    )

    # trainer.train()