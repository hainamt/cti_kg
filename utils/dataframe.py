import polars as pl
import nltk
from nltk.corpus import stopwords
from itertools import groupby
import string

other_punc = {"''"}
punc_list = set(string.punctuation).union(other_punc)
stop_words_en = set(stopwords.words('english'))

nltk.download('stopwords')

def to_df(file_path, delimiter=" ", len_threshold=5):
    def split_tokens_and_labels(sentence, item_delimiter=" "):
        tokens, labels = zip(
            *(item for pair in sentence if (item := pair.split(item_delimiter)) if (token := item[0].lower())
              and all([len(item) == 2,
                       token not in punc_list,
                       token not in stop_words_en])))
        return [list(map(lambda x: x.lower(), tokens)), list(labels)]

    with open(file_path, "r", encoding="utf-8") as f:
        data = [list(group) for is_empty, group in groupby((line.strip() for line in f), bool) if is_empty]
    sentences = [sentence for sentence in data if len(sentence) >= len_threshold]

    token_label = []
    for idx, sentence in enumerate(sentences):
        try:
            token_label.append(split_tokens_and_labels(sentence, delimiter))
        except Exception as ex:
            print(f"At index: {idx}")
            raise Exception(f"Error {ex} in sentence: ", sentence)
    df = pl.DataFrame(token_label, schema=["tokens", "labels"], orient="row")
    df = df.with_columns(pl.col("tokens").list.join(" ").alias("content"))
    mismatch_count = df.filter(pl.col("tokens").list.len() != pl.col("labels").list.len()).shape[0]
    assert mismatch_count == 0, "Mismatched tokens and labels"
    return df

def train_val_test_split(df, seed=0, test_size=0.2, val_size=0.1):
    shuffled = df.with_columns(
        pl.int_range(pl.len(), dtype=pl.UInt32)
        .shuffle(seed=seed)
        .alias("idx")
    )

    test_threshold = int(len(df) * test_size)
    val_threshold = int(len(df) * (test_size + val_size))

    train_df = shuffled.filter(pl.col("idx") >= val_threshold).drop("idx")
    val_df = shuffled.filter((pl.col("idx") >= test_threshold) & (pl.col("idx") < val_threshold)).drop("idx")
    test_df = shuffled.filter(pl.col("idx") < test_threshold).drop("idx")

    return train_df, val_df, test_df
