from itertools import groupby
import string
import nltk
from nltk.corpus import stopwords
import polars as pl

nltk.download('stopwords')


class CTINERPreprocessor:
    other_punc = {"''"}
    punc_list = set(string.punctuation).union(other_punc)
    stop_words_en = set(stopwords.words('english'))

    @staticmethod
    def read_dataset_sentence(file_path, len_threshold=5):
        with open(file_path, "r", encoding="utf-8") as f:
            sentences = [list(group) for is_empty, group in groupby((line.strip() for line in f), bool) if is_empty]
            return [sentence for sentence in sentences if len(sentence) >= len_threshold]

    @staticmethod
    def split_tokens_and_labels(data, delimiter=" "):
        tokens, labels = zip(*(item for pair in data if (item := pair.split(delimiter)) if (token := item[0].lower())
                               and all([len(item) == 2,
                                        token not in CTINERPreprocessor.punc_list,
                                        token not in CTINERPreprocessor.stop_words_en])))
        return [list(map(lambda x: x.lower(), tokens)), list(labels)]

    @staticmethod
    def to_df(sentences, delimiter=" "):
        token_label = []
        for idx, sentence in enumerate(sentences):
            try:
                token_label.append(CTINERPreprocessor.split_tokens_and_labels(sentence, delimiter))
            except Exception as ex:
                print(f"At index: {idx}")
                raise Exception(f"Error {ex} in sentence: ", sentence)
        df = pl.DataFrame(token_label, schema=["tokens", "labels"], orient="row")
        df = df.with_columns(pl.col("tokens").list.join(" ").alias("content"))
        mismatch_count = df.filter(pl.col("tokens").list.len() != pl.col("labels").list.len()).shape[0]
        assert mismatch_count == 0, "Mismatched tokens and labels"
        return df





