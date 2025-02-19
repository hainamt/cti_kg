import polars as pl


def train_val_test_split_df(df, seed=0, test_size=0.2, val_size=0.1):
    shuffled = df.with_columns(
        pl.int_range(pl.len(), dtype=pl.UInt32)
        .shuffle(seed=seed)
        .alias("idx")
    )

    test_threshold = int(len(df) * test_size)
    val_threshold = int(len(df) * (test_size + val_size))

    return (
        shuffled.filter(pl.col("idx") >= val_threshold).drop("idx"),
        shuffled.filter((pl.col("idx") >= test_threshold) & (pl.col("idx") < val_threshold)).drop("idx"),
        shuffled.filter(pl.col("idx") < test_threshold).drop("idx"),)


def train_val_test_split(X, y, seed=0, test_size=0.2, val_size=0.1):
    (X_train, X_val, X_test) = train_val_test_split_df(X, seed=seed, test_size=test_size, val_size=val_size)
    (y_train, y_val, y_test) = train_val_test_split_df(y, seed=seed, test_size=test_size, val_size=val_size)
    return X_train, X_val, X_test, y_train, y_val, y_test
