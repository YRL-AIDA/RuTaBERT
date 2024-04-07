from typing import Optional

from tqdm import tqdm

import torch
from torch.utils.data import Dataset

import pandas as pd
import glob

from transformers import BertTokenizer, PreTrainedTokenizerBase

from itertools import chain


class TableDataset(Dataset):
    """Wrapper class over the dataset.

    Designed to store tables as a single sequence (DODUO approach).

    Note:
        The tokenized columns data is stored like: [CLS] token_11 token_12 ... [SEP] [CLS] token_21 ... [SEP]

    Args:
        data_dir: path to directory, where dataset .csv files placed.
        tokenizer: pretrained BERT tokenizer instance.
        num_rows: amount of how many rows to read per .csv file, if None read all rows.
        transform: Optional transform to be applied on a sample
        target_transform: Optional transform to be applied on a target.
    """
    def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizerBase,
            num_rows: Optional[int],
            file_name=None,
            transform=None,
            target_transform=None,
    ):
        # Read dataset .csv files
        if file_name:
            df = pd.read_csv(
                data_dir + file_name,
                sep="|",
                engine="python",
                quotechar='"',
                on_bad_lines="warn",
                nrows=num_rows if num_rows is not None else None
            )
        else:
            df = self.read_multiple_csv(data_dir, num_rows)

        # Tokenize dataset with BERT tokenizer
        self.df = self._create_dataset(
            df,
            tokenizer
        )

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "data": self.df.iloc[idx]["data"],
            "labels": self.df.iloc[idx]["labels"],
            "table_id": self.df.iloc[idx]["table_id"]
        }

    def read_multiple_csv(self, data_dir: str, num_rows: Optional[int] = None) -> pd.DataFrame:
        """Read dataframe from multiple csv files.

        If dataset was split into multiple files, it will be concatenated. Dataset is stored
        in pd.Dataframe instance.

        Args:
            data_dir: path to directory, where dataset .csv files placed.
            num_rows: amount of how many rows to read per .csv file, if None read all rows.

        Returns:
            pd.Dataframe: Entire dataset as dataframe.
        """

        df_list = []
        num_chunks = len(glob.glob(data_dir + "data_*.csv"))
        if num_chunks > 1:
            for i in range(num_chunks):
                df = pd.read_csv(
                    data_dir + f"data_{i}.csv",
                    sep="|",
                    engine="python",
                    quotechar='"',
                    on_bad_lines="warn",
                    nrows=num_rows if num_rows is not None else None
                )
                df_list.append(df)
            return pd.concat(df_list, axis=0)
        return pd.read_csv(
            data_dir + "data.csv",
            sep="|",
            engine="python",
            quotechar='"',
            on_bad_lines="warn",
            nrows=num_rows if num_rows is not None else None
        )

    def _create_dataset(self, df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase) -> pd.DataFrame:
        """Tokenize columns data.

        Groups columns by table_id's and tokenizes columns data.

        Tokenized columns are flatten into sequence, like so:

        [CLS] token_11 token_12 ... [SEP] [CLS] token_21 ... [SEP]

        Args:
            df: Entire dataset as dataframe object.
            tokenizer: Pretrained BERT tokenizer.

        Returns:
            pd.Dataframe: Dataset, grouped by tables and tokenized.
        """

        data_list = []
        for table_id, table in tqdm(df.groupby("table_id")):
            num_cols = len(table)

            # Tokenize table columns.
            tokenized_table_columns = table["column_data"].apply(
                lambda x: tokenizer.encode(
                    # max_length for SINGLE COLUMN. Not for table as sequence.
                    # BERT maximum input length = 512. So, max_length = (512 // num_cols).
                    x, add_special_tokens=True, max_length=(512 // num_cols), truncation=True
                )
            ).tolist()

            # Concat table columns into one sequence.
            concat_tok_table_columns = list(chain.from_iterable(tokenized_table_columns))
            tokenized_columns_seq = torch.LongTensor(concat_tok_table_columns)

            # Use Long, because CrossEntropyLoss works with Long tensors.
            labels = torch.LongTensor(table["label_id"].values)

            data_list.append(
                [table_id, num_cols, tokenized_columns_seq, labels]
            )

        return pd.DataFrame(
            data_list,
            columns=["table_id", "n_cols", "data", "labels"]
        )


if __name__ == "__main__":
    from config import Config
    config = Config(config_path="../config.json")

    t = TableDataset(
        data_dir="../" + config["dataset"]["data_dir"] + config["dataset"]["train_path"],
        tokenizer=BertTokenizer.from_pretrained("bert-base-multilingual-uncased"),
        num_rows=100,
    )
    print(t.df["data"].apply(lambda x: len(x)).max())

