from typing import Optional

from tqdm import tqdm

import torch
from torch.utils.data import Dataset

import pandas as pd
import glob

from transformers import BertTokenizer, PreTrainedTokenizerBase

from itertools import chain


class TableDataset(Dataset):
    """
    TODO:
    """

    def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizerBase,
            num_rows: Optional[int],
            transform=None,
            target_transform=None,
    ):
        # Read dataset .csv files from ../data/ dir:
        df = TableDataset.read_multiple_csv(data_dir, num_rows)

        # Tokenize dataset with BERT tokenizer
        self.df = TableDataset._create_dataset(
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
        }

    @staticmethod
    def read_multiple_csv(data_dir: str, num_rows: Optional[int] = None) -> pd.DataFrame:
        """
        TODO
        :return:
        """

        df_list = []
        num_chunks = len(glob.glob(data_dir + "data_*.csv"))
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

    @staticmethod
    def _create_dataset(df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase) -> pd.DataFrame:
        """
        TODO
        :return:
        """

        data_list = []
        for table_id, table in tqdm(df.groupby("table_id")):
            num_cols = len(table)

            # Tokenize table columns
            # TODO: move to collate, do it in a batch?
            tokenized_table_columns = table["column_data"].apply(
                lambda x: tokenizer.encode(
                    # TODO: what if column is almost empty? then we reduce max_length for other columns.
                    # max_length for SINGLE COLUMN. Not for table as sequence.
                    # BERT maximum input length = 512. So, max_length = (512 // num_cols)
                    x, add_special_tokens=True, max_length=(512 // num_cols), truncation=True  # TODO: config
                )
            ).tolist()

            # Concat table columns into one sequence
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
    t = TableDataset(
        data_dir="../data",
        tokenizer=BertTokenizer.from_pretrained("bert-base-multilingual-uncased"),
        num_rows=1,
    )
