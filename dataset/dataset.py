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
            transform=None,
            target_transform=None,
            data_dir: str = "../data/",
            tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")
    ):
        # Read dataset .csv files from ../data/ dir:
        df = TableDataset.read_multiple_csv(data_dir, 50)

        # Tokenize dataset with BERT tokenizer
        data_list = TableDataset._create_dataset(
            df,
            tokenizer
        )
        # Assign tokenized dataset
        self.df = pd.DataFrame(
            data_list,
            columns=["table_id", "n_cols", "data", "labels", "cls_ids"]
        )

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "data": self.df.iloc[idx]["data"],
            "labels": self.df.iloc[idx]["labels"]
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
                on_bad_lines="warn"
            )

            # For now using subset for time optimization
            if num_rows is not None:
                df_list.append(df[:num_rows])
            else:
                df_list.append(df)
        return pd.concat(df_list, axis=0)

    @staticmethod
    def _create_dataset(df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase) -> list:
        """
        TODO
        :return:
        """

        data_list = []
        for index, table in tqdm(df.groupby("table_id")):
            num_cols = len(table)

            # Tokenize table columns
            tokenized_table_columns = table["column_data"].apply(
                lambda x: tokenizer.encode(
                    # max_length for SINGLE COLUMN. Not for table as sequence.
                    # BERT input length would be 512. max_length = 512 / num_cols) + 2 ([CLS] and [SEP])
                    x, add_special_tokens=True, max_length=(512 // num_cols) + 2, truncation=True
                )
            ).tolist()

            # Concat table columns into one sequence
            concat_tok_table_columns = list(chain.from_iterable(tokenized_table_columns))

            tokenized_columns_seq = torch.IntTensor(concat_tok_table_columns)

            labels = torch.IntTensor(table["label_id"].values)

            # Get [CLS] token indexes in tokenized sequence
            cls_ids = torch.IntTensor([
                i for i in range(len(tokenized_columns_seq)) if tokenized_columns_seq[i] == tokenizer.cls_token_id
            ])

            data_list.append(
                [index, num_cols, tokenized_columns_seq, labels, cls_ids]
            )
            # print('point')
        return data_list


if __name__ == "__main__":
    t = TableDataset()
    # print('point')
