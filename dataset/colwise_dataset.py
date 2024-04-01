from itertools import chain
from typing import Optional

import pandas as pd
import torch
from tqdm import tqdm

from config import Config
from transformers import BertTokenizer, PreTrainedTokenizerBase

from dataset.dataset import TableDataset


class ColWiseDataset(TableDataset):
    """Wrapper class over the dataset.

    Designed to store columns as a single sequence with context (other columns in table). Only use one [CLS] token.

    Note:
        The tokenized columns data is stored like: [CLS] token_11 token_12 ... [SEP] token_21 ... [SEP]

    Args:
        data_dir: path to directory, where dataset .csv files placed.
        tokenizer: pretrained BERT tokenizer instance.
        num_rows: amount of how many rows to read per .csv file, if None read all rows.
    """

    def __init__(self, data_dir: str, tokenizer: PreTrainedTokenizerBase, num_rows: Optional[int]):
        super().__init__(data_dir, tokenizer, num_rows)

    def _create_dataset(self, df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase) -> pd.DataFrame:
        """Tokenize columns data.

        Groups columns by table_id's and tokenizes columns data.

        Tokenized columns are flatten into sequence, like so:

        [CLS] token_11 token_12 ... [SEP] token_21 ... [SEP]

        Args:
            df: Entire dataset as dataframe object.
            tokenizer: Pretrained BERT tokenizer.

        Returns:
            pd.Dataframe: Dataset, grouped by tables and tokenized.
        """

        data_list = []
        for table_id, table in tqdm(df.groupby("table_id")):
            num_cols = len(table)

            # Tokenize table columns
            tokenized_table_columns = table["column_data"].apply(
                lambda x: tokenizer.encode(
                    # max_length for SINGLE COLUMN. Not for table as sequence.
                    # BERT maximum input length = 512. So, max_length = (512 // num_cols) - 2 (without special tokens)
                    x, add_special_tokens=False, max_length=(512 // num_cols) - 2, truncation=True
                )
            ).tolist()

            labels = table["label_id"].values
            for i in range(num_cols):
                tail = [
                    tokenized_table_columns[j] + [tokenizer.sep_token_id] if j != i
                    else [] for j in range(num_cols)
                ]
                head = [tokenizer.cls_token_id, *tokenized_table_columns[i][:], tokenizer.sep_token_id]

                # Concat table columns into one sequence
                tokenized_columns_seq = torch.LongTensor(
                    head + list(chain.from_iterable(tail))
                )

                # Use Long, because CrossEntropyLoss works with Long tensors.
                label = torch.LongTensor([labels[i]])

                data_list.append(
                    [table_id, num_cols, tokenized_columns_seq, label]
                )
        return pd.DataFrame(
            data_list,
            columns=["table_id", "n_cols", "data", "labels"]
        )


if __name__ == "__main__":
    config = Config(config_path="../config.json")

    t = ColWiseDataset(
        data_dir="../" + config["dataset"]["data_dir"] + config["dataset"]["train_path"],
        tokenizer=BertTokenizer.from_pretrained("bert-base-multilingual-uncased"),
        num_rows=None,
    )
    print(t.df["data"].apply(lambda x: len(x)).max())
