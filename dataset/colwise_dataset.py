from itertools import chain
from typing import Optional

import pandas as pd
import torch
from tqdm import tqdm

from config import Config
from transformers import BertTokenizer, PreTrainedTokenizerBase

from dataset.dataset import TableDataset


class ColWiseDataset(TableDataset):
    """TODO"""

    def __init__(self, data_dir: str, tokenizer: PreTrainedTokenizerBase, num_rows: Optional[int]):
        super().__init__(data_dir, tokenizer, num_rows)

    def _create_dataset(self, df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase) -> pd.DataFrame:
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
                    x, add_special_tokens=False, max_length=(512 // num_cols), truncation=True
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
        num_rows=10,
    )
    print(t)
