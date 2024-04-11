from collections import OrderedDict
from pathlib import Path

import pandas as pd
import torch

from transformers import BertTokenizer, BertConfig

from config import Config
from dataset.colwise_dataset import ColWiseDataset
from dataset.dataset import TableDataset
from model.model import BertForClassification
from utils.functions import prepare_device, set_rs, get_token_logits, get_map_location, filter_model_state_dict


class Inferencer:
    """Class for inference models.

    Note:
        Source .csv files should be placed in `data/inference` directory.
        You can choose what model to inference in `config.json`.
    """
    def __init__(self):
        self.config = Config(config_path="config.json")
        self.directory = self.config["inference_dir"]

        self.tokenizer = BertTokenizer.from_pretrained(self.config["pretrained_model_name"])

        table_serialization_type_dataset = {
            "table_wise": TableDataset,
            "column_wise": ColWiseDataset
        }
        dataset_type = table_serialization_type_dataset.get(
            self.config["table_serialization_type"],
            TableDataset
        )

        self.preprocess_tables()

        self.dataset = dataset_type(
            tokenizer=self.tokenizer,
            num_rows=self.config["dataset"]["num_rows"],
            data_dir=self.config["dataset"]["data_dir"] + "inference/preprocessed/"
        )

        self.model = BertForClassification(
            BertConfig.from_pretrained(self.config["pretrained_model_name"], num_labels=self.config["num_labels"])
        )

        checkpoint = torch.load(
            self.config["checkpoint_dir"] + self.config["inference_model_name"],
            map_location=get_map_location()
        )

        self.model.load_state_dict(filter_model_state_dict(checkpoint["model_state_dict"]))

        self.device, device_ids = prepare_device(self.config["num_gpu"])
        self.model = self.model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)

    def inference(self) -> None:
        """Inference tables and save result.

        Returns:
            None
        """
        inference_result = self._inference()
        sem_types = pd.read_csv("data/sem_types.csv")

        inference_result["labels"] = inference_result["labels"].apply(lambda x: x[0])

        def helper(x: list):
            r = []
            for i in x:
                label = sem_types.iloc[i]["label"]
                r.append(label)
            return r

        inference_result["labels"] = inference_result["labels"].apply(lambda x: helper(x))
        inference_result.to_csv(self.directory + "result.csv", index=False, sep="|")

    def _inference(self):
        """Inference samples in dataset.

        Returns:
            pd.DataFrame: Contains `table_id` and labels.
        """

        set_rs(self.config["random_seed"])

        result_df = []
        self.model.eval()
        with torch.no_grad():
            for sample in self.dataset:
                logits = []
                data = sample["data"].to(self.device)

                seq = data.unsqueeze(0)
                attention_mask = torch.clone(seq != 0)
                probs = self.model(seq, attention_mask=attention_mask)
                if isinstance(probs, tuple):
                    probs = probs[0]
                cls_probs = get_token_logits(self.device, seq, probs, self.tokenizer.cls_token_id)

                logits.append(cls_probs.argmax(1).cpu().detach().numpy().tolist())

                result_df.append([sample["table_id"], logits])

        return pd.DataFrame(
            result_df,
            columns=["table_id", "labels"]
        )

    def preprocess_tables(self) -> None:
        """Collect all tables to inference and call preprocess.

        Collects csv files from `data/inference` and then calls
        preprocess function.

        Returns:
            None
        """
        files = [
            f for f in Path(self.directory).iterdir()
            if f.is_file() and f.name.endswith(".csv") and f.name != "result.csv"
        ]
        for i, file_name in enumerate(files):
            self.preprocess_table(file_name.name, idx=i)

    def preprocess_table(self, filename: str, idx: int):
        """Preprocess table.

        Preprocess given table and save in `data/inference/preprocess/` directory.

        Args:
            filename: Table filename.
            idx: Index of current table.

        Returns:
            None
        """
        table = pd.read_csv(f"{self.directory}{filename}", header=None)

        data_list = []
        for i in table.columns:
            column_id = i
            label_id = 0
            label = "none"
            column_data = " ".join(list(map(lambda x: str(x).strip(), table[i])))
            data_list.append([
                filename, column_id, label_id, label, column_data
            ])

        preprocessed_table = pd.DataFrame(
            data_list,
            columns=["table_id", "column_id", "label_id", "label", "column_data"]
        )

        preprocessed_dir = self.directory + "preprocessed/"
        Path(preprocessed_dir).mkdir(parents=True, exist_ok=True)
        preprocessed_table.to_csv(f"{preprocessed_dir}/data_{idx}.csv", index=False, sep="|")


if __name__ == "__main__":
    inference = Inferencer()
    inference.inference()
