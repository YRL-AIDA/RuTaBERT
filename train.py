import numpy as np
import pandas as pd
import torch

from dataset.dataloader import CtaDataLoader
from dataset.dataset import TableDataset
from logs.logger import Logger
from model.metric import multiple_f1_score
from model.model import BertForClassification

from transformers import BertTokenizer, BertConfig, get_linear_schedule_with_warmup

import matplotlib.pyplot as plt

from config import Config
from trainer.trainer import Trainer
from utils.functions import prepare_device, collate

# Random seed
torch.manual_seed(13)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(13)


def train(config: Config):
    # TODO: assert config variables assigned and correct
    tokenizer = BertTokenizer.from_pretrained(config["pretrained_model_name"])

    dataset = TableDataset(
        tokenizer=tokenizer,
        num_rows=config["dataset"]["num_rows"],
        data_dir=config["dataset"]["data_dir"] + config["dataset"]["train_path"]
    )
    train_dataloader = CtaDataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["dataloader"]["num_workers"],
        split=config["dataloader"]["valid_split"],
        collate_fn=collate
    )
    valid_dataloader = train_dataloader.get_valid_dataloader()

    model = BertForClassification(
        BertConfig.from_pretrained(config["pretrained_model_name"], num_labels=config["num_labels"])
    )

    # TODO: multi-gpu support!
    device, device_ids = prepare_device(config["num_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, eps=1e-8)
    trainer = Trainer(
        model,
        tokenizer,
        config["num_labels"],
        torch.nn.CrossEntropyLoss(),
        multiple_f1_score,
        optimizer,
        config,
        device,
        config["batch_size"],
        train_dataloader,
        valid_dataloader,
        lr_scheduler=get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_dataloader) * config["num_epochs"]
        ),
        num_epochs=config["num_epochs"],
        logger=Logger(filename=config["train_log_filename"])
    )
    return trainer.train()


if __name__ == "__main__":
    # TODO: move saving dataframe and plot graphs into separate fn, classes
    results = pd.DataFrame()

    conf = Config(config_path="config.json")

    losses, metrics = train(conf)

    tr_loss, vl_loss = losses["train"], losses["valid"]
    results["train_loss"] = losses["train"]
    results["valid_loss"] = losses["valid"]

    plt.plot(tr_loss)
    plt.plot(vl_loss)
    plt.legend(["Train loss", "Valid loss"])
    plt.show()

    for metric in conf["metrics"]:
        tr_f1, vl_f1 = metrics["train"][metric], metrics["valid"][metric]
        results[f"train-{metric}"] = tr_f1
        results[f"valid-{metric}"] = vl_f1
        plt.plot(tr_f1)
        plt.plot(vl_f1)
        plt.legend([f"Train {metric}", f"Valid {metric}"])
        plt.show()

    results.to_csv("training_results.csv", index=False)
