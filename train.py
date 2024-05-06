import pandas as pd
import torch

from dataset.dataloader import CtaDataLoader

from logs.logger import Logger

from model.metric import multiple_f1_score
from model.model import BertForClassification

from transformers import BertTokenizer, BertConfig, get_linear_schedule_with_warmup

from config import Config
from trainer.trainer import Trainer
from utils.functions import prepare_device, collate, plot_graphs, set_rs, get_dataset_type


def train(config: Config):
    set_rs(config["random_seed"])

    # TODO: assert config variables assigned and correct
    tokenizer = BertTokenizer.from_pretrained(config["pretrained_model_name"])

    dataset_type = get_dataset_type(config["table_serialization_type"])
    dataset = dataset_type(
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
    results = pd.DataFrame()

    conf = Config(config_path="config.json")

    losses, metrics = train(conf)

    # plot_graphs(losses, metrics, conf)

    results["train_loss"] = losses["train"]
    results["valid_loss"] = losses["valid"]

    for metric in conf["metrics"]:
        tr_f1, vl_f1 = metrics["train"][metric], metrics["valid"][metric]
        results[f"train-{metric}"] = tr_f1
        results[f"valid-{metric}"] = vl_f1

    results.to_csv(conf["logs_dir"] + "training_results.csv", index=False)
