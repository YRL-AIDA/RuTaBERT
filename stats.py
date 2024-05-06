from pathlib import Path

import torch

from collections import OrderedDict

from config import Config
from dataset.dataloader import CtaDataLoader
from logs.logger import Logger
from model.metric import multiple_f1_score
from model.model import BertForClassification

from transformers import BertTokenizer, BertConfig

from utils.functions import collate, prepare_device, get_token_logits, set_rs, get_map_location, \
    filter_model_state_dict, get_dataset_type


def stat(
        config,
        model,
        dataloader,
        device,
        tokenizer,
        loss_fn,
        metric_fn,
        batch_size,
        num_labels
):
    set_rs(config["random_seed"])

    _logits, _targets = [], []

    model.eval()

    running_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            data = batch["data"].to(device)
            labels = batch["labels"].to(device)

            attention_mask = torch.clone(data != 0)
            probs = model(data, attention_mask=attention_mask)
            if isinstance(probs, tuple):
                probs = probs[0]
            cls_probs = get_token_logits(device, data, probs, tokenizer.cls_token_id)

            loss = loss_fn(cls_probs, labels)
            running_loss += loss.item()

            _logits.append(cls_probs.argmax(1).cpu().detach().numpy().tolist())
            _targets.append(labels.cpu().detach().numpy().tolist())

    return {
        "loss": running_loss / batch_size,
        "metrics": metric_fn(_logits, _targets, num_labels)
    }


if __name__ == "__main__":
    conf = Config()
    tokenizer = BertTokenizer.from_pretrained(conf["pretrained_model_name"])

    dataset_type = get_dataset_type(conf["table_serialization_type"])

    model = BertForClassification(
        BertConfig.from_pretrained(conf["pretrained_model_name"], num_labels=conf["num_labels"])
    )

    checkpoint = torch.load(conf["checkpoint_dir"] + conf["checkpoint_name"], map_location=get_map_location())

    model.load_state_dict(filter_model_state_dict(checkpoint["model_state_dict"]))

    device, device_ids = prepare_device(conf["num_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    logger = Logger("logs/stat.log")

    # all labels
    top_labels = dict()
    files = [
        f.name for f in Path(conf["dataset"]["data_dir"] + "stats/labels/").iterdir()
        if f.is_file() and f.name.endswith(".csv")
    ]
    for file_name in files[:3]:
        dataset = dataset_type(
            tokenizer=tokenizer,
            num_rows=conf["dataset"]["num_rows"],
            data_dir=conf["dataset"]["data_dir"] + "stats/labels/",
            file_name=file_name
        )
        dataloader = CtaDataLoader(
            dataset,
            batch_size=conf["batch_size"],
            num_workers=conf["dataloader"]["num_workers"],
            collate_fn=collate
        )

        loss_metrics = stat(
            conf,
            model,
            dataloader,
            device,
            tokenizer,
            torch.nn.CrossEntropyLoss(),
            multiple_f1_score,
            conf["batch_size"],
            conf["num_labels"]
        )

        top_labels[file_name[:-4]] = [
            loss_metrics["loss"],
            *[loss_metrics["metrics"][metric].item() for metric in conf["metrics"]]
        ]

        # Logging results
        logger.info(f"--- {file_name} ---", "STATS")
        logger.info(f"Loss: {loss_metrics['loss']};", "LOSS")
        for metric in conf["metrics"]:
            logger.info(f"{metric} = {loss_metrics['metrics'][metric]}", "METRIC")

    # Log top 5 / least 5 labels
    num_tops = 5
    top_labels_micro = OrderedDict(sorted(top_labels.items(), key=lambda item: item[1][1]))
    logger.info(f"--- top 5 micro ---", "TOP")
    t = list(top_labels_micro)
    for k in t[:num_tops]:
        logger.info(f"{k} / {top_labels_micro[k]}", "TOP")

    logger.info(f"--- least 5 micro ---", "TOP")
    t.reverse()
    for k in t[:num_tops]:
        logger.info(f"{k} / {top_labels_micro[k]}", "TOP")

    top_labels_macro = OrderedDict(sorted(top_labels.items(), key=lambda item: item[1][2]))
    logger.info(f"--- top 5 macro ---", "TOP")
    t = list(top_labels_macro)
    for k in t[:num_tops]:
        logger.info(f"{k} / {top_labels_macro[k]}", "TOP")

    logger.info(f"--- least 5 macro ---", "TOP")
    t.reverse()
    for k in t[:num_tops]:
        logger.info(f"{k} / {top_labels_macro[k]}", "TOP")

    top_labels_weighted = OrderedDict(sorted(top_labels.items(), key=lambda item: item[1][3]))
    logger.info(f"--- top 5 weighted ---", "TOP")
    t = list(top_labels_weighted)
    for k in t[:num_tops]:
        logger.info(f"{k} / {top_labels_weighted[k]}", "TOP")

    logger.info(f"--- least 5 weighted ---", "TOP")
    t.reverse()
    for k in t[:num_tops]:
        logger.info(f"{k} / {top_labels_weighted[k]}", "TOP")

    # date, numeric, ...
    labels = ["date.csv", "long_text.csv", "numeric.csv", "persons.csv", "short_text.csv", "url.csv"]
    for file_name in labels[-1:]:
        dataset = dataset_type(
            tokenizer=tokenizer,
            num_rows=conf["dataset"]["num_rows"],
            data_dir=conf["dataset"]["data_dir"] + "stats/",
            file_name=file_name
        )
        dataloader = CtaDataLoader(
            dataset,
            batch_size=conf["batch_size"],
            num_workers=conf["dataloader"]["num_workers"],
            collate_fn=collate
        )

        loss_metrics = stat(
            conf,
            model,
            dataloader,
            device,
            tokenizer,
            torch.nn.CrossEntropyLoss(),
            multiple_f1_score,
            conf["batch_size"],
            conf["num_labels"]
        )

        # Logging results
        logger = Logger("logs/stat.log")
        logger.info(f"--- {file_name} ---", "STATS")
        logger.info(f"Loss: {loss_metrics['loss']};", "LOSS")
        for metric in conf["metrics"]:
            logger.info(f"{metric} = {loss_metrics['metrics'][metric]}", "METRIC")
