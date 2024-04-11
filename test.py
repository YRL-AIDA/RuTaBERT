import torch

from collections import OrderedDict

from config import Config
from dataset.colwise_dataset import ColWiseDataset
from dataset.dataloader import CtaDataLoader
from dataset.dataset import TableDataset
from logs.logger import Logger
from model.metric import multiple_f1_score
from model.model import BertForClassification

from transformers import BertTokenizer, BertConfig

from utils.functions import collate, prepare_device, get_token_logits, set_rs, get_map_location


def test(
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
            # TODO: why it can return tuple(tensor), except for just tensor?
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
    conf = Config(config_path="config.json")

    tokenizer = BertTokenizer.from_pretrained(conf["pretrained_model_name"])

    table_serialization_type_dataset = {
        "table_wise": TableDataset,
        "column_wise": ColWiseDataset
    }
    dataset_type = table_serialization_type_dataset.get(
        conf["table_serialization_type"],
        TableDataset
    )
    dataset = dataset_type(
        tokenizer=tokenizer,
        num_rows=conf["dataset"]["num_rows"],
        data_dir=conf["dataset"]["data_dir"] + conf["dataset"]["test_path"]
    )
    dataloader = CtaDataLoader(
        dataset,
        batch_size=conf["batch_size"],
        num_workers=conf["dataloader"]["num_workers"],
        collate_fn=collate
    )

    model = BertForClassification(
        BertConfig.from_pretrained(conf["pretrained_model_name"], num_labels=conf["num_labels"])
    )

    checkpoint = torch.load(conf["checkpoint_dir"] + conf["checkpoint_name"], map_location=get_map_location())

    model_state_dict = checkpoint["model_state_dict"]
    filtered_model_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        if k.startswith("module."):
            filtered_model_state_dict[k[7:]] = v
        else:
            filtered_model_state_dict[k] = v
    model.load_state_dict(filtered_model_state_dict)

    device, device_ids = prepare_device(conf["num_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    loss_metrics = test(
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
    logger = Logger(conf["test_log_filename"])
    logger.info(f"--- --- ---", "TEST")
    logger.info(f"Loss: {loss_metrics['loss']};", "LOSS")
    for metric in conf["metrics"]:
        logger.info(f"{metric} = {loss_metrics['metrics'][metric]}", "METRIC")
