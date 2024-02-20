import torch

from config import Config
from dataset.dataloader import CtaDataLoader
from dataset.dataset import TableDataset
from model.metric import multiple_f1_score
from model.model import BertForClassification

from transformers import BertTokenizer, BertConfig

from utils.functions import collate, prepare_device, get_token_logits


def test(
        model,
        dataloader,
        device,
        tokenizer,
        loss_fn,
        metric_fn,
        batch_size,
        num_labels
):
    _logits, _targets = [], []

    model.eval()

    running_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            data = batch["data"].to(device)
            labels = batch["labels"].to(device)

            probs = model(data)
            # TODO: why it can return tuple(tensor), except for just tensor?
            if type(probs) == tuple:
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

    dataset = TableDataset(
        tokenizer=tokenizer,
        num_rows=conf["dataset"]["num_rows"],
        data_dir=conf["dataset"]["data_dir"]
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
    checkpoint = torch.load(conf["checkpoint_dir"] + conf["checkpoint_name"])
    model.load_state_dict(checkpoint['model_state_dict'])

    # TODO: multi-gpu support!
    device, device_ids = prepare_device(conf["num_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    loss_metrics = test(
        model,
        dataloader,
        device,
        tokenizer,
        torch.nn.CrossEntropyLoss(),
        multiple_f1_score,
        conf["batch_size"],
        conf["num_labels"]
    )

    print(f"Loss: {loss_metrics['loss']};")
    for metric in conf["metrics"]:
        print(f"{metric} = {loss_metrics['metrics'][metric]}")
