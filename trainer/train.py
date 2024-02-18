import itertools

import torch

from dataset.dataloader import CtaDataLoader
from dataset.dataset import TableDataset
from model.metric import multiple_f1_score
from model.model import BertForClassification

from transformers import BertTokenizer, BertConfig

import matplotlib.pyplot as plt

from config import Config


def collate(samples):
    """
    TODO: return cls indexes
    TODO: move to utils
    :param samples:
    :return:
    """
    data = torch.nn.utils.rnn.pad_sequence(
        [sample["data"] for sample in samples]
    )
    labels = torch.cat([sample["labels"] for sample in samples])

    batch = {"data": data.T, "labels": labels}
    return batch


def train(config: Config):
    # TODO: assert config variables assigned and correct
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    start_from_checkpoint = config["start_from_checkpoint"]
    num_labels = config["num_labels"]

    pretrained_model_name = config["pretrained_model_name"]
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TableDataset(
        tokenizer=tokenizer,
        num_rows=config["dataset"]["num_rows"],
        data_dir=config["dataset"]["data_dir"]
    )
    train_dataloader = CtaDataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=config["dataloader"]["num_workers"],
        split=config["dataloader"]["valid_split"],
        random_seed=config["random_seed"],
        collate_fn=collate
    )
    valid_dataloader = train_dataloader.get_valid_dataloader()

    model = BertForClassification(
        BertConfig.from_pretrained(pretrained_model_name, num_labels=num_labels)
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    # TODO: lr, args...
    optimizer = torch.optim.AdamW(model.parameters())
    # TODO: scheduler

    train_metrics = []
    valid_metrics = []

    train_losses = []
    valid_losses = []

    train_logits = []
    valid_logits = []

    train_targets = []
    valid_targets = []

    best_f1_wighted = 0.0

    # --- load from checkpoint ---
    if start_from_checkpoint:
        checkpoint = torch.load("model_best_f1_weighted.pt")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        num_epochs = num_epochs - checkpoint["epoch"]
        best_f1_wighted = checkpoint["loss"]

    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        for batch in train_dataloader:
            # TODO rename x y
            x = batch["data"].to(device)
            y = batch["labels"].to(device)

            logits, = model(x)

            # TODO: move to collate; we cant, need logits to calculate this
            cls_indexes = torch.nonzero(
                x == tokenizer.cls_token_id
            )
            cls_logits = torch.zeros(
                cls_indexes.shape[0],
                logits.shape[2]
            ).to(device)

            for i in range(cls_indexes.shape[0]):
                j, k = cls_indexes[i]
                logit_n = logits[j, k, :]
                cls_logits[i] = logit_n

            loss = loss_fn(cls_logits, y)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            # model.zero_grad with set_to_none is more efficient
            model.zero_grad(set_to_none=True)

            train_logits.append(cls_logits.argmax(1).cpu().detach().numpy().tolist())
            train_targets.append(y.cpu().detach().numpy().tolist())

        train_losses.append(train_loss / batch_size)

        # --- Metrics ---
        train_f1_micro, train_f1_macro, train_f1_weighted = multiple_f1_score(
            list(itertools.chain.from_iterable(train_logits)),
            list(itertools.chain.from_iterable(train_targets)),
            num_labels
        )
        print(f"micro: {train_f1_micro} \nmacro: {train_f1_macro} \nweighted: {train_f1_weighted}")
        train_metrics.append(train_f1_weighted)

        # --- Validation ---
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch in valid_dataloader:
                # TODO rename x y
                x = batch["data"].to(device)
                y = batch["labels"].to(device)

                # TODO: why it can return tuple(tensor), except for just tensor?
                probs = model(x)
                if type(probs) == tuple:
                    probs = probs[0]

                # TODO: move to collate; we cant, need logits to calculate this
                cls_indexes = torch.nonzero(
                    x == tokenizer.cls_token_id
                )
                cls_probs = torch.zeros(
                    cls_indexes.shape[0],
                    probs.shape[2]
                ).to(device)

                for i in range(cls_indexes.shape[0]):
                    j, k = cls_indexes[i]
                    prob_i = probs[j, k, :]
                    cls_probs[i] = prob_i

                loss = loss_fn(cls_probs, y)
                valid_loss += loss.item()

                valid_logits.append(cls_probs.argmax(1).cpu().detach().numpy().tolist())
                valid_targets.append(y.cpu().detach().numpy().tolist())

        # --- F1 Metrics ---
        valid_f1_micro, valid_f1_macro, valid_f1_weighted = multiple_f1_score(
            list(itertools.chain.from_iterable(valid_logits)),
            list(itertools.chain.from_iterable(valid_targets)),
            num_labels
        )
        print(f"valid_micro: {valid_f1_micro} \nvalid_macro: {valid_f1_macro} \nvalid_weighted: {valid_f1_weighted}")
        valid_metrics.append(valid_f1_weighted)

        valid_losses.append(valid_loss / batch_size)
        if best_f1_wighted < valid_f1_weighted:
            best_f1_wighted = valid_f1_weighted
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': valid_losses[-1],
                },
                "model_best_f1_weighted.pt"
            )
            print(f"epoch = {epoch}")
    return train_losses, valid_losses, train_metrics, valid_metrics


if __name__ == "__main__":
    tr_loss, vl_loss, tr_f1, vl_f1 = train(Config(config_path="../config.json"))

    plt.plot(tr_loss)
    plt.plot(vl_loss)
    plt.legend(['Train loss', 'Valid loss'])
    plt.show()

    plt.plot(tr_f1)
    plt.plot(vl_f1)
    plt.legend(['Train f1 weighted', 'Valid f1 weighted'])
    plt.show()
