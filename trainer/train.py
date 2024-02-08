import torch

from dataset.dataset import TableDataset
from model.model import BertForClassification

from transformers import BertTokenizer, BertConfig

from torch.utils.data import DataLoader, RandomSampler

import matplotlib.pyplot as plt


def collate(samples):
    """
    TODO: maybe put this logic into dataset?
    TODO: return cls indexes
    :param samples:
    :return:
    """
    data = torch.nn.utils.rnn.pad_sequence(
        [sample["data"] for sample in samples]
    )
    labels = torch.cat([sample["labels"] for sample in samples])

    batch = {"data": data.T, "labels": labels}
    return batch


def train(batch_size: int = 2):
    train_dataset = TableDataset(split="train")
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size,
        collate_fn=collate
    )

    valid_dataset = TableDataset(split="valid")
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        collate_fn=collate
    )

    # ---- params ----
    # shortcut_name = "bert-base-uncased"
    shortcut_name = "bert-base-multilingual-uncased"
    device = "cpu"
    n_labels = 339
    num_epochs = 10

    # ---- bert ----
    config = BertConfig.from_pretrained(shortcut_name, num_labels=n_labels)
    model = BertForClassification(config)
    tokenizer = BertTokenizer.from_pretrained(shortcut_name)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters())
    # TODO: scheduler

    train_losses = []
    valid_losses = []
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        for batch in train_dataloader:
            x = batch["data"].to(device)
            y = batch["labels"].to(device)

            logits, = model(x)

            # TODO: move to collate or dataset
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
        train_losses.append(train_loss / batch_size)

        # Validation
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch in valid_dataloader:
                x = batch["data"].to(device)
                y = batch["labels"].to(device)

                # TODO: why it can return tuple(tensor), except for just tensor?
                probs = model(x)
                if type(probs) == tuple:
                    probs = probs[0]

                # TODO: move to collate or dataset
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
        valid_losses.append(valid_loss / batch_size)
    return train_losses, valid_losses


if __name__ == "__main__":
    tr_loss, vl_loss = train()
    plt.plot(tr_loss)
    plt.plot(vl_loss)
    plt.legend(['Train loss', 'Valid loss'])
    plt.show()
