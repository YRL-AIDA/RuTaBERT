import itertools

import torch
from torcheval.metrics.functional import multiclass_f1_score


def multiple_f1_score(output: list, target: list, num_classes: int) -> dict:
    """TODO"""

    output = list(itertools.chain.from_iterable(output))
    target = list(itertools.chain.from_iterable(target))

    # TODO: memory cpy here, may be bad.
    logits_tensor = torch.tensor(output)
    targets_tensor = torch.tensor(target)
    f1_micro = multiclass_f1_score(
        logits_tensor,
        targets_tensor,
        num_classes=num_classes,
        average="micro"
    )
    # TODO: useless if has class imbalance
    f1_macro = multiclass_f1_score(
        logits_tensor,
        targets_tensor,
        num_classes=num_classes,
        average="macro"
    )
    f1_weighted = multiclass_f1_score(
        logits_tensor,
        targets_tensor,
        num_classes=num_classes,
        average="weighted"
    )
    return {
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted
    }
