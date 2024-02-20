from transformers import BertPreTrainedModel, BertModel

import torch.nn as nn
from torch.nn import CrossEntropyLoss


class BertForClassification(BertPreTrainedModel):
    """
    TODO
    """

    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            labels=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
    ):

        # TODO: pass attention mask? because seq may be padded
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        last_hidden_state = outputs[0]  # (batch_size, seq_len, 768)

        last_hidden_state = self.dropout(last_hidden_state)
        logits = self.classifier(last_hidden_state)  # (batch_size, seq_len, num_labels)
        outputs = (logits, ) + outputs[2:]

        if labels is not None:
            loss = CrossEntropyLoss(
                logits.view(-1, self.num_labels), labels.view(-1)
            )
            outputs = (loss, ) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)


if __name__ == "__main__":
    from config import Config

    _config = Config("../config.json")
    model = BertForClassification.from_pretrained(
        _config["pretrained_model_name"],
        num_labels=_config["num_labels"],
        output_attentions=False,
        output_hidden_states=False,
    )
    print(model)
