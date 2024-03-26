from transformers import BertPreTrainedModel, BertModel

import torch.nn as nn


class BertForClassification(BertPreTrainedModel):
    """BERT model for `Column Table Annotation` task.

    Args:
        config: Model configuration class with all the parameters of the BERT model.
    """

    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None) -> tuple:
        """Forward pass.

        Pass `input_ids` with `attentions_mask` to BERT model, and then take the `last_hidden_state` of the BERT output
        (batch_size x sequence_length x bert_output) and pass this output through the dropout and the linear layers.
        The output tensor have (batch_size x sequence_length x num_labels) size.

        Note:
            Logits have **(batch_size, sequence_length, num_labels)** size.

        Args:
            input_ids: Indices of input sequence tokens in the vocabulary.
            attention_mask:
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            tuple: Tuple of (logits, hidden_states, attentions).
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        last_hidden_state = outputs[0]  # (batch_size, seq_len, 768)

        last_hidden_state = self.dropout(last_hidden_state)
        logits = self.classifier(last_hidden_state)  # (batch_size, seq_len, num_labels)
        outputs = (logits, ) + outputs[2:]

        return outputs  # logits, (hidden_states), (attentions)


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
