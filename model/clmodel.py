import torch.nn as nn

from huggingface_hub import PyTorchModelHubMixin

from config import Config
from model.colem import CoLeM


class RuTaBERTCoLeM(nn.Module, PyTorchModelHubMixin):
    """RuTaBERT model, integrated with CoLeM (Contrastive Learning-based tabular Model).

    This model uses CoLeM as an encoder (in vanilla version was pretrained HF BERT). CoLem was
    trained in contrastive settings on unlabelled tabular data (The main language is Russian).

    Args:
        Config: RuTaBERT application config.
    """

    def __init__(self, config: Config):
        super().__init__()

        self.num_labels = config["num_labels"]

        self.bert = CoLeM.from_pretrained(config["colem"]["pretrained_model_name"])
        self.dropout = nn.Dropout()
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.num_labels)

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
        encoder_last_hidden_state = outputs[0]  # (batch_size, seq_len, hidden_size)

        last_hidden_state = self.dropout(encoder_last_hidden_state)
        logits = self.classifier(last_hidden_state)  # (batch_size, seq_len, num_labels)
        return (logits, ) + outputs[2:]  # logits, (hidden_states), (attentions)  
