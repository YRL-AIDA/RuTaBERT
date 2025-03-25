from transformers import DistilBertModel, DistilBertPreTrainedModel
from huggingface_hub import PyTorchModelHubMixin

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoLeM(DistilBertPreTrainedModel, PyTorchModelHubMixin):
    """Modified CoLeM model.
    
    CoLeM is a contrastive learning model for table understanding tasks. The model has no projector head* (it has only 
    the first linear layer of CoLeM projector head). Also this model uses `leaky_relu`, instead of `relu` in forward 
    pass.
    """
    def __init__(self, config):
        super().__init__(config)

        # Encoder layers
        self.encoder = DistilBertModel(config)
        
        # Projector layers
        self.projector_lin1 = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
        
        # Initialize non-pretrained layers
        self.init_weights()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple:
        """Modified CoLeM forward pass.

        Args:
            input (torch.Tensor): batch (batch_size, sequence_length) of table columns.

        Returns:
            tuple: Tuple of (logits, hidden_states, attentions).
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )  
        encoder_last_hidden_state = outputs[0]  # (batch_size, sequence_length, encoder_output)
        output = F.leaky_relu(self.projector_lin1(encoder_last_hidden_state))  # (batch_size, encoder_output)
        return (output, ) + outputs[2:]
