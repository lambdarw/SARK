import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoModel, AutoConfig, PretrainedConfig, BertModel

class BgeJoinedConfig(PretrainedConfig):
    model_type = "bge_joined"
    def __init__(self, hidden_dropout_prob=0.3, attention_probs_dropout_prob=0.3, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob

class BgeJoinedModel(PreTrainedModel):
    config_class = BgeJoinedConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        self.bge = BertModel(config) 

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(256, 2)
        )
        
        self.post_init()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bge(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # [CLS] embedding
        h = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(h)
        return logits
