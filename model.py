import torch
import torch.nn as nn
from transformers import BertModel

class ABSA_Model(nn.Module):
    def __init__(self):
        super(ABSA_Model, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(1536, 3)  # Updated input size and output classes

    def forward(self, text_input_ids, text_attention_mask, aspect_input_ids, aspect_attention_mask):
        # Pass text through BERT
        text_output = self.bert(input_ids=text_input_ids, attention_mask=text_attention_mask)
        cls_output = text_output.pooler_output  # [CLS] token output

        # Pass aspect tokens through BERT
        aspect_output = self.bert(input_ids=aspect_input_ids, attention_mask=aspect_attention_mask)
        aspect_cls_output = aspect_output.pooler_output  # [CLS] token output for aspect

        # Concatenate text and aspect representations
        combined_features = torch.cat((cls_output, aspect_cls_output), dim=1)

        # Apply dropout and classifier
        combined_features = self.dropout(combined_features)
        logits = self.classifier(combined_features)

        return logits
