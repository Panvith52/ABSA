import torch
from transformers import BertTokenizer
from model import ABSA_Model

# Initialize the model and tokenizer
pretrained_model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
model = ABSA_Model(pretrained_model_name=pretrained_model_name, num_labels=3)

# Example text and aspect
text = "The battery life of this phone is amazing."
aspect = "battery life"

# Tokenize the text and aspect
text_encoding = tokenizer(
    text,
    padding='max_length',
    truncation=True,
    max_length=128,
    return_tensors='pt'
)
aspect_encoding = tokenizer(
    aspect,
    padding='max_length',
    truncation=True,
    max_length=128,
    return_tensors='pt'
)

# Extract input_ids and attention_mask
text_input_ids = text_encoding['input_ids']
text_attention_mask = text_encoding['attention_mask']
aspect_input_ids = aspect_encoding['input_ids']
aspect_attention_mask = aspect_encoding['attention_mask']

# Ensure correct shape
print(f"text_input_ids shape: {text_input_ids.shape}")
print(f"aspect_input_ids shape: {aspect_input_ids.shape}")

# Forward pass through the model
outputs = model(text_input_ids, text_attention_mask, aspect_input_ids, aspect_attention_mask)
print(outputs)
