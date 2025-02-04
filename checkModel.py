import torch
from transformers import BertForSequenceClassification

# Define the path to your saved model
MODEL_PATH = r"C:\Users\Panvith\absa_env\absa_model.pt"

# Load the pre-trained BERT model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

# Load the saved weights
state_dict = torch.load(MODEL_PATH)

# Get the current model's state_dict
model_state_dict = model.state_dict()

# Remove the classifier weights from the saved state_dict (to avoid the mismatch)
state_dict = {k: v for k, v in state_dict.items() if "classifier" not in k}

# Update the current model's state_dict with the compatible weights
model_state_dict.update(state_dict)

# Load the weights into the model
model.load_state_dict(model_state_dict)

# Verify that the model loaded correctly
print("Model Architecture:")
print(model)
