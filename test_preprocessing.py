import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

# Step 1: Load the XML File
# Update the path to your XML file
dataset_path = r"C:\Users\Panvith\absa_env\data\Restaurants_Test_Data_phaseB.xml"

# Parse the XML file
tree = ET.parse(dataset_path)
root = tree.getroot()

# Step 2: Extract Data from XML
# Store data in a list of dictionaries
data = []
for sentence in root.findall("sentence"):
    text = sentence.find("text").text
    aspect_terms = sentence.find("aspectTerms")
    if aspect_terms is not None:
        for aspect_term in aspect_terms.findall("aspectTerm"):
            aspect = aspect_term.get("term")
            # For now, assign a placeholder for sentiment
            # Replace this with actual sentiment labels if available
            sentiment = "neutral"
            data.append({"Sentence": text, "Aspect Term": aspect, "Sentiment": sentiment})

# Display a sample of extracted data
print("Extracted Data:", data[:5])

# Step 3: Initialize the BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Step 4: Define Sentiment Mapping
sentiment_map = {"positive": 0, "neutral": 1, "negative": 2}

# Step 5: Preprocess the Data
def preprocess_data(row):
    """
    Preprocess a single data entry:
    - Tokenize the sentence and the aspect term
    - Create input_ids, attention_mask, and labels
    """
    sentence = row["Sentence"]
    aspect = row["Aspect Term"]
    
    # Handle missing sentiment or assign default sentiment
    sentiment = row.get("Sentiment", "neutral")  # Default to 'neutral' if no sentiment is found

    # Tokenize the sentence with the aspect term
    encoding = tokenizer(
        sentence,
        aspect,
        truncation=True,
        max_length=128,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Map sentiment label to integer
    if sentiment not in sentiment_map:
        print(f"Warning: Sentiment '{sentiment}' not found in sentiment_map. Using 'neutral'.")
        sentiment = "neutral"  # Default to 'neutral' if sentiment is invalid
    
    label = sentiment_map[sentiment]

    return {
        "input_ids": encoding["input_ids"].squeeze(0),
        "attention_mask": encoding["attention_mask"].squeeze(0),
        "labels": torch.tensor(label, dtype=torch.long)
    }

# Apply preprocessing
processed_data = [preprocess_data(row) for row in data]

# Step 6: Create a Custom Dataset Class
class ABSADataset(Dataset):
    def __init__(self, processed_data):
        self.processed_data = processed_data

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]

# Step 7: Convert to PyTorch Dataset and Save
test_dataset = ABSADataset(processed_data)

# Save the dataset as a .pt file
torch.save(test_dataset, "test_dataset.pt")
print("Preprocessed test dataset saved as 'test_dataset.pt'")
