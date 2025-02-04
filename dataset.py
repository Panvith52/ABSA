import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import xml.etree.ElementTree as ET

# Step 1: Load the Dataset
def load_data(file_path):
    """
    Load data from an XML file.

    :param file_path: Path to the XML dataset file.
    :return: List of dictionaries with keys 'text', 'aspect_tokens', and 'labels'.
    """
    data = []
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Parse the XML structure
    for review in root.findall('review'):
        text = review.find('text').text.strip()
        aspects = []
        labels = []

        for opinion in review.findall('opinions/opinion'):
            aspect = opinion.get('category')  # Replace 'category' with the appropriate XML attribute
            sentiment = opinion.get('polarity')  # Replace 'polarity' with the appropriate XML attribute

            # Map sentiment to numerical label (modify as needed)
            if sentiment == "positive":
                label = 1
            elif sentiment == "negative":
                label = 0
            else:  # Assuming 'neutral' or other
                label = 2

            aspects.append(aspect)
            labels.append(label)

        # Append the processed entry
        data.append({
            "text": text,
            "aspect_tokens": aspects,
            "labels": labels,
        })

    return data

# Step 2: Preprocess Data
def preprocess_data(data, tokenizer, max_length=512):
    """
    Preprocesses the dataset for aspect-based sentiment analysis.

    :param data: List of data samples.
    :param tokenizer: BERT tokenizer.
    :param max_length: Maximum sequence length.
    :return: Processed dataset as a PyTorch Dataset object.
    """
    return ABSA_Dataset(data, tokenizer, max_length)

# Step 3: Define ABSA_Dataset
class ABSA_Dataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        """
        :param data: List of samples (dictionaries containing 'text', 'aspect_tokens', 'labels').
        :param tokenizer: BERT tokenizer.
        :param max_length: Maximum sequence length.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        text = sample['text']
        aspect_tokens = sample['aspect_tokens']
        labels = sample['labels']  # Labels should match the number of aspects

        # Tokenize the text
        text_encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        text_input_ids = text_encoding['input_ids'].squeeze(0)  # Shape: (max_length)
        text_attention_mask = text_encoding['attention_mask'].squeeze(0)  # Shape: (max_length)

        # Tokenize each aspect
        aspect_input_ids = []
        aspect_attention_mask = []
        for aspect in aspect_tokens:
            aspect_encoding = self.tokenizer(
                aspect,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            aspect_input_ids.append(aspect_encoding['input_ids'].squeeze(0))  # Shape: (max_length)
            aspect_attention_mask.append(aspect_encoding['attention_mask'].squeeze(0))  # Shape: (max_length)

        # Stack aspect tokens
        aspect_input_ids = torch.stack(aspect_input_ids)  # Shape: (num_aspects, max_length)
        aspect_attention_mask = torch.stack(aspect_attention_mask)  # Shape: (num_aspects, max_length)

        # Convert labels to tensor
        labels = torch.tensor(labels, dtype=torch.long)  # Shape: (num_aspects)

        return {
            'text_input_ids': text_input_ids,
            'text_attention_mask': text_attention_mask,
            'aspect_input_ids': aspect_input_ids,
            'aspect_attention_mask': aspect_attention_mask,
            'labels': labels
        }

# Step 4: Example Usage
if __name__ == "__main__":
    # File path and tokenizer
    dataset_path = r"C:\Users\Panvith\absa_env\data\Restaurants_Train.xml"  # Replace with your XML dataset file path
    tokenizer_path = "bert-base-uncased"  # Replace with your tokenizer path
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    # Load and preprocess data
    raw_data = load_data(dataset_path)
    processed_dataset = preprocess_data(raw_data, tokenizer)

    # Create DataLoader
    dataloader = DataLoader(processed_dataset, batch_size=16, shuffle=True)

    # Iterate through DataLoader
    for batch in dataloader:
        print("Text Input IDs:", batch["text_input_ids"].shape)
        print("Text Attention Mask:", batch["text_attention_mask"].shape)
        print("Aspect Input IDs:", batch["aspect_input_ids"].shape)
        print("Aspect Attention Mask:", batch["aspect_attention_mask"].shape)
        print("Labels:", batch["labels"].shape)
        break
