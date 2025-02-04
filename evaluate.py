import xml.etree.ElementTree as ET
import pandas as pd
import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from torch.utils.data import DataLoader
def parse_test_xml(file_path):
    """
    Parse the SemEval test dataset XML file.
    
    Args:
        file_path (str): Path to the test XML file.
        
    Returns:
        pd.DataFrame: DataFrame with columns ['text', 'aspect', 'category'].
    """
    tree = ET.parse(file_path)
    root = tree.getroot()

    data = []

    for sentence in root.findall('sentence'):
        text = sentence.find('text').text  # Extract the sentence text

        # Extract aspect terms
        aspect_terms = sentence.find('aspectTerms')
        if aspect_terms is not None:
            for aspect in aspect_terms.findall('aspectTerm'):
                term = aspect.get('term')  # Aspect term
                category = None  # Default if no category is present

                # Extract aspect categories if available
                aspect_categories = sentence.find('aspectCategories')
                if aspect_categories is not None:
                    for category_elem in aspect_categories.findall('aspectCategory'):
                        category = category_elem.get('category')

                # Append each aspect term to the dataset
                data.append({'text': text, 'aspect': term, 'category': category})

    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(data)
    return df


# Example usage
file_path = r"C:\Users\Panvith\absa_env\data\Restaurants_Test_Data_phaseB.xml"  # Replace with your file path
test_df = parse_test_xml(file_path)

def preprocess_test_data(df, tokenizer, max_length=128):
    """
    Preprocess the test data for BERT input.

    Args:
        df (pd.DataFrame): DataFrame with columns ['text', 'aspect'].
        tokenizer (BertTokenizer): BERT tokenizer.
        max_length (int): Maximum sequence length for tokenization.

    Returns:
        tuple: Tensors (input_ids, attention_masks, aspect_ids, aspect_masks).
    """
    input_ids, attention_masks, aspect_ids, aspect_masks = [], [], [], []

    for _, row in df.iterrows():
        # Tokenize the text
        text_encoding = tokenizer(
            row['text'],
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        # Tokenize the aspect term
        aspect_encoding = tokenizer(
            row['aspect'],
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        # Collect the tokens and masks
        input_ids.append(text_encoding['input_ids'].squeeze(0))
        attention_masks.append(text_encoding['attention_mask'].squeeze(0))
        aspect_ids.append(aspect_encoding['input_ids'].squeeze(0))
        aspect_masks.append(aspect_encoding['attention_mask'].squeeze(0))

    return (
        torch.stack(input_ids),
        torch.stack(attention_masks),
        torch.stack(aspect_ids),
        torch.stack(aspect_masks)
    )


class TestABSADataset(Dataset):
    def __init__(self, input_ids, attention_masks, aspect_ids, aspect_masks):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.aspect_ids = aspect_ids
        self.aspect_masks = aspect_masks

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'text_input_ids': self.input_ids[idx],
            'text_attention_mask': self.attention_masks[idx],
            'aspect_input_ids': self.aspect_ids[idx],
            'aspect_attention_mask': self.aspect_masks[idx]
        }
def evaluate_model_on_test(model, test_loader, device):
    """
    Evaluate the model on test data and calculate predictions.
    Args:
    - model: Trained model
    - test_loader: DataLoader for test dataset
    - device: Device to run the model on (CPU/GPU)

    Returns:
    - all_preds: Predicted labels
    """
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            # Move data to device
            text_input_ids = batch['text_input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)
            aspect_input_ids = batch['aspect_input_ids'].to(device)
            aspect_attention_mask = batch['aspect_attention_mask'].to(device)

            # Forward pass
            outputs = model(text_input_ids, text_attention_mask, aspect_input_ids, aspect_attention_mask)
            preds = torch.argmax(outputs, dim=1)  # Get predictions

            # Collect predictions
            all_preds.extend(preds.cpu().numpy())

    return all_preds


 

# Example usage
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
test_input_ids, test_attention_masks, test_aspect_ids, test_aspect_masks = preprocess_test_data(
    test_df, tokenizer, max_length=128
)

TEST_BATCH_SIZE = 16

# Create the test dataset
test_dataset = TestABSADataset(
    test_input_ids, 
    test_attention_masks, 
    test_aspect_ids, 
    test_aspect_masks
)

# Create the DataLoader
test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

# Check if DataLoader is working
for batch in test_loader:
    print("Sample batch:")
    print("Text Input IDs:", batch['text_input_ids'].shape)
    print("Text Attention Masks:", batch['text_attention_mask'].shape)
    print("Aspect Input IDs:", batch['aspect_input_ids'].shape)
    print("Aspect Attention Masks:", batch['aspect_attention_mask'].shape)
    break

# Confirm the shape of tensors
print("Test Input IDs Shape:", test_input_ids.shape)
print("Test Attention Masks Shape:", test_attention_masks.shape)
print("Test Aspect IDs Shape:", test_aspect_ids.shape)
print("Test Aspect Masks Shape:", test_aspect_masks.shape)

# Display the DataFrame
print(test_df.head())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. **Load the Trained Model**
from model import ABSA_Model  # Import your ABSA_Model definition
model = ABSA_Model()  # Initialize the model
model.load_state_dict(torch.load("absa_model.pt", map_location=device, weights_only=True)) # Load the trained weights
model = model.to(device)  # Move model to the device
model.eval()  # Set model to evaluation mode

print("Model loaded successfully.")

# 2. **Define Evaluation Function**


# 3. **Evaluate on Test Data**
test_predictions = evaluate_model_on_test(model, test_loader, device)

print("Test Predictions:", test_predictions)

# 4. **Save Predictions**
# Add predictions to the test DataFrame
test_df['predicted_sentiment'] = ['positive' if p == 0 else 'neutral' if p == 1 else 'negative' for p in test_predictions]

# Save the predictions to a CSV file
test_df.to_csv("test_predictions.csv", index=False)
print("Predictions saved to test_predictions.csv")


# Print predictions
print("Test Predictions:", test_predictions)