import torch
from transformers import BertTokenizer
from model import ABSA_Model  # Import your ABSA_Model class

def load_model(model_path):
    # Define the model architecture
    model = ABSA_Model()
    
    # Load the model weights
    model.load_state_dict(torch.load(model_path))
    
    # Set the model to evaluation mode
    model.eval()
    
    return model

def prepare_input(text, aspect, tokenizer):
    # Tokenize the text and aspect
    text_encoding = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    aspect_encoding = tokenizer(aspect, padding=True, truncation=True, max_length=128, return_tensors="pt")
    
    # Extract the input_ids and attention_mask for both text and aspect
    text_input_ids = text_encoding['input_ids']
    text_attention_mask = text_encoding['attention_mask']
    aspect_input_ids = aspect_encoding['input_ids']
    aspect_attention_mask = aspect_encoding['attention_mask']
    
    return text_input_ids, text_attention_mask, aspect_input_ids, aspect_attention_mask

def make_prediction(model, text_input_ids, text_attention_mask, aspect_input_ids, aspect_attention_mask):
    # Perform inference
    with torch.no_grad():
        outputs = model(text_input_ids, text_attention_mask, aspect_input_ids, aspect_attention_mask)
        
    # Assuming the model outputs a classification score (positive, negative, neutral)
    prediction = torch.argmax(outputs, dim=-1)
    return prediction.item()

if __name__ == "__main__":
    # Load model
    model = load_model("C:/Users/Panvith/absa_env/absa_model.pt")  # Path to your saved model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Example input
    text = "The food at this restaurant is amazing."
    aspect = "food"

    # Prepare the input data
    text_input_ids, text_attention_mask, aspect_input_ids, aspect_attention_mask = prepare_input(text, aspect, tokenizer)

    # Make prediction
    prediction = make_prediction(model, text_input_ids, text_attention_mask, aspect_input_ids, aspect_attention_mask)
    
    # Output the predicted sentiment class
    print("Predicted Sentiment: ", prediction)  # If you want a class name, map it
