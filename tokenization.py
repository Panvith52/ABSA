from transformers import BertTokenizer
import json
import os

def tokenize_data(data, output_file="tokenized_data.json"):
    """
    Tokenize text and aspects using BERT tokenizer and save to a file.

    Args:
        data (list): List of dictionaries with text, aspects, and sentiments.
        output_file (str): Path to save the tokenized data.

    Returns:
        None
    """
    # Initialize the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    tokenized_data = []

    for sample in data:
        # Tokenize the full text
        text_tokens = tokenizer(
            sample['text'],
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        # Tokenize each aspect
        aspect_tokens = [
            tokenizer(
                aspect,
                padding='max_length',
                truncation=True,
                max_length=32,
                return_tensors='pt'
            ) for aspect in sample['aspects']
        ]

        # Append tokenized results to the output
        tokenized_data.append({
            'text_tokens': {
                "input_ids": text_tokens['input_ids'].tolist(),
                "attention_mask": text_tokens['attention_mask'].tolist()
            },
            'aspect_tokens': [
                {
                    "input_ids": aspect_token['input_ids'].tolist(),
                    "attention_mask": aspect_token['attention_mask'].tolist()
                } for aspect_token in aspect_tokens
            ],
            'sentiments': sample['sentiments']
        })

    # Save the tokenized data to the output file
    with open(output_file, 'w') as f:
        json.dump(tokenized_data, f, indent=4)

    print(f"Tokenization completed and saved to {output_file}.")
