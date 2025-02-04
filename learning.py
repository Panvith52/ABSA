import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

# Example DataLoader (replace with your actual dataset loader)
def get_dataloader():
    # Dummy data for illustration purposes
    input_ids = torch.randint(0, 100, (100, 50))
    attention_masks = torch.ones_like(input_ids)
    labels = torch.randint(0, 3, (100,))  # 3 sentiment classes

    dataset = TensorDataset(input_ids, attention_masks, labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    return dataloader

# Training function
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_accuracy = 0
    
    for batch in dataloader:
        input_ids, attention_masks, labels = [x.to(device) for x in batch]
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_masks, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).detach().cpu()
        total_accuracy += accuracy_score(labels.cpu(), preds)

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
    return avg_loss, avg_accuracy

# Learning rate tuning
def tune_learning_rate(learning_rates, num_epochs=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_dataloader()
    results = {}

    for lr in learning_rates:
        print(f"\nTraining with learning rate: {lr}")
        
        # Load model
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3).to(device)
        optimizer = AdamW(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            avg_loss, avg_accuracy = train_model(model, dataloader, optimizer, criterion, device)
            print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

        results[lr] = (avg_loss, avg_accuracy)

    return results

# Define learning rates to test
learning_rates = [1e-5, 3e-5, 5e-5, 1e-4]
results = tune_learning_rate(learning_rates)

# Print results
print("\nLearning Rate Tuning Results:")
for lr, (loss, accuracy) in results.items():
    print(f"Learning Rate: {lr} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
