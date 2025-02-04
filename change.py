import pandas as pd

# Step 1: Load the ground truth data from the CSV file
ground_truth_df = pd.read_csv("ground_truth_test.csv")

# Step 2: Define a function to assign sentiment based on specific keywords
def assign_sentiment(text):
    # Define keywords for positive, negative, and neutral sentiments
    positive_keywords = ['good', 'great', 'excellent', 'best', 'amazing', 'love','top']
    negative_keywords = ['bad', 'horrible', 'worst', 'terrible', 'awful', 'disappointing']
    
    # Check for positive sentiment based on keywords
    if any(keyword in text.lower() for keyword in positive_keywords):
        return 'positive'
    # Check for negative sentiment based on keywords
    elif any(keyword in text.lower() for keyword in negative_keywords):
        return 'negative'
    # Default to neutral if no positive or negative keywords are found
    else:
        return 'neutral'

# Step 3: Apply the function to the 'text' column to assign sentiment
ground_truth_df['sentiment'] = ground_truth_df['text'].apply(assign_sentiment)

# Step 4: Save the updated DataFrame to a new CSV file
ground_truth_df.to_csv("modified_ground_truth_test.csv", index=False)

# Print the first few rows of the updated DataFrame to verify the changes
print(ground_truth_df.head())
