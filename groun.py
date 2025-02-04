import xml.etree.ElementTree as ET
import pandas as pd

# Parse the XML file (replace with your actual test XML file path)
tree = ET.parse(r'C:\Users\Panvith\absa_env\data\Restaurants_Test_Data_phaseB.xml')  # Update this path with your actual XML file
root = tree.getroot()

# Initialize empty lists to store data
texts = []
aspects = []
categories = []
sentiments = []  # If you have sentiment labels, otherwise you can add them later

# Iterate over each sentence in the XML
for sentence in root.findall('sentence'):
    text = sentence.find('text').text
    
    # Check if aspectTerms exists before accessing it
    aspect_terms = sentence.find('aspectTerms')
    if aspect_terms is not None:
        for aspectTerm in aspect_terms.findall('aspectTerm'):
            aspect = aspectTerm.get('term')
            category = sentence.find('aspectCategories').find('aspectCategory').get('category')
            
            # If sentiment is available in the XML, extract it, otherwise, you can leave it empty
            # Here, we assume you don't have sentiment for the test data, so we leave it empty for now
            sentiment = 'neutral'  # You can set a default value if needed
            
            # Append the extracted information to lists
            texts.append(text)
            aspects.append(aspect)
            categories.append(category)
            sentiments.append(sentiment)  # Replace with actual sentiment if available

# Create a DataFrame
ground_truth_test_df = pd.DataFrame({
    'text': texts,
    'aspect': aspects,
    'category': categories,
    'sentiment': sentiments
})

# Save the DataFrame to a CSV file
ground_truth_test_df.to_csv('ground_truth_test.csv', index=False)

print("Generated ground_truth_test.csv successfully.")
