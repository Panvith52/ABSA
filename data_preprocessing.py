import os
import xml.etree.ElementTree as ET

def preprocess_data(dataset_path):
    """
    Load and preprocess data from the XML file.
    
    Args:
        dataset_path (str): Path to the dataset folder containing the XML file.

    Returns:
        list: List of dictionaries with text, aspects, and sentiments.
    """
    xml_file = os.path.join(dataset_path, 'Restaurants_Train.xml')
    print(f"XML File Path: {xml_file}")  # Debug: Check file path

    # Check if the file exists
    if not os.path.exists(xml_file):
        print("Error: XML file not found!")
        return []

    # Parse the XML file
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return []

    data = []

    # Iterate through sentences in the XML
    for sentence in root.findall('sentence'):
        text = sentence.find('text')
        if text is None:
            continue

        text = text.text
        aspects, sentiments = [], []

        # Look for AspectTerms instead of Opinions
        aspect_terms = sentence.find('aspectTerms')
        if aspect_terms is not None:
            for aspect_term in aspect_terms.findall('aspectTerm'):
                aspect = aspect_term.attrib.get('term')  # Extract 'term' attribute
                sentiment = aspect_term.attrib.get('polarity')  # Extract 'polarity' attribute

                if aspect and sentiment:
                    aspects.append(aspect)
                    sentiments.append(sentiment)
                    #print(f"Extracted Aspect: {aspect}, Sentiment: {sentiment}")  # Debug

        if aspects and sentiments:
            data.append({
                'text': text,
                'aspects': aspects,
                'sentiments': sentiments
            })

    print(f"Total Samples Processed: {len(data)}")  # Debug: Final count
    return data
