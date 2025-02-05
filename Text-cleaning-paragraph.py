import re

def clean_text(text):
    """
    Cleans the input text by:
    - Removing extra spaces within lines but preserving newline characters
    - Preserving apostrophes and essential punctuation
    - Removing non-ASCII characters
    - Converting to lowercase

    Args:
        text (str): The raw document text

    Returns:
        str: The cleaned text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Preserve punctuation and apostrophes, remove non-ASCII characters
    # Keep newlines to preserve paragraphs
    text = re.sub(r"[^\x00-\x7F’'a-zA-Z0-9.,!? \n]+", '', text)
    
    # Replace multiple spaces or tabs within lines with a single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Ensure spaces around newlines are trimmed, but keep newlines intact
    text = re.sub(r' *\n *', '\n', text)
    
    return text


def clean_text_from_file(input_file, output_file):
    """
    Opens a text file, cleans its contents, and saves the cleaned text to a new file.
    
    Args:
        input_file (str): The path to the input text file
        output_file (str): The path to the output text file where the cleaned text will be saved
    """

    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()
    
    cleaned_text = clean_text(text)
    
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(cleaned_text)
    
    print(f"Cleaned text has been saved to {output_file}")

input_file = './Documents/raw/Paragraphs.txt'
output_file = './Documents/cleaned/cleaned-Paragraphs.txt'

clean_text_from_file(input_file, output_file)
