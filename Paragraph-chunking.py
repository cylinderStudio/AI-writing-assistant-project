def chunk_paragraphs(text):
    """
    Breaks the input text into paragraphs. Paragraphs are determined by double newlines.

    Args:
        text (str): The cleaned document text

    Returns:
        list: A list of paragraphs (strings)
    """
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    return paragraphs

def chunk_paragraphs_from_file(input_file, output_file):
    """
    Opens a text file, chunks its contents into paragraphs, and saves the chunked paragraphs to a new file.

    Args:
        input_file (str): The path to the input text file
        output_file (str): The path to the output text file where chunked paragraphs will be saved
    """

    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()
    
    paragraphs = chunk_paragraphs(text)
    
    with open(output_file, 'w', encoding='utf-8') as file:
        for paragraph in paragraphs:
            file.write(paragraph + '\n')  # Write each paragraph on a new line
    
    print(f"Chunked paragraphs have been saved to {output_file}")

input_file = './Documents/cleaned/cleaned-Paragraphs.txt'
output_file = './Documents/chunked/chunked-Paragraphs.txt'

chunk_paragraphs_from_file(input_file, output_file)
