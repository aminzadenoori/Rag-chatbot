from bs4 import BeautifulSoup

# Load your HTML file
html_file_path = './user-manual-en 2.html'
output_text_file = './output_text_file.txt'

# Read the HTML file
with open(html_file_path, 'r', encoding='utf-8') as file:
    html_content = file.read()

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(html_content, 'html.parser')

# Find the main content
main_content = soup.find('div', {'id': 'main'})

# Extract text from h1, h2, h3, h4, p, and li tags within main_content
content = []
if main_content:
    for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'li']):
        text = element.get_text(separator=' ', strip=True)
        if text:
            content.append(text)
else:
    print("Main content not found in the HTML file.")

# Join the content ensuring proper spacing between blocks
text_content = '\n\n'.join(content)

# Save the extracted text to a text file
with open(output_text_file, 'w', encoding='utf-8') as output_file:
    output_file.write(text_content)

print(f"Text extracted and saved to {output_text_file}")
