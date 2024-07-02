# Bights: Transforming Content into Structured Notes and Questions

Bights is a powerful tool that transforms your content into structured notes and questions. It's designed to handle various input types, including text, URLs, and PDFs. The tool uses advanced language models to analyze the content, generate lecture notes, and create question papers based on the topics discussed and their importance.

## Features

- Supports text, URL, and PDF inputs
- Generates lecture notes and question papers
- Uses advanced language models for content analysis
- Clusters content based on embeddings for better organization
- Supports the use of images in PDFs for enhanced analysis (optional)

## Usage

1. Choose the input type (text, URL, or PDF) from the dropdown menu.
2. Enter the content or upload the file as required.
3. If using a PDF, you can choose to use images in the analysis (may increase processing time).
4. Click "Analyze and Generate" to start the process.
5. Wait for the processing to complete. The tool will display a loading spinner during this time.
6. Once the process is complete, the tool will display a success message and provide a link to download the generated PDF.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/picklehari/practicaly.git
```

2. Install the required dependencies using pipenv:

```bash
pipenv install
```

3. Set up the environment variables by creating a `.env` file in the project root and adding the following:

```
GROQ_API_KEY=your_groq_api_key
```

4. Run the Flask application:

```bash
pipenv run python app.py
```

5. Open `index.html` on your browser to use the tool.

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push your changes to your fork.
5. Submit a pull request.

