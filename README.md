# Text Summarization Tool

A simple and effective text summarization application that uses 5 different techniques to create summaries and compares them with Google's Gemini AI to find the best approach.

## 🚀 Features

### 5 Summarization Methods
- **Frequency-based**: Selects sentences with the most important words
- **Position-based**: Prefers sentences at the beginning and end of documents
- **TF-IDF**: Finds sentences with unique important words
- **TextRank**: Selects sentences most similar to others (main themes)
- **Clustering**: Chooses diverse sentences covering different topics

### Key Features
- **Gemini AI Integration**: Compares all summaries against Google Gemini's summary
- **Accuracy Scoring**: Shows which method works best using ROUGE-L F1 scores
- **Easy to Use**: Simple GUI with drag-and-drop file selection
- **Export Results**: Save comparison results to text files
- **No Complex Dependencies**: Uses only basic Python libraries

### Supported File Types
- PDF documents (.pdf)
- Microsoft Word documents (.docx)
- Plain text files (.txt)

## 📋 Requirements

### Python Version
- Python 3.7 or higher

### Dependencies
Install required packages using:
```bash
pip install -r requirements.txt
```

### Essential Dependencies
- `PyPDF2>=3.0.0` - PDF processing
- `python-docx>=0.8.11` - Word document processing
- `rouge-score>=0.1.2` - Evaluation metrics
- `google-generativeai>=0.3.0` - Gemini AI integration
- `nltk>=3.8` - Text processing (optional - has fallback)

## 🛠️ Installation

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Get Gemini API Key** (required for comparison)
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Enter it in the application GUI

## 🚀 Usage

### Running the Application
```bash
python main.py
```

### Simple Workflow
1. **Load Document**: Click "Select Document" to choose a PDF, DOCX, or TXT file
2. **Enter API Key**: Add your Gemini API key in the text field
3. **Adjust Settings**: Set summary length percentage (5-50%)
4. **Run Summarizers**: Click "Run All Summarizers"
5. **View Results**: See all 5 summaries in the main window
6. **Compare Results**: View the comparison window showing the best method
7. **Export**: Save results to a text file

### Configuration

You can modify settings in `config.py`:
- Summary length range (5-50%)
- Maximum sentences per summary
- Gemini model settings
- Window size

## 📁 Project Files

- **`main.py`** - Main application with GUI
- **`core_summarizer.py`** - 5 summarization algorithms
- **`config.py`** - Settings and configuration
- **`requirements.txt`** - Required Python packages

## 📈 How It Works

1. **Document Processing**: Extracts text from PDF, DOCX, or TXT files
2. **5 Summarization Methods**: Each uses a different approach to select important sentences
3. **Gemini Comparison**: Gets a summary from Google's AI for reference
4. **ROUGE Scoring**: Measures how similar each summary is to Gemini's
5. **Best Result**: Shows which method performed best with accuracy score

## 🐛 Troubleshooting

**Can't select files**: Make sure to choose "Supported Documents" in file dialog

**Summarization not working**: 
- Check if dependencies are installed: `pip install -r requirements.txt`
- Make sure your document has enough text (at least a few sentences)

**Gemini API errors**:
- Verify your API key is correct
- Check your internet connection
- Make sure you have API quota remaining

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py

# 1. Click "Select Document" and choose your file
# 2. Enter your Gemini API key
# 3. Click "Run All Summarizers"
# 4. View results and comparison!
```

---

**Simple, Fast, Effective Text Summarization** 🚀
