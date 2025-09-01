import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import nltk
import re
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import PyPDF2
from docx import Document
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class FrequencyBasedSummarizer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.root = None
        self.current_summary = ""
        
    def read_pdf(self, file_path):
        """Extract text from PDF file"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            messagebox.showerror("Error", f"Error reading PDF: {str(e)}")
            return None
    
    def read_docx(self, file_path):
        """Extract text from Word document"""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            messagebox.showerror("Error", f"Error reading Word document: {str(e)}")
            return None
    
    def read_txt(self, file_path):
        """Extract text from text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            messagebox.showerror("Error", f"Error reading text file: {str(e)}")
            return None
    
    def preprocess_text(self, text):
        """Clean and preprocess the input text"""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', ' ', text)
        return text.strip()
    
    def sentence_tokenize(self, text):
        """Tokenize text into sentences"""
        sentences = sent_tokenize(text)
        return [sent.strip() for sent in sentences if len(sent.strip()) > 10]
    
    def word_tokenize_and_clean(self, text):
        """Tokenize and clean words"""
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalnum() and word not in self.stop_words]
        return words
    
    def frequency_based_summary(self, text, num_sentences=5):
        """Generate frequency-based summary"""
        if not text:
            return []
            
        sentences = self.sentence_tokenize(text)
        if len(sentences) <= num_sentences:
            return sentences
        
        # Calculate word frequencies
        words = self.word_tokenize_and_clean(text)
        word_freq = Counter(words)
        
        # Score sentences based on word frequencies
        sentence_scores = {}
        for sentence in sentences:
            sentence_words = self.word_tokenize_and_clean(sentence)
            score = sum(word_freq[word] for word in sentence_words)
            sentence_scores[sentence] = score / len(sentence_words) if sentence_words else 0
        
        # Select top sentences
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        summary_sentences = [sent[0] for sent in top_sentences[:num_sentences]]
        
        # Maintain original order
        summary_sentences = sorted(summary_sentences, key=lambda x: sentences.index(x))
        return summary_sentences
    
    def select_file(self):
        """Open file dialog to select document"""
        file_types = [
            ('All Supported', '*.pdf;*.docx;*.txt'),
            ('PDF files', '*.pdf'),
            ('Word documents', '*.docx'),
            ('Text files', '*.txt'),
            ('All files', '*.*')
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select Document to Summarize",
            filetypes=file_types
        )
        
        if file_path:
            self.process_file(file_path)
    
    def process_file(self, file_path):
        """Process the selected file and generate summary"""
        # Determine file type and read content
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            text = self.read_pdf(file_path)
        elif file_extension == '.docx':
            text = self.read_docx(file_path)
        elif file_extension == '.txt':
            text = self.read_txt(file_path)
        else:
            messagebox.showerror("Error", "Unsupported file format!")
            return
        
        if not text:
            messagebox.showerror("Error", "Could not extract text from the file!")
            return
        
        # Preprocess text
        cleaned_text = self.preprocess_text(text)
        
        if not cleaned_text:
            messagebox.showerror("Error", "No valid content found in the file!")
            return
        
        # Generate summary
        try:
            summary_sentences = self.frequency_based_summary(cleaned_text)
            if summary_sentences:
                self.show_summary(summary_sentences, os.path.basename(file_path))
            else:
                messagebox.showwarning("Warning", "Could not generate summary!")
        except Exception as e:
            messagebox.showerror("Error", f"Error generating summary: {str(e)}")
    
    def show_summary(self, summary_sentences, filename):
        """Display summary in a new window with bullet points"""
        summary_window = tk.Toplevel(self.root)
        summary_window.title(f"Frequency-based Summary - {filename}")
        summary_window.geometry("800x600")
        summary_window.configure(bg='#f0f0f0')
        
        # Create main frame
        main_frame = tk.Frame(summary_window, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(main_frame, 
                              text="FREQUENCY-BASED SUMMARY", 
                              font=('Arial', 16, 'bold'),
                              bg='#f0f0f0', fg='#2c3e50')
        title_label.pack(pady=(0, 10))
        
        # File info
        info_label = tk.Label(main_frame, 
                             text=f"Source: {filename} | Method: Frequency-based | Sentences: {len(summary_sentences)}", 
                             font=('Arial', 10),
                             bg='#f0f0f0', fg='#7f8c8d')
        info_label.pack(pady=(0, 15))
        
        # Summary text area
        text_frame = tk.Frame(main_frame, bg='#f0f0f0')
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        summary_text = scrolledtext.ScrolledText(text_frame, 
                                               wrap=tk.WORD, 
                                               width=80, height=25,
                                               font=('Arial', 11),
                                               bg='white', fg='#2c3e50',
                                               relief=tk.FLAT, borderwidth=2)
        summary_text.pack(fill=tk.BOTH, expand=True)
        
        # Format summary as bullet points
        formatted_summary = ""
        for i, sentence in enumerate(summary_sentences, 1):
            formatted_summary += f"• {sentence}\n\n"
        
        self.current_summary = formatted_summary
        summary_text.insert(tk.END, formatted_summary)
        summary_text.config(state=tk.DISABLED)
        
        # Buttons frame
        buttons_frame = tk.Frame(main_frame, bg='#f0f0f0')
        buttons_frame.pack(fill=tk.X, pady=(15, 0))
        
        # Save button
        save_btn = tk.Button(buttons_frame, 
                           text="💾 Save Summary", 
                           command=lambda: self.save_summary(filename),
                           font=('Arial', 11, 'bold'),
                           bg='#3498db', fg='white',
                           relief=tk.FLAT, padx=20, pady=8,
                           cursor='hand2')
        save_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Close button
        close_btn = tk.Button(buttons_frame, 
                            text="✕ Close", 
                            command=summary_window.destroy,
                            font=('Arial', 11, 'bold'),
                            bg='#e74c3c', fg='white',
                            relief=tk.FLAT, padx=20, pady=8,
                            cursor='hand2')
        close_btn.pack(side=tk.RIGHT)
        
        # Center the window
        summary_window.transient(self.root)
        summary_window.grab_set()
    
    def save_summary(self, original_filename):
        """Save the summary to a file"""
        if not self.current_summary:
            messagebox.showwarning("Warning", "No summary to save!")
            return
        
        # Suggest filename
        base_name = os.path.splitext(original_filename)[0]
        suggested_name = f"{base_name}_frequency_summary.txt"
        
        file_path = filedialog.asksaveasfilename(
            title="Save Summary",
            defaultextension=".txt",
            initialvalue=suggested_name,
            filetypes=[
                ('Text files', '*.txt'),
                ('All files', '*.*')
            ]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(f"FREQUENCY-BASED SUMMARY\n")
                    file.write(f"Source: {original_filename}\n")
                    file.write(f"Generated on: {tk.datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    file.write("="*50 + "\n\n")
                    file.write(self.current_summary)
                
                messagebox.showinfo("Success", f"Summary saved successfully!\n\nLocation: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving file: {str(e)}")
    
    def create_gui(self):
        """Create the main GUI"""
        self.root = tk.Tk()
        self.root.title("Frequency-based Text Summarizer")
        self.root.geometry("600x400")
        self.root.configure(bg='#ecf0f1')
        
        # Main container
        main_container = tk.Frame(self.root, bg='#ecf0f1')
        main_container.pack(fill=tk.BOTH, expand=True, padx=30, pady=30)
        
        # Title
        title_label = tk.Label(main_container, 
                              text="FREQUENCY-BASED TEXT SUMMARIZER", 
                              font=('Arial', 20, 'bold'),
                              bg='#ecf0f1', fg='#2c3e50')
        title_label.pack(pady=(0, 10))
        
        # Subtitle
        subtitle_label = tk.Label(main_container, 
                                 text="Extract key sentences based on word frequency analysis", 
                                 font=('Arial', 12),
                                 bg='#ecf0f1', fg='#7f8c8d')
        subtitle_label.pack(pady=(0, 30))
        
        # Instructions
        instructions = """
📄 SUPPORTED FORMATS: PDF, Word (.docx), Text (.txt)

🎯 HOW IT WORKS:
• Analyzes word frequency in your document
• Scores sentences based on important word occurrences  
• Selects top-ranking sentences for summary
• Displays results as bullet points

🔧 INSTRUCTIONS:
1. Click 'Select Document' to choose your file
2. View the generated summary in a new window
3. Save the summary if needed
4. Process another document or press ESC to exit
        """
        
        instructions_label = tk.Label(main_container, 
                                    text=instructions,
                                    font=('Arial', 11),
                                    bg='#ecf0f1', fg='#34495e',
                                    justify=tk.LEFT)
        instructions_label.pack(pady=(0, 30))
        
        # Buttons frame
        buttons_frame = tk.Frame(main_container, bg='#ecf0f1')
        buttons_frame.pack(pady=20)
        
        # Select file button
        select_btn = tk.Button(buttons_frame, 
                             text="📁 Select Document", 
                             command=self.select_file,
                             font=('Arial', 14, 'bold'),
                             bg='#27ae60', fg='white',
                             relief=tk.FLAT, padx=30, pady=15,
                             cursor='hand2')
        select_btn.pack(pady=10)
        
        # Exit instruction
        exit_label = tk.Label(main_container, 
                             text="Press ESC to exit the application", 
                             font=('Arial', 10, 'italic'),
                             bg='#ecf0f1', fg='#95a5a6')
        exit_label.pack(side=tk.BOTTOM, pady=(20, 0))
        
        # Bind ESC key to exit
        self.root.bind('<Escape>', lambda e: self.root.quit())
        self.root.focus_set()
        
        # Center window on screen
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def run(self):
        """Run the application"""
        self.create_gui()
        
        while True:
            try:
                self.root.mainloop()
                break
            except tk.TclError:
                break

if __name__ == "__main__":
    # Import datetime for timestamp
    import datetime
    tk.datetime = datetime
    
    # Install required packages message
    required_packages = ['nltk', 'PyPDF2', 'python-docx', 'tkinter']
    print("Frequency-based Text Summarizer")
    print("="*40)
    print("Required packages:", ", ".join(required_packages))
    print("To install: pip install nltk PyPDF2 python-docx")
    print("\nStarting application...")
    print("Press ESC anytime to exit")
    print("="*40)
    
    summarizer = FrequencyBasedSummarizer()
    summarizer.run()