import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import nltk
import re
import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import PyPDF2
from docx import Document
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class TextRankSummarizer:
    def __init__(self):
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
    
    def textrank_summary(self, text, num_sentences=5):
        """Generate TextRank based summary using PageRank algorithm"""
        if not text:
            return []
            
        sentences = self.sentence_tokenize(text)
        if len(sentences) <= num_sentences:
            return sentences
        
        try:
            # Create TF-IDF matrix for similarity calculation
            vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, max_features=500)
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Calculate cosine similarity matrix
            similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
            
            # Create graph and apply PageRank
            nx_graph = nx.from_numpy_array(similarity_matrix)
            
            # Apply PageRank algorithm
            scores = nx.pagerank(nx_graph, max_iter=100, tol=1e-4, alpha=0.85)
            
            # Get top sentences based on PageRank scores
            ranked_sentences = sorted(((scores[i], i) for i in range(len(sentences))), reverse=True)
            top_indices = [ranked_sentences[i][1] for i in range(num_sentences)]
            top_indices = sorted(top_indices)
            
            summary_sentences = [sentences[i] for i in top_indices]
            return summary_sentences
            
        except Exception as e:
            messagebox.showerror("Error", f"Error in TextRank processing: {str(e)}")
            return []
    
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
            summary_sentences = self.textrank_summary(cleaned_text)
            if summary_sentences:
                self.show_summary(summary_sentences, os.path.basename(file_path))
            else:
                messagebox.showwarning("Warning", "Could not generate summary!")
        except Exception as e:
            messagebox.showerror("Error", f"Error generating summary: {str(e)}")
    
    def show_summary(self, summary_sentences, filename):
        """Display summary in a new window with bullet points"""
        summary_window = tk.Toplevel(self.root)
        summary_window.title(f"TextRank Summary - {filename}")
        summary_window.geometry("800x600")
        summary_window.configure(bg='#f1f8e9')
        
        # Create main frame
        main_frame = tk.Frame(summary_window, bg='#f1f8e9')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(main_frame, 
                              text="TEXTRANK BASED SUMMARY", 
                              font=('Arial', 16, 'bold'),
                              bg='#f1f8e9', fg='#2e7d32')
        title_label.pack(pady=(0, 10))
        
        # File info
        info_label = tk.Label(main_frame, 
                             text=f"Source: {filename} | Method: TextRank (PageRank) | Sentences: {len(summary_sentences)}", 
                             font=('Arial', 10),
                             bg='#f1f8e9', fg='#558b2f')
        info_label.pack(pady=(0, 15))
        
        # Summary text area
        text_frame = tk.Frame(main_frame, bg='#f1f8e9')
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        summary_text = scrolledtext.ScrolledText(text_frame, 
                                               wrap=tk.WORD, 
                                               width=80, height=25,
                                               font=('Arial', 11),
                                               bg='white', fg='#1b5e20',
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
        buttons_frame = tk.Frame(main_frame, bg='#f1f8e9')
        buttons_frame.pack(fill=tk.X, pady=(15, 0))
        
        # Save button
        save_btn = tk.Button(buttons_frame, 
                           text="💾 Save Summary", 
                           command=lambda: self.save_summary(filename),
                           font=('Arial', 11, 'bold'),
                           bg='#4caf50', fg='white',
                           relief=tk.FLAT, padx=20, pady=8,
                           cursor='hand2')
        save_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Close button
        close_btn = tk.Button(buttons_frame, 
                            text="✕ Close", 
                            command=summary_window.destroy,
                            font=('Arial', 11, 'bold'),
                            bg='#f44336', fg='white',
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
        suggested_name = f"{base_name}_textrank_summary.txt"
        
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
                    file.write(f"TEXTRANK BASED SUMMARY\n")
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
        self.root.title("TextRank Text Summarizer")
        self.root.geometry("600x400")
        self.root.configure(bg='#e8f5e8')
        
        # Main container
        main_container = tk.Frame(self.root, bg='#e8f5e8')
        main_container.pack(fill=tk.BOTH, expand=True, padx=30, pady=30)
        
        # Title
        title_label = tk.Label(main_container, 
                              text="TEXTRANK TEXT SUMMARIZER", 
                              font=('Arial', 20, 'bold'),
                              bg='#e8f5e8', fg='#1b5e20')
        title_label.pack(pady=(0, 10))
        
        # Subtitle
        subtitle_label = tk.Label(main_container, 
                                 text="Graph-based ranking using PageRank algorithm for sentence extraction", 
                                 font=('Arial', 12),
                                 bg='#e8f5e8', fg='#388e3c')
        subtitle_label.pack(pady=(0, 30))
        
        # Instructions
        instructions = """
📄 SUPPORTED FORMATS: PDF, Word (.docx), Text (.txt)

🎯 HOW IT WORKS:
• Creates a graph of sentences based on similarity
• Applies PageRank algorithm (used by Google) 
• Ranks sentences by their centrality in the graph
• Selects highest-ranked sentences for summary

🔧 INSTRUCTIONS:
1. Click 'Select Document' to choose your file
2. View the generated summary in a new window
3. Save the summary if needed
4. Process another document or press ESC to exit
        """
        
        instructions_label = tk.Label(main_container, 
                                    text=instructions,
                                    font=('Arial', 11),
                                    bg='#e8f5e8', fg='#2e7d32',
                                    justify=tk.LEFT)
        instructions_label.pack(pady=(0, 30))
        
        # Buttons frame
        buttons_frame = tk.Frame(main_container, bg='#e8f5e8')
        buttons_frame.pack(pady=20)
        
        # Select file button
        select_btn = tk.Button(buttons_frame, 
                             text="📁 Select Document", 
                             command=self.select_file,
                             font=('Arial', 14, 'bold'),
                             bg='#43a047', fg='white',
                             relief=tk.FLAT, padx=30, pady=15,
                             cursor='hand2')
        select_btn.pack(pady=10)
        
        # Exit instruction
        exit_label = tk.Label(main_container, 
                             text="Press ESC to exit the application", 
                             font=('Arial', 10, 'italic'),
                             bg='#e8f5e8', fg='#689f38')
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
    required_packages = ['nltk', 'scikit-learn', 'networkx', 'PyPDF2', 'python-docx', 'tkinter']
    print("TextRank Text Summarizer")
    print("="*40)
    print("Required packages:", ", ".join(required_packages))
    print("To install: pip install nltk scikit-learn networkx PyPDF2 python-docx")
    print("\nStarting application...")
    print("Press ESC anytime to exit")
    print("="*40)
    
    summarizer = TextRankSummarizer()
    summarizer.run()