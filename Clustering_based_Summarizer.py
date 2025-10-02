import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import nltk
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
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

class ClusteringSummarizer:
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
    
    def clustering_based_summary(self, text, num_sentences=5):
        """Generate clustering-based summary"""
        if not text:
            return []
            
        sentences = self.sentence_tokenize(text)
        if len(sentences) <= num_sentences:
            return sentences
        
        try:
            # Create TF-IDF matrix
            vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, max_features=200)
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Perform clustering
            n_clusters = min(num_sentences, len(sentences))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(tfidf_matrix.toarray())
            
            # Select representative sentences from each cluster
            summary_sentences = []
            for cluster_id in range(n_clusters):
                cluster_sentences = [sentences[i] for i in range(len(sentences)) if clusters[i] == cluster_id]
                if cluster_sentences:
                    # Pick the sentence closest to cluster center
                    cluster_indices = [i for i in range(len(sentences)) if clusters[i] == cluster_id]
                    cluster_center = kmeans.cluster_centers_[cluster_id]
                    
                    best_sentence = None
                    best_similarity = -1
                    
                    for idx in cluster_indices:
                        similarity = cosine_similarity([tfidf_matrix[idx].toarray()[0]], [cluster_center])[0][0]
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_sentence = sentences[idx]
                    
                    if best_sentence:
                        summary_sentences.append(best_sentence)
            
            # Maintain original order
            ordered_summary = []
            for sentence in sentences:
                if sentence in summary_sentences:
                    ordered_summary.append(sentence)
            
            return ordered_summary[:num_sentences]
            
        except Exception as e:
            messagebox.showerror("Error", f"Error in clustering processing: {str(e)}")
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
            summary_sentences = self.clustering_based_summary(cleaned_text)
            if summary_sentences:
                self.show_summary(summary_sentences, os.path.basename(file_path))
            else:
                messagebox.showwarning("Warning", "Could not generate summary!")
        except Exception as e:
            messagebox.showerror("Error", f"Error generating summary: {str(e)}")
    
    def show_summary(self, summary_sentences, filename):
        """Display summary in a new window with bullet points"""
        summary_window = tk.Toplevel(self.root)
        summary_window.title(f"Clustering Summary - {filename}")
        summary_window.geometry("800x600")
        summary_window.configure(bg='#fce4ec')
        
        # Create main frame
        main_frame = tk.Frame(summary_window, bg='#fce4ec')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(main_frame, 
                              text="CLUSTERING BASED SUMMARY", 
                              font=('Arial', 16, 'bold'),
                              bg='#fce4ec', fg='#880e4f')
        title_label.pack(pady=(0, 10))
        
        # File info
        info_label = tk.Label(main_frame, 
                             text=f"Source: {filename} | Method: K-Means Clustering | Sentences: {len(summary_sentences)}", 
                             font=('Arial', 10),
                             bg='#fce4ec', fg='#ad1457')
        info_label.pack(pady=(0, 15))
        
        # Summary text area
        text_frame = tk.Frame(main_frame, bg='#fce4ec')
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        summary_text = scrolledtext.ScrolledText(text_frame, 
                                               wrap=tk.WORD, 
                                               width=80, height=25,
                                               font=('Arial', 11),
                                               bg='white', fg='#4a148c',
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
        buttons_frame = tk.Frame(main_frame, bg='#fce4ec')
        buttons_frame.pack(fill=tk.X, pady=(15, 0))
        
        # Save button
        save_btn = tk.Button(buttons_frame, 
                           text="💾 Save Summary", 
                           command=lambda: self.save_summary(filename),
                           font=('Arial', 11, 'bold'),
                           bg='#e91e63', fg='white',
                           relief=tk.FLAT, padx=20, pady=8,
                           cursor='hand2')
        save_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Close button
        close_btn = tk.Button(buttons_frame, 
                            text="✕ Close", 
                            command=summary_window.destroy,
                            font=('Arial', 11, 'bold'),
                            bg='#9c27b0', fg='white',
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
        suggested_name = f"{base_name}_clustering_summary.txt"
        
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
                    file.write(f"CLUSTERING BASED SUMMARY\n")
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
        self.root.title("Clustering-based Text Summarizer")
        self.root.geometry("600x400")
        self.root.configure(bg='#f3e5f5')
        
        # Main container
        main_container = tk.Frame(self.root, bg='#f3e5f5')
        main_container.pack(fill=tk.BOTH, expand=True, padx=30, pady=30)
        
        # Title
        title_label = tk.Label(main_container, 
                              text="CLUSTERING TEXT SUMMARIZER", 
                              font=('Arial', 20, 'bold'),
                              bg='#f3e5f5', fg='#4a148c')
        title_label.pack(pady=(0, 10))
        
        # Subtitle
        subtitle_label = tk.Label(main_container, 
                                 text="Groups similar sentences and selects representative samples", 
                                 font=('Arial', 12),
                                 bg='#f3e5f5', fg='#7b1fa2')
        subtitle_label.pack(pady=(0, 30))
        
        # Instructions
        instructions = """
📄 SUPPORTED FORMATS: PDF, Word (.docx), Text (.txt)

🎯 HOW IT WORKS:
• Groups sentences into clusters using K-Means algorithm
• Each cluster represents a different topic/theme
• Selects the most representative sentence from each cluster
• Ensures topic diversity in the final summary

🔧 INSTRUCTIONS:
1. Click 'Select Document' to choose your file
2. View the generated summary in a new window
3. Save the summary if needed
4. Process another document or press ESC to exit
        """
        
        instructions_label = tk.Label(main_container, 
                                    text=instructions,
                                    font=('Arial', 11),
                                    bg='#f3e5f5', fg='#6a1b9a',
                                    justify=tk.LEFT)
        instructions_label.pack(pady=(0, 30))
        
        
        buttons_frame = tk.Frame(main_container, bg='#f3e5f5')
        buttons_frame.pack(pady=20)
        
        
        select_btn = tk.Button(buttons_frame, 
                             text="📁 Select Document", 
                             command=self.select_file,
                             font=('Arial', 14, 'bold'),
                             bg='#8e24aa', fg='white',
                             relief=tk.FLAT, padx=30, pady=15,
                             cursor='hand2')
        select_btn.pack(pady=10)
        
        
        exit_label = tk.Label(main_container, 
                             text="Press ESC to exit the application", 
                             font=('Arial', 10, 'italic'),
                             bg='#f3e5f5', fg='#9575cd')
        exit_label.pack(side=tk.BOTTOM, pady=(20, 0))
        
        
        self.root.bind('<Escape>', lambda e: self.root.quit())
        self.root.focus_set()
        
        
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
    
    import datetime
    tk.datetime = datetime
    
    required_packages = ['nltk', 'scikit-learn', 'PyPDF2', 'python-docx', 'tkinter']
    print("Clustering-based Text Summarizer")
    print("="*40)
    print("Required packages:", ", ".join(required_packages))
    print("To install: pip install nltk scikit-learn PyPDF2 python-docx")
    print("\nStarting application...")
    print("Press ESC anytime to exit")
    print("="*40)
    
    summarizer = ClusteringSummarizer()
    summarizer.run()

# ----------------------- Importable API for new.py -----------------------

from typing import List as _List


def clustering_based_summary(sentences: _List[str], num_sentences: int) -> str:
    """Return a KMeans clustering-based summary string matching new.py behavior.

    - Vectorizes sentences using TF-IDF
    - Clusters into min(num_sentences, len(sentences)) clusters
    - Picks the sentence closest to each centroid
    - Returns selected sentences in original order joined by \n
    Uses scikit-learn, consistent with new.py.
    """
    if not sentences or num_sentences <= 0:
        return ""

    try:
        import numpy as _np
        from sklearn.feature_extraction.text import TfidfVectorizer as _TfidfVectorizer
        from sklearn.cluster import KMeans as _KMeans
        from sklearn.metrics.pairwise import cosine_similarity as _cosine_similarity
    except Exception:
        return "\n".join(sentences[:num_sentences])

    vectorizer = _TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)

    n_clusters = min(num_sentences, len(sentences))
    if n_clusters < 1:
        return ""

    # KMeans
    try:
        kmeans = _KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    except TypeError:
        # Older scikit-learn versions don't support n_init='auto'
        kmeans = _KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    kmeans.fit(tfidf_matrix)

    selected_indices = []
    for i in range(n_clusters):
        centroid = kmeans.cluster_centers_[i]
        distances = _cosine_similarity([centroid], tfidf_matrix)
        closest_idx = int(_np.argmax(distances))
        if closest_idx not in selected_indices:
            selected_indices.append(closest_idx)

    selected_indices.sort()
    summary = [sentences[i] for i in selected_indices]
    return "\n".join(summary)