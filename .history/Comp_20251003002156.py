
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import os
import re
import nltk
import PyPDF2
from docx import Document
from nltk.tokenize import sent_tokenize
from rouge_score import rouge_scorer

# Download NLTK resources if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from Clustering_based_Summarizer import ClusteringSummarizer
from Frequency_based_Summarizer import FrequencyBasedSummarizer
from Position_based_Summarizer import PositionBasedSummarizer
from TextRank_Summarizer import TextRankSummarizer
from TF_IDF_Summarizer import TFIDFSummarizer


def read_file_content(file_path):
    """Extract text from PDF, DOCX, or TXT file."""
    file_extension = os.path.splitext(file_path)[1].lower()
    text = None
    
    try:
        if file_extension == '.pdf':
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                text = "\n".join(page.extract_text() or "" for page in pdf_reader.pages)
        elif file_extension == '.docx':
            doc = Document(file_path)
            text = "\n".join(paragraph.text or "" for paragraph in doc.paragraphs)
        elif file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        else:
            messagebox.showerror("Error", f"Unsupported file format: {file_extension}")
            return None

    except Exception as e:
        messagebox.showerror("File Read Error", f"Error reading {file_path}: {str(e)}")
        return None
    
    if not text:
        messagebox.showwarning("Warning", "Could not extract any text from the file.")
        return None
        
    return text.strip()

def preprocess_text(text):
    """Clean and preprocess the input text (adapted from your files)"""
    
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', ' ', text)
    return text.strip()

def evaluate_summary(reference, candidate):
    """
    Evaluates a candidate summary against a reference summary using ROUGE-L.
    ROUGE-L measures the longest common subsequence (LCS) match.
    """
    if not reference or not candidate:
        return 0.0

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    
    return scores['rougeL'].fmeasure


class SummarizerComparator:
    def __init__(self):
       
        self.root = tk.Tk()
        self.root.withdraw() 
        
    
        self.summarizers = {
            "Clustering-based": ClusteringSummarizer(),
            "Frequency-based": FrequencyBasedSummarizer(),
            "Position-based": PositionBasedSummarizer(),
            "TextRank": TextRankSummarizer(),
            "TF-IDF": TFIDFSummarizer(),
        }
        
        self.summary_methods = {
            "Clustering-based": lambda text, n: self.summarizers["Clustering-based"].clustering_based_summary(text, n),
            "Frequency-based": lambda text, n: self.summarizers["Frequency-based"].frequency_based_summary(text, n),
            "Position-based": lambda text, n: self.summarizers["Position-based"].position_based_summary(text, n),
            "TextRank": lambda text, n: self.summarizers["TextRank"].textrank_summary(text, n),
            "TF-IDF": lambda text, n: self.summarizers["TF-IDF"].tfidf_based_summary(text, n),
        }
        
    def run_comparison(self, num_sentences=5):
        
        messagebox.showinfo("Start", "Please select the document (PDF, DOCX, or TXT) you wish to summarize.")
        doc_file_path = filedialog.askopenfilename(
            title="Select Document to Summarize",
            filetypes=[('Supported Documents', '*.pdf;*.docx;*.txt')]
        )
        if not doc_file_path: return

        raw_text = read_file_content(doc_file_path)
        if not raw_text: return
        cleaned_text = preprocess_text(raw_text)

        results = {}
        for name, method in self.summary_methods.items():
            try:
                summary_sentences = method(cleaned_text, num_sentences)
                summary_text = " ".join(summary_sentences)
                results[name] = {"summary": summary_text, "score": 0.0}
            except Exception as e:
                results[name] = {"summary": f"ERROR: {name} failed: {e}", "score": 0.0}
        
        
        messagebox.showinfo("Reference Summary Required", 
                            "The next step requires a **Human-Written Reference Summary** (Ground Truth) to measure 'accuracy'.\n"
                            "Please select a file containing the ideal summary for evaluation.")

        ref_file_path = filedialog.askopenfilename(
            title="Select Human-Written Reference Summary (Ground Truth)",
            filetypes=[('Text files', '*.txt'), ('All files', '*.*')]
        )
        
        reference_text = None
        if ref_file_path:
            reference_text = read_file_content(ref_file_path)
            if reference_text:
                reference_text = preprocess_text(reference_text)

        
        best_score = -1.0
        best_technique = "N/A"
        
        if reference_text:
            for name, result in results.items():
                if not result["summary"].startswith("ERROR"):
                    score = evaluate_summary(reference_text, result["summary"])
                    results[name]["score"] = score
                    
                    if score > best_score:
                        best_score = score
                        best_technique = name
        
        
        self.show_results(doc_file_path, results, best_technique, reference_text)


    def show_results(self, doc_path, results, best_technique, reference_text):
        """Displays the results in a dedicated Tkinter window."""
        
        result_window = tk.Toplevel(self.root)
        result_window.title(f"Comparison Results for: {os.path.basename(doc_path)}")
        result_window.geometry("1400x800")
        
        
        title_label = tk.Label(result_window, text="TEXT SUMMARIZATION COMPARISON", font=('Arial', 20, 'bold'), fg='#1A237E', pady=10)
        title_label.pack(pady=(10, 5))
        
        
        if reference_text:
            best_label_text = f"🥇 MOST ACCURATE TECHNIQUE (ROUGE-L F1-Score): {best_technique}"
            best_label = tk.Label(result_window, text=best_label_text, font=('Arial', 18, 'bold'), fg='#2E7D32', pady=10)
            best_label.pack(pady=(0, 10))
        else:
            warning_label_text = "⚠️ NO ACCURACY EVALUATION PERFORMED: Cannot compare without a Human-Written Reference Summary."
            warning_label = tk.Label(result_window, text=warning_label_text, font=('Arial', 14, 'bold'), fg='#FF9800', pady=10)
            warning_label.pack(pady=(0, 10))
            
        
        if reference_text:
            ref_frame = tk.LabelFrame(result_window, text="Human-Written Reference Summary (Ground Truth)", font=('Arial', 12, 'bold'), fg='#0D47A1')
            ref_frame.pack(fill=tk.X, padx=20, pady=5)
            ref_text_area = scrolledtext.ScrolledText(ref_frame, height=5, wrap=tk.WORD, font=('Arial', 10))
            ref_text_area.insert(tk.END, reference_text)
            ref_text_area.configure(state='disabled')
            ref_text_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
        
        results_frame = tk.Frame(result_window)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        
        techniques_list = list(results.keys())
        for i in range(len(techniques_list)):
            results_frame.grid_columnconfigure(i, weight=1)
        
        
        for i, name in enumerate(techniques_list):
            result = results[name]
            score_display = f"ROUGE-L F1: {result['score']:.4f}" if reference_text else "(No Score)"
            
            summary_frame = tk.LabelFrame(results_frame, text=f"{name}\n{score_display}", font=('Arial', 12, 'bold'), fg='#424242')
            summary_frame.grid(row=0, column=i, padx=10, pady=10, sticky="nsew")
            
            summary_text = scrolledtext.ScrolledText(summary_frame, height=20, wrap=tk.WORD, font=('Arial', 10))
            summary_text.insert(tk.END, result['summary'])
            summary_text.configure(state='disabled')
            summary_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)


if __name__ == "__main__":
    app = SummarizerComparator()
    app.run_comparison(num_sentences=5)
   