import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import nltk
import re
import os
import numpy as np
import threading
import time
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import PyPDF2
from docx import Document
from rouge_score import rouge_scorer
from google import genai
import datetime
import logging

# Import enhanced core summarizer
from core_summarizer import (
    frequency_based_summary, position_based_summary, tfidf_based_summary,
    textrank_summary, clustering_based_summary, get_stop_words, clear_cache
)
from config import get_config, validate_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. NLTK Downloads (Ensuring dependencies are met) ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class IntegratedSummarizerApp:
    """
    Enhanced integrated summarization tool with improved algorithms, async processing,
    and better user experience.
    """
    def __init__(self, master):
        self.master = master
        self.config = get_config()
        
        # Validate configuration
        if not validate_config(self.config):
            messagebox.showerror("Configuration Error", "Invalid configuration detected. Using defaults.")
        
        self.master.title("Enhanced Text Summarization Tool")
        self.master.geometry(self.config['window_size'])
        self.master.configure(bg='#f0f0f0')

        self.full_text = ""
        self.sentences = []
        self.summaries = {}
        self.api_key = tk.StringVar()
        self.summary_size_percent = tk.DoubleVar(value=self.config['default_summary_percent'])
        self.stop_words = get_stop_words()
        
        # Async processing
        self.processing = False
        self.gemini_thread = None
        self.progress_window = None

        self._create_main_gui()

    # ----------------------- 2. File Handling -----------------------
    def read_pdf(self, file_path):
        """Extract text from PDF file."""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = "".join(page.extract_text() + "\n" for page in pdf_reader.pages)
                return text
        except Exception as e:
            messagebox.showerror("Error", f"Error reading PDF: {str(e)}")
            return None
    
    def read_docx(self, file_path):
        """Extract text from Word document."""
        try:
            doc = Document(file_path)
            text = "".join(paragraph.text + "\n" for paragraph in doc.paragraphs)
            return text
        except Exception as e:
            messagebox.showerror("Error", f"Error reading DOCX: {str(e)}")
            return None

    def read_txt(self, file_path):
        """Extract text from TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            messagebox.showerror("Error", f"Error reading TXT: {str(e)}")
            return None

    def select_file(self):
        """Opens a file dialog for document selection and loads text."""
        file_path = filedialog.askopenfilename(
            defaultextension=".txt",
            filetypes=[
                ("Text documents", "*.txt"),
                ("PDF documents", "*.pdf"),
                ("Word documents", "*.docx"),
                ("All files", "*.*")
            ]
        )
        if not file_path:
            return

        file_extension = os.path.splitext(file_path)[1].lower()
        self.file_path_label.config(text=f"File: {os.path.basename(file_path)}")
        self.full_text = ""
        self.sentences = []
        
        if file_extension == '.pdf':
            self.full_text = self.read_pdf(file_path)
        elif file_extension == '.docx':
            self.full_text = self.read_docx(file_path)
        elif file_extension == '.txt':
            self.full_text = self.read_txt(file_path)
        else:
            messagebox.showwarning("Unsupported Format", "Please select a supported file (.txt, .pdf, .docx).")
            return
        
        if self.full_text:
            self.sentences = sent_tokenize(self.full_text)
            self._update_sentence_count()
            messagebox.showinfo("Success", f"Document loaded with {len(self.sentences)} sentences.")
            self.run_button.config(state=tk.NORMAL)
            self.clear_summaries()
    
    def _update_sentence_count(self):
        """Update the sentence count display."""
        if self.sentences:
            num_sentences = self.calculate_summary_size()
            self.sentence_count_label.config(text=f"Will use {num_sentences} sentences")
        else:
            self.sentence_count_label.config(text="")

    # ----------------------- 3. Preprocessing & Utilities -----------------------

    def preprocess_text(self, text):
        """Clean and normalize text for summarization."""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove non-alphanumeric/whitespace
        return text

    def calculate_summary_size(self):
        """Returns the number of sentences for the summary."""
        if not self.sentences:
            return 0
        size = max(1, int(len(self.sentences) * (self.summary_size_percent.get() / 100)))
        return min(size, self.config['max_sentences'])

    # ----------------------- 4. Enhanced Summarization Methods -----------------------
    
    def _show_progress(self, message="Processing..."):
        """Show progress window for long operations."""
        if self.progress_window:
            self.progress_window.destroy()
        
        self.progress_window = tk.Toplevel(self.master)
        self.progress_window.title("Processing")
        self.progress_window.geometry("300x100")
        self.progress_window.configure(bg='#f0f0f0')
        self.progress_window.transient(self.master)
        self.progress_window.grab_set()
        
        # Center the window
        self.progress_window.update_idletasks()
        x = (self.progress_window.winfo_screenwidth() // 2) - (150)
        y = (self.progress_window.winfo_screenheight() // 2) - (50)
        self.progress_window.geometry(f"300x100+{x}+{y}")
        
        tk.Label(self.progress_window, text=message, font=('Arial', 12), 
                bg='#f0f0f0').pack(pady=20)
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(self.progress_window, mode='indeterminate')
        self.progress_bar.pack(pady=10)
        self.progress_bar.start()
    
    def _hide_progress(self):
        """Hide progress window."""
        if self.progress_window:
            self.progress_window.destroy()
            self.progress_window = None

    # ----------------------- 5. Gemini & Evaluation Methods -----------------------

    def get_gemini_summary_async(self, callback):
        """Generate Gemini summary asynchronously."""
        def _gemini_worker():
            try:
                api_key_value = self.api_key.get()
                if not api_key_value:
                    callback("ERROR: Gemini API Key not provided.")
                    return
                
                if not self.full_text:
                    callback("ERROR: No document text loaded.")
                    return

                # Truncate text if too long
                max_chars = self.config['max_chars_for_gemini']
                text_to_summarize = self.full_text[:max_chars]

                client = genai.Client(api_key=api_key_value)
                
                prompt = (
                    "You are an expert summarizer. Generate a concise, objective, and abstractive summary "
                    f"of the following text. The summary should be approximately {self.summary_size_percent.get():.0f}% "
                    "of the original length and must use entirely new sentences where possible. "
                    "Text to summarize:\n\n"
                    f"{text_to_summarize}"
                )
                
                response = client.generate_content(
                    model=self.config['gemini_model'],
                    contents=prompt
                )
                callback(response.text)
                
            except Exception as e:
                callback(f"API ERROR: Failed to call Gemini API. Check your key and network connection. Details: {e}")
        
        self.gemini_thread = threading.Thread(target=_gemini_worker, daemon=True)
        self.gemini_thread.start()

    def calculate_rouge_l(self, reference_summary, candidate_summary):
        """Calculates ROUGE-L F1 score and returns it as a percentage."""
        if not reference_summary or not candidate_summary:
            return 0.0

        # ROUGE-L is used to measure sequence and content overlap (standard for summarization)
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(reference_summary, candidate_summary)
        
        # Return F1-score as a percentage (e.g., 0.35 -> 35.0)
        return scores['rougeL'].fmeasure * 100

    # ----------------------- 6. GUI Logic and Execution -----------------------

    def clear_summaries(self):
        """Clear all summary text boxes."""
        for text_widget in self.text_widgets.values():
            text_widget.config(state=tk.NORMAL)
            text_widget.delete('1.0', tk.END)
            text_widget.config(state=tk.DISABLED)
        self.summaries = {}

    def run_summarization(self):
        """Execute all summarization methods with enhanced error handling."""
        if not self.full_text:
            messagebox.showwarning("Warning", "Please select and load a document first.")
            return

        if self.processing:
            messagebox.showwarning("Warning", "Summarization already in progress.")
            return

        self.processing = True
        num_sentences = self.calculate_summary_size()
        self.clear_summaries()

        # Show progress
        self._show_progress("Running extractive summarizers...")
        self.master.config(cursor="wait")
        self.master.update()

        try:
            # Clear cache for fresh computation
            clear_cache()
            
            # Run the 5 Extractive Summarizers using enhanced core functions
            logger.info(f"Starting summarization with {num_sentences} sentences from {len(self.sentences)} total")
            
            self.summaries['Frequency'] = frequency_based_summary(self.sentences, num_sentences, self.stop_words)
            self.summaries['Position'] = position_based_summary(self.sentences, num_sentences)
            self.summaries['TF-IDF'] = tfidf_based_summary(self.sentences, num_sentences)
            self.summaries['TextRank'] = textrank_summary(self.sentences, num_sentences)
            self.summaries['Clustering'] = clustering_based_summary(self.sentences, num_sentences)

            # Display the 5 results in the main window
            for name, summary in self.summaries.items():
                widget = self.text_widgets[name]
                widget.config(state=tk.NORMAL)
                widget.insert(tk.END, summary)
                widget.config(state=tk.DISABLED)

            # Hide progress and open comparison window
            self._hide_progress()
            self.open_comparison_window()
            
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            messagebox.showerror("Execution Error", f"An error occurred during summarization: {e}")
            self._hide_progress()
            
        finally:
            self.master.config(cursor="")
            self.processing = False

    def open_comparison_window(self):
        """Creates the second window for Gemini and comparison with async processing."""
        
        comparison_win = tk.Toplevel(self.master)
        comparison_win.title("Gemini Abstractive Summary & Comparison")
        comparison_win.geometry("1000x800")
        comparison_win.configure(bg='#e0f7fa')
        
        tk.Label(comparison_win, text="Gemini Abstractive Summary (Reference)", 
                 font=('Arial', 14, 'bold'), bg='#e0f7fa', fg='#00796b').pack(pady=10)

        # --- Gemini Summary Widget ---
        gemini_summary_text = scrolledtext.ScrolledText(comparison_win, wrap=tk.WORD, width=100, height=15, 
                                                        font=('Arial', 10), bg='white', fg='#424242', bd=1, relief=tk.SUNKEN)
        gemini_summary_text.pack(pady=5, padx=20)
        
        # Show loading message
        gemini_summary_text.insert(tk.END, "Loading Gemini summary... Please wait.")
        comparison_win.update()
        
        # Start async Gemini call
        def on_gemini_complete(summary):
            gemini_summary_text.delete('1.0', tk.END)
            gemini_summary_text.insert(tk.END, summary)
            self._update_comparison_results(comparison_win, summary)
        
        self.get_gemini_summary_async(on_gemini_complete)
    
    def _update_comparison_results(self, comparison_win, gemini_summary):
        """Update comparison results after Gemini completes."""
        
        # --- Comparison Section ---
        tk.Label(comparison_win, text="ROUGE-L F1 Score Comparison (vs. Gemini)", 
                 font=('Arial', 14, 'bold'), bg='#e0f7fa', fg='#00796b').pack(pady=20)

        comparison_frame = tk.Frame(comparison_win, bg='#e0f7fa')
        comparison_frame.pack(pady=10)

        headers = ["Technique", "ROUGE-L F1 (%)", "Sentences"]
        results = {}
        best_technique = ""
        max_score = -1.0
        
        if not gemini_summary.startswith("ERROR:"):
            # Calculate scores only if Gemini summary was successful
            for name, summary in self.summaries.items():
                score = self.calculate_rouge_l(gemini_summary, summary)
                sentence_count = len(summary.split('\n')) if summary else 0
                results[name] = (score, sentence_count)
                if score > max_score:
                    max_score = score
                    best_technique = name
        else:
            # If API failed, report the error in the display area
            results = {name: (0.0, 0) for name in self.summaries}

        # Table Headers
        for i, header in enumerate(headers):
            tk.Label(comparison_frame, text=header, font=('Arial', 12, 'bold'), 
                     bg='#00796b', fg='white', width=20, bd=1, relief=tk.RAISED).grid(row=0, column=i, padx=5, pady=5)

        # Table Rows
        row_num = 1
        for name, (score, sentence_count) in results.items():
            color = '#b2dfdb' if name == best_technique and max_score > 0 else 'white'
            tk.Label(comparison_frame, text=name, font=('Arial', 11), 
                     bg=color, fg='black', width=20, bd=1, relief=tk.GROOVE).grid(row=row_num, column=0, padx=5, pady=2)
            tk.Label(comparison_frame, text=f"{score:.2f}", font=('Arial', 11), 
                     bg=color, fg='black', width=20, bd=1, relief=tk.GROOVE).grid(row=row_num, column=1, padx=5, pady=2)
            tk.Label(comparison_frame, text=str(sentence_count), font=('Arial', 11), 
                     bg=color, fg='black', width=20, bd=1, relief=tk.GROOVE).grid(row=row_num, column=2, padx=5, pady=2)
            row_num += 1

        # --- Best Result Display ---
        if best_technique and max_score > 0:
            final_message = f"🏆 Best Summarizer: {best_technique} with an Accuracy (ROUGE-L F1) of {max_score:.2f}%"
        else:
            final_message = "❌ Comparison failed due to an error (see Gemini Summary box)."
            
        tk.Label(comparison_win, text=final_message, 
                 font=('Arial', 15, 'bold'), bg='#e0f7fa', fg='red' if max_score == 0 else '#004d40').pack(pady=20)
        
        # Add export button
        export_btn = tk.Button(comparison_win, text="📊 Export Results", 
                              command=lambda: self._export_results(results, gemini_summary),
                              font=('Arial', 12, 'bold'), bg='#4caf50', fg='white',
                              relief=tk.FLAT, padx=20, pady=8)
        export_btn.pack(pady=10)
    
    def _export_results(self, results, gemini_summary):
        """Export comparison results to a file."""
        try:
            file_path = filedialog.asksaveasfilename(
                title="Export Results",
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("Text Summarization Results\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Total sentences: {len(self.sentences)}\n")
                    f.write(f"Summary length: {self.summary_size_percent.get():.1f}%\n\n")
                    
                    f.write("ROUGE-L F1 Scores:\n")
                    f.write("-" * 30 + "\n")
                    for name, (score, sentence_count) in results.items():
                        f.write(f"{name}: {score:.2f}% ({sentence_count} sentences)\n")
                    
                    f.write(f"\nBest Method: {max(results.items(), key=lambda x: x[1][0])[0]}\n\n")
                    
                    f.write("Gemini Summary (Reference):\n")
                    f.write("-" * 30 + "\n")
                    f.write(gemini_summary + "\n\n")
                    
                    f.write("Extractive Summaries:\n")
                    f.write("-" * 30 + "\n")
                    for name, summary in self.summaries.items():
                        f.write(f"\n{name} Summary:\n")
                        f.write("-" * 20 + "\n")
                        f.write(summary + "\n")
                
                messagebox.showinfo("Success", f"Results exported to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export results: {e}")

    # ----------------------- 7. Main GUI Structure -----------------------

    def _create_main_gui(self):
        """Sets up the main Tkinter window layout."""
        main_frame = tk.Frame(self.master, bg='#f0f0f0')
        main_frame.pack(pady=10, padx=10, fill='both', expand=True)

        # --- Control Panel Frame ---
        control_frame = tk.Frame(main_frame, bg='#e0e0e0', bd=2, relief=tk.GROOVE)
        control_frame.pack(pady=10, fill='x')

        # API Key Input
        tk.Label(control_frame, text="Gemini API Key:", bg='#e0e0e0', font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        api_entry = tk.Entry(control_frame, textvariable=self.api_key, width=30, show='*')
        api_entry.pack(side=tk.LEFT, padx=5)
        
        # Summary Size Slider
        tk.Label(control_frame, text="Summary Length (%):", bg='#e0e0e0', font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(20, 5))
        scale = tk.Scale(control_frame, 
                         from_=self.config['summary_size_range'][0], 
                         to=self.config['summary_size_range'][1], 
                         orient=tk.HORIZONTAL, 
                         variable=self.summary_size_percent, 
                         length=150, bg='#e0e0e0', 
                         troughcolor='#bdbdbd', highlightthickness=0,
                         command=lambda x: self._update_sentence_count())
        scale.pack(side=tk.LEFT, padx=5)
        
        # Add sentence count display
        self.sentence_count_label = tk.Label(control_frame, text="", bg='#e0e0e0', font=('Arial', 9))
        self.sentence_count_label.pack(side=tk.LEFT, padx=(10, 5))

        # File Button and Label
        select_button = tk.Button(control_frame, text="Select Document", command=self.select_file, 
                                  bg='#4caf50', fg='white', font=('Arial', 10, 'bold'))
        select_button.pack(side=tk.LEFT, padx=(20, 5))
        self.file_path_label = tk.Label(control_frame, text="File: (No file loaded)", bg='#e0e0e0')
        self.file_path_label.pack(side=tk.LEFT, padx=5)

        # Run Button
        self.run_button = tk.Button(control_frame, text="Run All Summarizers", command=self.run_summarization, 
                                    state=tk.DISABLED, bg='#2196f3', fg='white', font=('Arial', 10, 'bold'))
        self.run_button.pack(side=tk.RIGHT, padx=10)

        # --- Summaries Grid Frame ---
        summaries_frame = tk.Frame(main_frame, bg='#f0f0f0')
        summaries_frame.pack(fill='both', expand=True, pady=10)
        
        self.text_widgets = {}
        summarizer_names = ['Frequency', 'Position', 'TF-IDF', 'TextRank', 'Clustering']
        
        for i, name in enumerate(summarizer_names):
            row = i // 2
            col = i % 2
            
            # Create a frame for each summary box
            box_frame = tk.LabelFrame(summaries_frame, text=f"{name} Summary", 
                                     font=('Arial', 12, 'bold'), bg='white', fg='#616161')
            box_frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            
            # Create a scrolled text widget for the summary
            text_widget = scrolledtext.ScrolledText(box_frame, wrap=tk.WORD, width=40, height=10, 
                                                    font=('Arial', 9), bg='#f7f7f7', bd=1, relief=tk.FLAT)
            text_widget.pack(fill='both', expand=True, padx=5, pady=5)
            text_widget.config(state=tk.DISABLED) # Make read-only
            self.text_widgets[name] = text_widget
        
        # Add a 6th slot for padding if the number of summaries is odd
        if len(summarizer_names) % 2 != 0:
             tk.Frame(summaries_frame, bg='#f0f0f0').grid(row=len(summarizer_names) // 2, column=1, sticky="nsew")

        # Configure grid expansion
        summaries_frame.grid_rowconfigure(0, weight=1)
        summaries_frame.grid_rowconfigure(1, weight=1)
        summaries_frame.grid_rowconfigure(2, weight=1)
        summaries_frame.grid_columnconfigure(0, weight=1)
        summaries_frame.grid_columnconfigure(1, weight=1)

if __name__ == "__main__":
    root = tk.Tk()
    app = IntegratedSummarizerApp(root)
    root.mainloop()