import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import nltk
import re
import os
import numpy as np
import networkx as nx
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import PyPDF2
from docx import Document
from rouge_score import rouge_scorer
from google import genai
import datetime

# Import pure summarizer functions for delegation
from Frequency_based_Summarizer import frequency_based_summary as _frequency_based_summary
from Position_based_Summarizer import position_based_summary as _position_based_summary
from TF_IDF_Summarizer import tfidf_based_summary as _tfidf_based_summary
from TextRank_Summarizer import textrank_summary as _textrank_summary
from Clustering_based_Summarizer import clustering_based_summary as _clustering_based_summary

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
    Combines five extractive summarization methods with Gemini's abstractive 
    method and compares the results using ROUGE-L.
    """
    def __init__(self, master):
        self.master = master
        self.master.title("Integrated Summarization Tool")
        self.master.geometry("800x600")
        self.master.configure(bg='#f0f0f0')

        self.full_text = ""
        self.sentences = []
        self.summaries = {}
        self.api_key = tk.StringVar()
        self.summary_size_percent = tk.DoubleVar(value=20.0)
        self.stop_words = set(stopwords.words('english'))

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
            messagebox.showinfo("Success", f"Document loaded with {len(self.sentences)} sentences.")
            self.run_button.config(state=tk.NORMAL)
            self.clear_summaries()

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
        return max(1, int(len(self.sentences) * (self.summary_size_percent.get() / 100)))

    # ----------------------- 4. Summarization Methods (User's 5 Techniques) -----------------------

    def frequency_based_summary(self, num_sentences):
        """Summarization based on word frequency."""
        if not self.full_text: return ""
        
        cleaned_text = self.preprocess_text(self.full_text)
        words = word_tokenize(cleaned_text)
        
        # Calculate word frequency, excluding stop words
        word_frequencies = Counter(word for word in words if word not in self.stop_words)
        
        # Calculate sentence score based on contained word frequencies
        sentence_scores = {}
        for i, sentence in enumerate(self.sentences):
            score = sum(word_frequencies.get(word, 0) for word in word_tokenize(self.preprocess_text(sentence)))
            sentence_scores[i] = score

        # Select top N sentences
        sorted_sentences = sorted(sentence_scores.items(), key=lambda item: item[1], reverse=True)
        top_sentence_indices = [index for index, score in sorted_sentences[:num_sentences]]
        
        # Reconstruct summary in original order
        summary = [self.sentences[i] for i in sorted(top_sentence_indices)]
        return "\n".join(summary)

    def position_based_summary(self, num_sentences):
        """Summarization based on sentence position (extractive)."""
        # Simplest form: assume most important sentences are at the beginning/end
        if not self.sentences: return ""
        
        # Take a proportional amount from the start, middle, and end, but for simplicity
        # and to match the 'extractive' nature, we often just take the first N.
        # Let's take the first N sentences for a pure position-based model.
        top_sentences = self.sentences[:num_sentences]
        return "\n".join(top_sentences)

    def tfidf_based_summary(self, num_sentences):
        """Summarization based on TF-IDF score."""
        if not self.sentences: return ""

        # TF-IDF model
        vectorizer = TfidfVectorizer(stop_words='english', norm='l1')
        tfidf_matrix = vectorizer.fit_transform(self.sentences)
        
        # Calculate sentence score as the sum of its words' TF-IDF values (or max value)
        # Using the sum of feature weights for each sentence row
        sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)
        
        # Select top N sentences
        ranked_sentences = sorted(enumerate(sentence_scores), key=lambda x: x[1], reverse=True)
        top_sentence_indices = [index for index, score in ranked_sentences[:num_sentences]]
        
        # Reconstruct summary
        summary = [self.sentences[i] for i in sorted(top_sentence_indices)]
        return "\n".join(summary)

    def textrank_summary(self, num_sentences):
        """Summarization using the TextRank graph algorithm."""
        if not self.sentences: return ""
        
        # TF-IDF and Cosine Similarity for graph edge weights
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(self.sentences)
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # Build the graph
        nx_graph = nx.from_numpy_array(similarity_matrix)
        
        # Apply PageRank algorithm (TextRank is based on PageRank)
        scores = nx.pagerank(nx_graph)
        
        # Rank sentences by score
        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(self.sentences)), reverse=True)
        
        # Select top N sentences and restore original order
        top_sentences = sorted(ranked_sentences[:num_sentences], key=lambda x: self.sentences.index(x[1]))
        
        return "\n".join([s[1] for s in top_sentences])

    def clustering_based_summary(self, num_sentences):
        """Summarization using KMeans clustering."""
        if not self.sentences: return ""
        
        # Use TF-IDF to vectorize sentences
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(self.sentences)
        
        # Determine the number of clusters (must be <= total sentences)
        n_clusters = min(num_sentences, len(self.sentences))
        if n_clusters < 1: return ""

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
        kmeans.fit(tfidf_matrix)
        
        # Get the index of the closest sentence to each cluster centroid
        selected_indices = []
        for i in range(n_clusters):
            # Get distance from centroid to all sentences
            centroid = kmeans.cluster_centers_[i]
            distances = cosine_similarity([centroid], tfidf_matrix)
            
            # Find the index of the sentence closest to the centroid
            closest_sentence_index = np.argmax(distances)
            
            if closest_sentence_index not in selected_indices:
                selected_indices.append(closest_sentence_index)
                
        # Reconstruct the summary
        summary = [self.sentences[i] for i in sorted(selected_indices)]
        return "\n".join(summary)

    # ----------------------- 5. Gemini & Evaluation Methods -----------------------

    def get_gemini_summary(self):
        """Generates an abstractive summary using the Gemini API."""
        api_key_value = self.api_key.get()
        if not api_key_value:
            return "ERROR: Gemini API Key not provided."
        
        if not self.full_text:
            return "ERROR: No document text loaded."

        # Truncate text if too long (Gemini context window limits)
        MAX_CHARS = 20000 
        text_to_summarize = self.full_text[:MAX_CHARS]

        try:
            client = genai.Client(api_key=api_key_value)
            
            # Use a high-quality model for abstractive summary
            prompt = (
                "You are an expert summarizer. Generate a concise, objective, and abstractive summary "
                f"of the following text. The summary should be approximately {self.summary_size_percent.get():.0f}% "
                "of the original length and must use entirely new sentences where possible. "
                "Text to summarize:\n\n"
                f"{text_to_summarize}"
            )
            
            response = client.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text
        
        except Exception as e:
            return f"API ERROR: Failed to call Gemini API. Check your key and network connection. Details: {e}"

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
        """Execute all summarization methods."""
        if not self.full_text:
            messagebox.showwarning("Warning", "Please select and load a document first.")
            return

        num_sentences = self.calculate_summary_size()
        self.clear_summaries()

        # Update GUI status
        self.master.config(cursor="wait")
        self.master.update()

        try:
            # Run the 5 Extractive Summarizers via imported pure functions
            self.summaries['Frequency'] = _frequency_based_summary(self.sentences, num_sentences, self.stop_words)
            self.summaries['Position'] = _position_based_summary(self.sentences, num_sentences)
            self.summaries['TF-IDF'] = _tfidf_based_summary(self.sentences, num_sentences)
            self.summaries['TextRank'] = _textrank_summary(self.sentences, num_sentences)
            self.summaries['Clustering'] = _clustering_based_summary(self.sentences, num_sentences)

            # Display the 5 results in the main window
            for name, summary in self.summaries.items():
                widget = self.text_widgets[name]
                widget.config(state=tk.NORMAL)
                widget.insert(tk.END, summary)
                widget.config(state=tk.DISABLED)

            # Open the comparison window
            self.open_comparison_window()
            
        except Exception as e:
            messagebox.showerror("Execution Error", f"An error occurred during summarization: {e}")
            
        finally:
            self.master.config(cursor="")

    def open_comparison_window(self):
        """Creates the second window for Gemini and comparison."""
        
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
        
        # Run Gemini API call (can take a few seconds)
        gemini_summary = self.get_gemini_summary()
        gemini_summary_text.insert(tk.END, gemini_summary)
        
        # --- Comparison Section ---
        tk.Label(comparison_win, text="ROUGE-L F1 Score Comparison (vs. Gemini)", 
                 font=('Arial', 14, 'bold'), bg='#e0f7fa', fg='#00796b').pack(pady=20)

        comparison_frame = tk.Frame(comparison_win, bg='#e0f7fa')
        comparison_frame.pack(pady=10)

        headers = ["Technique", "ROUGE-L F1 (%)"]
        results = {}
        best_technique = ""
        max_score = -1.0
        
        if not gemini_summary.startswith("ERROR:"):
            # Calculate scores only if Gemini summary was successful
            for name, summary in self.summaries.items():
                score = self.calculate_rouge_l(gemini_summary, summary)
                results[name] = score
                if score > max_score:
                    max_score = score
                    best_technique = name
        else:
            # If API failed, report the error in the display area
            results = {name: 0.0 for name in self.summaries}

        # Table Headers
        for i, header in enumerate(headers):
            tk.Label(comparison_frame, text=header, font=('Arial', 12, 'bold'), 
                     bg='#00796b', fg='white', width=20, bd=1, relief=tk.RAISED).grid(row=0, column=i, padx=5, pady=5)

        # Table Rows
        row_num = 1
        for name, score in results.items():
            color = '#b2dfdb' if name == best_technique and max_score > 0 else 'white'
            tk.Label(comparison_frame, text=name, font=('Arial', 11), 
                     bg=color, fg='black', width=20, bd=1, relief=tk.GROOVE).grid(row=row_num, column=0, padx=5, pady=2)
            tk.Label(comparison_frame, text=f"{score:.2f}", font=('Arial', 11), 
                     bg=color, fg='black', width=20, bd=1, relief=tk.GROOVE).grid(row=row_num, column=1, padx=5, pady=2)
            row_num += 1

        # --- Best Result Display ---
        if best_technique and max_score > 0:
            final_message = f"🏆 Best Summarizer: {best_technique} with an Accuracy (ROUGE-L F1) of {max_score:.2f}%"
        else:
            final_message = "❌ Comparison failed due to an error (see Gemini Summary box)."
            
        tk.Label(comparison_win, text=final_message, 
                 font=('Arial', 15, 'bold'), bg='#e0f7fa', fg='red' if max_score == 0 else '#004d40').pack(pady=20)

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
        scale = tk.Scale(control_frame, from_=5, to=50, orient=tk.HORIZONTAL, 
                         variable=self.summary_size_percent, length=150, bg='#e0e0e0', 
                         troughcolor='#bdbdbd', highlightthickness=0)
        scale.pack(side=tk.LEFT, padx=5)

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