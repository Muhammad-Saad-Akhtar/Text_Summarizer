"""
Core summarization algorithms with shared computation and improved quality.
This module provides optimized, reusable summarization functions.
"""

import re
import numpy as np
from typing import List, Set, Optional, Tuple, Dict, Any
from collections import Counter
from functools import lru_cache
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SummarizationCache:
    """Cache for shared computations across summarization methods."""
    
    def __init__(self):
        self._tfidf_matrix = None
        self._vectorizer = None
        self._sentences = None
        self._cleaned_text = None
        self._word_frequencies = None
        self._stop_words = None
    
    def clear(self):
        """Clear all cached data."""
        self._tfidf_matrix = None
        self._vectorizer = None
        self._sentences = None
        self._cleaned_text = None
        self._word_frequencies = None
        self._stop_words = None
    
    def get_tfidf_matrix(self, sentences: List[str], force_rebuild: bool = False):
        """Get or build TF-IDF matrix for sentences."""
        if force_rebuild or self._tfidf_matrix is None or self._sentences != sentences:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                self._vectorizer = TfidfVectorizer(
                    stop_words='english', 
                    lowercase=True, 
                    max_features=1000,
                    ngram_range=(1, 2)  # Add bigrams
                )
                self._tfidf_matrix = self._vectorizer.fit_transform(sentences)
                self._sentences = sentences.copy()
                logger.info(f"Built TF-IDF matrix for {len(sentences)} sentences")
            except ImportError:
                logger.warning("scikit-learn not available, using fallback")
                return None, None
        return self._tfidf_matrix, self._vectorizer
    
    def get_word_frequencies(self, text: str, stop_words: Set[str]):
        """Get or build word frequency counter."""
        if (self._word_frequencies is None or 
            self._cleaned_text != text or 
            self._stop_words != stop_words):
            self._cleaned_text = text
            self._stop_words = stop_words
            cleaned = self._clean_text(text)
            words = self._tokenize_words(cleaned)
            self._word_frequencies = Counter(w for w in words if w not in stop_words)
        return self._word_frequencies
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean text for processing."""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text
    
    @staticmethod
    def _tokenize_words(text: str) -> List[str]:
        """Tokenize text into words."""
        try:
            from nltk.tokenize import word_tokenize
            return word_tokenize(text)
        except ImportError:
            return text.split()

# Global cache instance
_cache = SummarizationCache()

def get_stop_words() -> Set[str]:
    """Get English stop words with fallback."""
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words('english'))
    except (ImportError, LookupError):
        # Fallback stop words
        return {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 
                'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 
                'to', 'was', 'will', 'with'}

def preprocess_sentences(sentences: List[str]) -> List[str]:
    """Clean and filter sentences."""
    if not sentences:
        return []
    
    # Remove very short sentences and clean whitespace
    cleaned = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) > 15:  # Minimum sentence length
            cleaned.append(sent)
    
    return cleaned

def calculate_mmr_scores(sentences: List[str], scores: Dict[int, float], 
                        lambda_param: float = 0.7, max_sentences: int = 5) -> List[int]:
    """Apply Maximal Marginal Relevance to reduce redundancy."""
    if not sentences or not scores:
        return []
    
    # Convert to list of (score, index) tuples
    scored_sentences = [(scores[i], i) for i in range(len(sentences)) if i in scores]
    scored_sentences.sort(reverse=True)
    
    selected_indices = []
    remaining_indices = list(range(len(sentences)))
    
    # Select first sentence (highest score)
    if scored_sentences:
        best_score, best_idx = scored_sentences[0]
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
    
    # Select remaining sentences using MMR
    while len(selected_indices) < min(max_sentences, len(sentences)) and remaining_indices:
        best_mmr_score = -1
        best_mmr_idx = -1
        
        for idx in remaining_indices:
            if idx not in scores:
                continue
                
            # Relevance score
            relevance = scores[idx]
            
            # Redundancy score (max similarity to already selected)
            max_similarity = 0
            if selected_indices:
                try:
                    from sklearn.metrics.pairwise import cosine_similarity
                    tfidf_matrix, _ = _cache.get_tfidf_matrix(sentences)
                    if tfidf_matrix is not None:
                        similarities = cosine_similarity(
                            tfidf_matrix[idx:idx+1], 
                            tfidf_matrix[selected_indices]
                        )
                        max_similarity = float(np.max(similarities))
                except Exception:
                    # Fallback: simple word overlap
                    selected_words = set()
                    for sel_idx in selected_indices:
                        selected_words.update(sentences[sel_idx].lower().split())
                    current_words = set(sentences[idx].lower().split())
                    if selected_words:
                        max_similarity = len(selected_words & current_words) / len(selected_words | current_words)
            
            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
            
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_mmr_idx = idx
        
        if best_mmr_idx != -1:
            selected_indices.append(best_mmr_idx)
            remaining_indices.remove(best_mmr_idx)
        else:
            break
    
    return sorted(selected_indices)

def frequency_based_summary(sentences: List[str], num_sentences: int, 
                          stop_words: Optional[Set[str]] = None) -> str:
    """Enhanced frequency-based summarization with MMR."""
    if not sentences or num_sentences <= 0:
        return ""
    
    sentences = preprocess_sentences(sentences)
    if len(sentences) <= num_sentences:
        return "\n".join(sentences)
    
    if stop_words is None:
        stop_words = get_stop_words()
    
    # Get word frequencies
    full_text = " ".join(sentences)
    word_frequencies = _cache.get_word_frequencies(full_text, stop_words)
    
    # Score sentences
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        cleaned = _cache._clean_text(sentence)
        words = _cache._tokenize_words(cleaned)
        # Normalize by sentence length
        score = sum(word_frequencies.get(w, 0) for w in words)
        sentence_scores[i] = score / max(len(words), 1)
    
    # Apply MMR for diversity
    selected_indices = calculate_mmr_scores(sentences, sentence_scores, 
                                          max_sentences=num_sentences)
    
    summary = [sentences[i] for i in selected_indices]
    return "\n".join(summary)

def position_based_summary(sentences: List[str], num_sentences: int) -> str:
    """Enhanced position-based summarization with content weighting."""
    if not sentences or num_sentences <= 0:
        return ""
    
    sentences = preprocess_sentences(sentences)
    if len(sentences) <= num_sentences:
        return "\n".join(sentences)
    
    stop_words = get_stop_words()
    word_frequencies = _cache.get_word_frequencies(" ".join(sentences), stop_words)
    
    # Score sentences with position and content
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        # Position score (higher for beginning and end)
        if i < len(sentences) // 3:
            position_score = 3.0
        elif i > 2 * len(sentences) // 3:
            position_score = 2.0
        else:
            position_score = 1.0
        
        # Content score
        cleaned = _cache._clean_text(sentence)
        words = _cache._tokenize_words(cleaned)
        content_score = sum(word_frequencies.get(w, 0) for w in words)
        content_score = content_score / max(len(words), 1)
        
        # Length bonus (prefer moderate length)
        length_score = min(len(words) / 20, 1.0)
        
        sentence_scores[i] = position_score * content_score * (1 + length_score)
    
    # Apply MMR
    selected_indices = calculate_mmr_scores(sentences, sentence_scores, 
                                          max_sentences=num_sentences)
    
    summary = [sentences[i] for i in selected_indices]
    return "\n".join(summary)

def tfidf_based_summary(sentences: List[str], num_sentences: int) -> str:
    """Enhanced TF-IDF summarization with sublinear scaling."""
    if not sentences or num_sentences <= 0:
        return ""
    
    sentences = preprocess_sentences(sentences)
    if len(sentences) <= num_sentences:
        return "\n".join(sentences)
    
    try:
        tfidf_matrix, vectorizer = _cache.get_tfidf_matrix(sentences)
        if tfidf_matrix is None:
            return "\n".join(sentences[:num_sentences])
        
        # Use sublinear TF scaling (log + 1)
        tfidf_array = tfidf_matrix.toarray()
        tfidf_array = np.log(tfidf_array + 1)
        
        # Score sentences
        sentence_scores = {}
        for i in range(len(sentences)):
            score = np.sum(tfidf_array[i])
            # Normalize by sentence length
            sentence_length = len(sentences[i].split())
            sentence_scores[i] = score / max(sentence_length, 1)
        
        # Apply MMR
        selected_indices = calculate_mmr_scores(sentences, sentence_scores, 
                                              max_sentences=num_sentences)
        
        summary = [sentences[i] for i in selected_indices]
        return "\n".join(summary)
        
    except Exception as e:
        logger.error(f"TF-IDF summarization failed: {e}")
        return "\n".join(sentences[:num_sentences])

def textrank_summary(sentences: List[str], num_sentences: int) -> str:
    """Enhanced TextRank with sparse similarity and thresholding."""
    if not sentences or num_sentences <= 0:
        return ""
    
    sentences = preprocess_sentences(sentences)
    if len(sentences) <= num_sentences:
        return "\n".join(sentences)
    
    try:
        import networkx as nx
        from sklearn.metrics.pairwise import cosine_similarity
        
        tfidf_matrix, _ = _cache.get_tfidf_matrix(sentences)
        if tfidf_matrix is None:
            return "\n".join(sentences[:num_sentences])
        
        # Build similarity matrix with threshold
        similarity_matrix = cosine_similarity(tfidf_matrix)
        threshold = 0.1  # Only keep significant similarities
        similarity_matrix[similarity_matrix < threshold] = 0
        
        # Create graph
        graph = nx.from_numpy_array(similarity_matrix)
        
        # Apply PageRank with damping
        scores = nx.pagerank(graph, alpha=0.85, max_iter=100)
        
        # Apply MMR
        selected_indices = calculate_mmr_scores(sentences, scores, 
                                              max_sentences=num_sentences)
        
        summary = [sentences[i] for i in selected_indices]
        return "\n".join(summary)
        
    except Exception as e:
        logger.error(f"TextRank summarization failed: {e}")
        return "\n".join(sentences[:num_sentences])

def clustering_based_summary(sentences: List[str], num_sentences: int) -> str:
    """Enhanced clustering with medoid selection and diversity."""
    if not sentences or num_sentences <= 0:
        return ""
    
    sentences = preprocess_sentences(sentences)
    if len(sentences) <= num_sentences:
        return "\n".join(sentences)
    
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics.pairwise import cosine_similarity
        
        tfidf_matrix, _ = _cache.get_tfidf_matrix(sentences)
        if tfidf_matrix is None:
            return "\n".join(sentences[:num_sentences])
        
        n_clusters = min(num_sentences, len(sentences))
        if n_clusters < 1:
            return ""
        
        # KMeans clustering
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        except TypeError:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        kmeans.fit(tfidf_matrix)
        
        # Select medoids (closest to centroids)
        selected_indices = []
        for i in range(n_clusters):
            cluster_mask = kmeans.labels_ == i
            if not np.any(cluster_mask):
                continue
                
            cluster_indices = np.where(cluster_mask)[0]
            cluster_center = kmeans.cluster_centers_[i]
            
            # Find closest sentence to centroid
            similarities = cosine_similarity(
                tfidf_matrix[cluster_indices], 
                [cluster_center]
            ).flatten()
            
            best_idx = cluster_indices[np.argmax(similarities)]
            if best_idx not in selected_indices:
                selected_indices.append(int(best_idx))
        
        # Sort by original order
        selected_indices.sort()
        summary = [sentences[i] for i in selected_indices]
        return "\n".join(summary)
        
    except Exception as e:
        logger.error(f"Clustering summarization failed: {e}")
        return "\n".join(sentences[:num_sentences])

def clear_cache():
    """Clear the global summarization cache."""
    _cache.clear()
