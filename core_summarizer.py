"""
Simple and effective summarization algorithms.
"""

import re
import math
from typing import List, Set, Optional
from collections import Counter

# Simple stop words list
STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its', 
    'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with', 'i', 'you', 'this', 'they', 'them', 'their',
    'we', 'our', 'your', 'his', 'her', 'him', 'she', 'me', 'my', 'mine', 'yours', 'theirs', 'ours'
}

def get_stop_words() -> Set[str]:
    """Get English stop words."""
    return STOP_WORDS

def preprocess_sentences(sentences: List[str]) -> List[str]:
    """Clean and filter sentences."""
    if not sentences:
        return []
    
    cleaned = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) > 10 and len(sent.split()) > 3:  # At least 3 words
            cleaned.append(sent)
    
    return cleaned if cleaned else sentences  # Return original if nothing passes filter

def select_best_sentences(scores, num_sentences):
    """Select top scoring sentences."""
    if not scores:
        return []
    
    # Sort by score and take top N
    sorted_sentences = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected_indices = [idx for idx, _ in sorted_sentences[:num_sentences]]
    return sorted(selected_indices)  # Keep original order

def frequency_based_summary(sentences: List[str], num_sentences: int, 
                          stop_words: Optional[Set[str]] = None) -> str:
    """Frequency-based summarization - selects sentences with most important words."""
    if not sentences or num_sentences <= 0:
        return ""
    
    processed_sentences = preprocess_sentences(sentences)
    if len(processed_sentences) <= num_sentences:
        return "\n".join(processed_sentences)
    
    if stop_words is None:
        stop_words = get_stop_words()
    
    # Count word frequencies across all sentences
    all_words = []
    for sentence in processed_sentences:
        words = [w.lower() for w in sentence.split() if w.lower() not in stop_words and len(w) > 2]
        all_words.extend(words)
    
    word_freq = Counter(all_words)
    
    # Score each sentence based on word frequencies
    sentence_scores = {}
    for i, sentence in enumerate(processed_sentences):
        words = [w.lower() for w in sentence.split() if w.lower() not in stop_words]
        if words:
            score = sum(word_freq.get(w, 0) for w in words) / len(words)
            sentence_scores[i] = score
        else:
            sentence_scores[i] = 0
    
    # Select top scoring sentences
    selected_indices = select_best_sentences(sentence_scores, num_sentences)
    summary = [processed_sentences[i] for i in selected_indices]
    return "\n".join(summary)

def position_based_summary(sentences: List[str], num_sentences: int) -> str:
    """Position-based summarization - prefers sentences at beginning and end."""
    if not sentences or num_sentences <= 0:
        return ""
    
    processed_sentences = preprocess_sentences(sentences)
    if len(processed_sentences) <= num_sentences:
        return "\n".join(processed_sentences)
    
    # Score sentences based on position
    sentence_scores = {}
    for i, sentence in enumerate(processed_sentences):
        # Higher score for beginning and end sentences
        if i < len(processed_sentences) // 3:
            position_score = 3.0
        elif i > 2 * len(processed_sentences) // 3:
            position_score = 2.0
        else:
            position_score = 1.0
        
        # Bonus for reasonable length
        word_count = len(sentence.split())
        length_bonus = min(word_count / 15, 1.5)
        
        sentence_scores[i] = position_score * length_bonus
    
    # Select top scoring sentences
    selected_indices = select_best_sentences(sentence_scores, num_sentences)
    summary = [processed_sentences[i] for i in selected_indices]
    return "\n".join(summary)

def tfidf_based_summary(sentences: List[str], num_sentences: int) -> str:
    """TF-IDF based summarization - finds sentences with unique important words."""
    if not sentences or num_sentences <= 0:
        return ""
    
    processed_sentences = preprocess_sentences(sentences)
    if len(processed_sentences) <= num_sentences:
        return "\n".join(processed_sentences)
    
    # Get words for each sentence
    sentence_words = []
    all_words = set()
    for sentence in processed_sentences:
        words = [w.lower() for w in sentence.split() if len(w) > 2 and w.lower() not in STOP_WORDS]
        sentence_words.append(words)
        all_words.update(words)
    
    # Calculate IDF for each word
    word_idf = {}
    for word in all_words:
        doc_count = sum(1 for words in sentence_words if word in words)
        word_idf[word] = math.log(len(processed_sentences) / max(doc_count, 1))
    
    # Score sentences using TF-IDF
    sentence_scores = {}
    for i, words in enumerate(sentence_words):
        if not words:
            sentence_scores[i] = 0
            continue
            
        word_counts = Counter(words)
        score = 0
        for word, count in word_counts.items():
            tf = count / len(words)  # Term frequency
            idf = word_idf.get(word, 0)  # Inverse document frequency
            score += tf * idf
        sentence_scores[i] = score
    
    # Select top scoring sentences
    selected_indices = select_best_sentences(sentence_scores, num_sentences)
    summary = [processed_sentences[i] for i in selected_indices]
    return "\n".join(summary)

def textrank_summary(sentences: List[str], num_sentences: int) -> str:
    """TextRank-like summarization - finds sentences most similar to others."""
    if not sentences or num_sentences <= 0:
        return ""
    
    processed_sentences = preprocess_sentences(sentences)
    if len(processed_sentences) <= num_sentences:
        return "\n".join(processed_sentences)
    
    # Get word sets for each sentence
    sentence_word_sets = []
    for sentence in processed_sentences:
        words = set(w.lower() for w in sentence.split() if len(w) > 2 and w.lower() not in STOP_WORDS)
        sentence_word_sets.append(words)
    
    # Calculate similarity scores
    sentence_scores = {}
    for i, words_i in enumerate(sentence_word_sets):
        score = 0
        for j, words_j in enumerate(sentence_word_sets):
            if i != j and words_i and words_j:
                # Jaccard similarity
                intersection = len(words_i & words_j)
                union = len(words_i | words_j)
                if union > 0:
                    score += intersection / union
        sentence_scores[i] = score
    
    # Select top scoring sentences
    selected_indices = select_best_sentences(sentence_scores, num_sentences)
    summary = [processed_sentences[i] for i in selected_indices]
    return "\n".join(summary)

def clustering_based_summary(sentences: List[str], num_sentences: int) -> str:
    """Clustering-like summarization - selects diverse sentences."""
    if not sentences or num_sentences <= 0:
        return ""
    
    processed_sentences = preprocess_sentences(sentences)
    if len(processed_sentences) <= num_sentences:
        return "\n".join(processed_sentences)
    
    # Get word sets for each sentence
    sentence_word_sets = []
    for sentence in processed_sentences:
        words = set(w.lower() for w in sentence.split() if len(w) > 2 and w.lower() not in STOP_WORDS)
        sentence_word_sets.append(words)
    
    selected_indices = []
    
    # Start with the sentence that has most unique words
    word_uniqueness = []
    for i, words in enumerate(sentence_word_sets):
        unique_score = len(words)
        word_uniqueness.append((unique_score, i))
    
    word_uniqueness.sort(reverse=True)
    if word_uniqueness:
        selected_indices.append(word_uniqueness[0][1])
    
    # Add most diverse sentences
    while len(selected_indices) < num_sentences and len(selected_indices) < len(processed_sentences):
        best_diversity = -1
        best_idx = -1
        
        for i, words_i in enumerate(sentence_word_sets):
            if i in selected_indices:
                continue
            
            # Calculate diversity score (how different from selected sentences)
            diversity = 0
            for sel_idx in selected_indices:
                words_sel = sentence_word_sets[sel_idx]
                if words_i and words_sel:
                    intersection = len(words_i & words_sel)
                    union = len(words_i | words_sel)
                    if union > 0:
                        diversity += 1 - (intersection / union)  # 1 - similarity = diversity
            
            avg_diversity = diversity / len(selected_indices) if selected_indices else 1
            if avg_diversity > best_diversity:
                best_diversity = avg_diversity
                best_idx = i
        
        if best_idx != -1:
            selected_indices.append(best_idx)
        else:
            # If no more diverse sentences, just add remaining in order
            for i in range(len(processed_sentences)):
                if i not in selected_indices:
                    selected_indices.append(i)
                    break
            break
    
    selected_indices.sort()  # Keep original order
    summary = [processed_sentences[i] for i in selected_indices]
    return "\n".join(summary)

def clear_cache():
    """Clear cache - simplified version."""
    pass  # No cache to clear in simplified version
