"""
Scoring module for computing similarity and keyword matching.
"""
import logging
import re
from typing import Tuple, List
import numpy as np

logger = logging.getLogger(__name__)

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available, using numpy fallback")


class Scorer:
    """Compute similarity scores and keyword matching."""
    
    def __init__(self, config):
        """
        Initialize scorer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.must_have_keywords = config.must_have_keywords
        self.keyword_boost_value = config.keyword_boost_value
    
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score (0-1)
        """
        if SKLEARN_AVAILABLE:
            # Use sklearn implementation
            sim = cosine_similarity(
                embedding1.reshape(1, -1),
                embedding2.reshape(1, -1)
            )[0][0]
        else:
            # Numpy fallback
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                sim = 0.0
            else:
                sim = dot_product / (norm1 * norm2)
        
        # Ensure in [0, 1] range
        sim = max(0.0, min(1.0, float(sim)))
        
        return sim
    
    def compute_keyword_boost(
        self,
        job_description: str,
        resume_text: str
    ) -> Tuple[float, List[str]]:
        """
        Compute keyword boost based on must-have keywords.
        
        Args:
            job_description: Job description text
            resume_text: Resume text
            
        Returns:
            Tuple of (boost_value, matched_keywords)
        """
        matched_keywords = []
        
        job_desc_lower = job_description.lower()
        resume_lower = resume_text.lower()
        
        for keyword_set in self.must_have_keywords:
            # Split keyword set (e.g., "python|java" means any of these)
            keywords = [kw.strip() for kw in keyword_set.split('|')]
            
            # Check if keyword appears in job description first
            jd_has_keyword = any(
                re.search(r'\b' + re.escape(kw) + r'\b', job_desc_lower)
                for kw in keywords
            )
            
            if not jd_has_keyword:
                continue
            
            # Check if any keyword in set matches resume
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', resume_lower):
                    matched_keywords.append(keyword)
                    break
        
        boost = len(matched_keywords) * self.keyword_boost_value
        
        return boost, matched_keywords
    
    def extract_top_keywords(
        self,
        text: str,
        n_keywords: int = 5
    ) -> List[str]:
        """
        Extract top keywords from text using TF-IDF.
        
        Args:
            text: Input text
            n_keywords: Number of top keywords to extract
            
        Returns:
            List of top keywords
        """
        if not SKLEARN_AVAILABLE:
            # Simple fallback: most common words
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
            return [word for word, count in sorted_words[:n_keywords]]
        
        try:
            # Use TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                max_df=0.95,
                min_df=1
            )
            
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            
            # Get top features
            tfidf_scores = tfidf_matrix.toarray()[0]
            top_indices = tfidf_scores.argsort()[-n_keywords:][::-1]
            
            top_keywords = [feature_names[i] for i in top_indices if tfidf_scores[i] > 0]
            
            return top_keywords
            
        except Exception as e:
            logger.warning(f"TF-IDF extraction failed: {str(e)}")
            return []
