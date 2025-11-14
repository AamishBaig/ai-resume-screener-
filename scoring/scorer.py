"""
Enhanced scoring module with requirement extraction and fit evaluation.
Goes beyond semantic similarity to evaluate actual candidate qualifications.
"""
import logging
import re
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available")


class EnhancedScorer:
    """
    Multi-factor scoring that evaluates:
    1. Semantic similarity (existing)
    2. Experience years match
    3. Required skills coverage
    4. Education requirements
    5. Keyword presence
    """
    
    def __init__(self, config):
        self.config = config
        self.must_have_keywords = config.must_have_keywords
        self.keyword_boost_value = config.keyword_boost_value
        
        # Scoring weights (should sum to 1.0)
        self.weights = {
            'semantic_similarity': 0.15,  # Reduced - less reliable
            'experience_match': 0.30,     # Increased - most important
            'skills_coverage': 0.30,      # Increased - very important
            'keyword_match': 0.15,        # Same
            'education_match': 0.10       # Increased slightly
        }
    
    def extract_requirements(self, job_description: str) -> Dict[str, Any]:
        """
        Extract specific requirements from job description.
        
        Returns dict with:
        - required_years: int or None
        - required_skills: list of skills
        - education_level: str or None
        - nice_to_have_skills: list
        """
        requirements = {
            'required_years': None,
            'min_years': 0,
            'max_years': 100,
            'required_skills': [],
            'nice_to_have_skills': [],
            'education_level': None,
            'certifications': []
        }
        
        jd_lower = job_description.lower()
        
        # Extract years of experience with multiple patterns
        year_patterns = [
            r'(\d+)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:experience|exp)',
            r'(?:minimum|min|at least)\s+(\d+)\s*(?:years?|yrs?)',
            r'(\d+)\s*-\s*(\d+)\s*(?:years?|yrs?)',
            r'(?:experience|exp)(?::\s*|\s+)(\d+)\+?\s*(?:years?|yrs?)',
            r'(\d+)\+?\s*(?:years?|yrs?)\s+(?:experience|exp)',
        ]
        
        for pattern in year_patterns:
            match = re.search(pattern, jd_lower)
            if match:
                groups = match.groups()
                if len(groups) == 2 and groups[1]:
                    # Range like "5-10 years"
                    requirements['min_years'] = int(groups[0])
                    requirements['max_years'] = int(groups[1])
                    requirements['required_years'] = int(groups[0])
                else:
                    # Single value like "10+ years"
                    years = int(groups[0])
                    requirements['required_years'] = years
                    requirements['min_years'] = years
                break
        
        # Extract required skills
        skill_patterns = [
            r'(?:required|must have|essential|mandatory)[\s\S]{0,100}?(?:skills?|qualifications?|requirements?)[\s\S]{0,200}',
            r'(?:skills?|qualifications?|requirements?)(?:\s*required)?:[\s\S]{0,300}',
        ]
        
        # Common technical/professional skills to look for
        common_skills = [
            'python', 'javascript', 'java', 'sql', 'react', 'angular', 'vue',
            'node', 'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git',
            'machine learning', 'data analysis', 'excel', 'powerpoint', 'word',
            'project management', 'agile', 'scrum', 'communication', 'leadership',
            'procurement', 'inventory', 'logistics', 'operations', 'supply chain',
            'budgeting', 'forecasting', 'reporting', 'erp', 'sap', 'oracle',
            'crm', 'salesforce', 'tableau', 'power bi', 'analytics',
            'vendor management', 'negotiation', 'contract', 'compliance',
            'warehouse', 'distribution', 'shipping', 'receiving', 'quality control',
            'lean', 'six sigma', 'continuous improvement', 'kpi', 'metrics'
        ]
        
        for skill in common_skills:
            if re.search(r'\b' + re.escape(skill) + r'\b', jd_lower):
                requirements['required_skills'].append(skill)
        
        # Extract education requirements
        education_patterns = [
            (r'\b(?:bachelor|bs|ba|b\.s\.|b\.a\.)\b', 'bachelors'),
            (r'\b(?:master|ms|ma|mba|m\.s\.|m\.a\.)\b', 'masters'),
            (r'\b(?:phd|ph\.d\.|doctorate)\b', 'phd'),
            (r'\b(?:diploma|associate)\b', 'diploma'),
            (r'\b(?:degree)\b', 'degree')
        ]
        
        for pattern, level in education_patterns:
            if re.search(pattern, jd_lower):
                requirements['education_level'] = level
                break
        
        # Extract certifications
        cert_patterns = [
            r'(?:certified|certification|certificate)[\s\S]{0,50}?(\w+(?:\s+\w+)?)',
            r'(pmp|cpa|cfa|aws certified|azure certified|google certified)',
        ]
        
        for pattern in cert_patterns:
            matches = re.findall(pattern, jd_lower)
            requirements['certifications'].extend(matches)
        
        logger.info(f"Extracted requirements: {requirements}")
        return requirements
    
    def extract_resume_info(self, resume_text: str) -> Dict[str, Any]:
        """
        Extract key information from resume.
        
        Returns dict with:
        - total_years: estimated years of experience
        - skills: list of identified skills
        - education: highest education level
        - certifications: list
        """
        resume_info = {
            'total_years': 0,
            'skills': [],
            'education': None,
            'certifications': [],
            'job_titles': []
        }
        
        resume_lower = resume_text.lower()
        
        # Extract years of experience from resume
        # Method 1: Look for explicit statements
        explicit_years = re.findall(
            r'(\d+)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:experience|exp)',
            resume_lower
        )
        if explicit_years:
            resume_info['total_years'] = max(int(y) for y in explicit_years)
        
        # Method 2: Calculate from date ranges (e.g., "2020 - Present")
        if resume_info['total_years'] == 0:
            date_ranges = re.findall(
                r'(\d{4})\s*[-–]\s*(?:(\d{4})|present|current|now)',
                resume_lower
            )
            total_years = 0
            current_year = 2025  # Current year
            
            for start, end in date_ranges:
                start_year = int(start)
                if end and end.isdigit():
                    end_year = int(end)
                else:
                    end_year = current_year
                total_years += max(0, end_year - start_year)
            
            resume_info['total_years'] = total_years
        
        # Extract skills (same list as in requirements)
        common_skills = [
            'python', 'javascript', 'java', 'sql', 'react', 'angular', 'vue',
            'node', 'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git',
            'machine learning', 'data analysis', 'excel', 'powerpoint', 'word',
            'project management', 'agile', 'scrum', 'communication', 'leadership',
            'procurement', 'inventory', 'logistics', 'operations', 'supply chain',
            'budgeting', 'forecasting', 'reporting', 'erp', 'sap', 'oracle',
            'crm', 'salesforce', 'tableau', 'power bi', 'analytics',
            'vendor management', 'negotiation', 'contract', 'compliance',
            'warehouse', 'distribution', 'shipping', 'receiving', 'quality control',
            'lean', 'six sigma', 'continuous improvement', 'kpi', 'metrics'
        ]
        
        for skill in common_skills:
            if re.search(r'\b' + re.escape(skill) + r'\b', resume_lower):
                resume_info['skills'].append(skill)
        
        # Extract education
        education_patterns = [
            (r'\b(?:phd|ph\.d\.|doctorate)\b', 'phd', 4),
            (r'\b(?:master|ms|ma|mba|m\.s\.|m\.a\.)\b', 'masters', 3),
            (r'\b(?:bachelor|bs|ba|b\.s\.|b\.a\.|degree)\b', 'bachelors', 2),
            (r'\b(?:diploma|associate)\b', 'diploma', 1),
        ]
        
        highest_level = None
        highest_rank = 0
        for pattern, level, rank in education_patterns:
            if re.search(pattern, resume_lower) and rank > highest_rank:
                highest_level = level
                highest_rank = rank
        
        resume_info['education'] = highest_level
        
        # Extract job titles
        title_patterns = [
            r'(?:^|\n)\s*([A-Z][a-zA-Z\s]+(?:Manager|Director|Engineer|Developer|Analyst|Coordinator|Specialist|Lead|Senior|Junior|Assistant))',
        ]
        
        for pattern in title_patterns:
            titles = re.findall(pattern, resume_text)
            resume_info['job_titles'].extend(titles[:5])  # Top 5 titles
        
        logger.info(f"Extracted resume info: years={resume_info['total_years']}, skills={len(resume_info['skills'])}")
        return resume_info
    
    def compute_semantic_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """Compute cosine similarity between embeddings."""
        if SKLEARN_AVAILABLE:
            sim = cosine_similarity(
                embedding1.reshape(1, -1),
                embedding2.reshape(1, -1)
            )[0][0]
        else:
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            if norm1 == 0 or norm2 == 0:
                sim = 0.0
            else:
                sim = dot_product / (norm1 * norm2)
        
        return max(0.0, min(1.0, float(sim)))
    
    def compute_experience_score(
        self,
        required_years: Optional[int],
        min_years: int,
        max_years: int,
        candidate_years: int
    ) -> float:
        """
        Score based on years of experience match.
        
        Returns:
        - 1.0 if meets or exceeds requirement
        - 0.0-1.0 based on how close they are
        - Small penalty for being overqualified (might not be interested)
        """
        if required_years is None:
            return 0.7  # No requirement specified, give decent score
        
        if candidate_years >= required_years:
            # Meets requirement
            if candidate_years <= max_years:
                return 1.0
            else:
                # Slightly overqualified (might leave for better position)
                overage = candidate_years - max_years
                return max(0.8, 1.0 - (overage * 0.02))
        else:
            # Below requirement - score proportionally
            if required_years == 0:
                return 1.0
            ratio = candidate_years / required_years
            return min(0.9, ratio)  # Cap at 0.9 if below requirement
    
    def compute_skills_coverage(
        self,
        required_skills: List[str],
        candidate_skills: List[str]
    ) -> Tuple[float, List[str], List[str]]:
        """
        Calculate what percentage of required skills the candidate has.
        
        Returns:
        - coverage_score (0-1)
        - matched_skills list
        - missing_skills list
        """
        if not required_skills:
            return 0.8, [], []  # No specific requirements
        
        matched = []
        missing = []
        
        for skill in required_skills:
            if skill in candidate_skills:
                matched.append(skill)
            else:
                missing.append(skill)
        
        coverage = len(matched) / len(required_skills)
        
        return coverage, matched, missing
    
    def compute_education_score(
        self,
        required_level: Optional[str],
        candidate_level: Optional[str]
    ) -> float:
        """Score based on education match."""
        if required_level is None:
            return 0.8  # No requirement
        
        if candidate_level is None:
            return 0.3  # Can't determine education
        
        education_ranks = {
            'diploma': 1,
            'degree': 2,
            'bachelors': 2,
            'masters': 3,
            'phd': 4
        }
        
        required_rank = education_ranks.get(required_level, 2)
        candidate_rank = education_ranks.get(candidate_level, 0)
        
        if candidate_rank >= required_rank:
            return 1.0
        else:
            return 0.5  # Below requirement
    
    def compute_keyword_boost(
        self,
        job_description: str,
        resume_text: str
    ) -> Tuple[float, List[str]]:
        """
        Compute keyword presence score.
        Only counts keywords that appear in BOTH the JD and resume.
        Score is based on how many JD keywords the resume matches.
        """
        matched_keywords = []
        jd_keywords_found = []  # Keywords that appear in the JD
        job_desc_lower = job_description.lower()
        resume_lower = resume_text.lower()
        
        # First, find which keywords from config appear in the JD
        for keyword_set in self.must_have_keywords:
            keywords = [kw.strip() for kw in keyword_set.split('|')]
            
            # Check if keyword appears in job description
            jd_has_keyword = any(
                re.search(r'\b' + re.escape(kw) + r'\b', job_desc_lower)
                for kw in keywords
            )
            
            if not jd_has_keyword:
                continue
            
            # This keyword is relevant to the JD
            jd_keywords_found.append(keyword_set)
            
            # Check if resume has this keyword
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', resume_lower):
                    matched_keywords.append(keyword)
                    break
        
        # Score based on JD keywords matched, not total config keywords
        if not jd_keywords_found:
            # No keywords from config found in JD - give neutral score
            score = 0.7
        else:
            # Percentage of JD keywords found in resume
            score = len(matched_keywords) / len(jd_keywords_found)
        
        return score, matched_keywords
    
    def compute_comprehensive_score(
        self,
        job_description: str,
        resume_text: str,
        jd_embedding: np.ndarray,
        resume_embedding: np.ndarray,
        cached_requirements: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Compute comprehensive multi-factor score.
        
        Returns detailed breakdown including:
        - final_score
        - component_scores
        - matched_skills
        - missing_skills
        - experience_gap
        - recommendations
        """
        # Extract requirements (use cached if available for batch processing)
        if cached_requirements:
            requirements = cached_requirements
        else:
            requirements = self.extract_requirements(job_description)
        
        # Extract candidate info
        resume_info = self.extract_resume_info(resume_text)
        
        # Compute individual scores
        semantic_score = self.compute_semantic_similarity(jd_embedding, resume_embedding)
        
        experience_score = self.compute_experience_score(
            requirements['required_years'],
            requirements['min_years'],
            requirements['max_years'],
            resume_info['total_years']
        )
        
        skills_score, matched_skills, missing_skills = self.compute_skills_coverage(
            requirements['required_skills'],
            resume_info['skills']
        )
        
        education_score = self.compute_education_score(
            requirements['education_level'],
            resume_info['education']
        )
        
        keyword_score, matched_keywords = self.compute_keyword_boost(
            job_description,
            resume_text
        )
        
        # Compute weighted final score
        final_score = (
            self.weights['semantic_similarity'] * semantic_score +
            self.weights['experience_match'] * experience_score +
            self.weights['skills_coverage'] * skills_score +
            self.weights['keyword_match'] * keyword_score +
            self.weights['education_match'] * education_score
        )
        
        # Generate experience gap analysis
        experience_gap = None
        if requirements['required_years'] and resume_info['total_years'] < requirements['required_years']:
            experience_gap = requirements['required_years'] - resume_info['total_years']
        
        # Generate recommendations
        recommendations = []
        if experience_gap and experience_gap > 2:
            recommendations.append(f"Candidate lacks {experience_gap} years of required experience")
        if missing_skills:
            recommendations.append(f"Missing skills: {', '.join(missing_skills[:5])}")
        if skills_score < 0.5:
            recommendations.append("Low skills coverage - may need significant training")
        if semantic_score > 0.7 and skills_score > 0.7:
            recommendations.append("Strong match - recommend interview")
        
        return {
            'final_score': final_score,
            'component_scores': {
                'semantic_similarity': semantic_score,
                'experience_match': experience_score,
                'skills_coverage': skills_score,
                'keyword_match': keyword_score,
                'education_match': education_score
            },
            'candidate_info': {
                'years_experience': resume_info['total_years'],
                'skills': resume_info['skills'],
                'education': resume_info['education'],
                'job_titles': resume_info['job_titles']
            },
            'requirements': {
                'required_years': requirements['required_years'],
                'required_skills': requirements['required_skills'],
                'education_level': requirements['education_level']
            },
            'matched_skills': matched_skills,
            'missing_skills': missing_skills,
            'matched_keywords': matched_keywords,
            'experience_gap': experience_gap,
            'recommendations': recommendations
        }
    
    def extract_top_keywords(self, text: str, n_keywords: int = 5) -> List[str]:
        """Extract top keywords using TF-IDF."""
        if not SKLEARN_AVAILABLE:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
            return [word for word, count in sorted_words[:n_keywords]]
        
        try:
            vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                max_df=0.95,
                min_df=1
            )
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            top_indices = tfidf_scores.argsort()[-n_keywords:][::-1]
            return [feature_names[i] for i in top_indices if tfidf_scores[i] > 0]
        except Exception as e:
            logger.warning(f"TF-IDF extraction failed: {str(e)}")
            return []
