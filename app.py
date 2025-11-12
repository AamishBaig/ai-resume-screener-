"""
AI Resume Screening Streamlit Application
Main entry point for the HR resume screening tool.
"""
import streamlit as st
import pandas as pd
import json
import time
from typing import List, Dict, Any, Optional
from io import BytesIO
import logging

from extractors.pdf_extractor import PDFExtractor
from embeddings.embedder import Embedder
from scoring.scorer import Scorer
from utils.config import Config
from utils.logging_config import setup_logging
from utils.sanitizers import sanitize_filename
from utils.file_helpers import validate_file

# Setup logging
logger = setup_logging()

# Page configuration
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for accessibility and styling
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .top-3-row {
        background-color: #d4edda !important;
        font-weight: bold;
    }
    .error-row {
        background-color: #f8d7da !important;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    /* High contrast for accessibility */
    .stButton>button {
        font-weight: 600;
        border: 2px solid #0066cc;
    }
    /* Keyboard focus visible */
    *:focus {
        outline: 3px solid #0066cc !important;
        outline-offset: 2px !important;
    }
    /* Info hint styling - DARK BACKGROUND with WHITE TEXT */
    .info-hint {
        background-color: #0066cc;
        color: #ffffff;
        border-left: 4px solid #004085;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
        font-size: 0.9em;
    }
    .info-hint strong {
        color: #ffffff;
        font-weight: 700;
    }
    
/* DATAFRAME - Remove blue background, make text black */
.stDataFrame {
    background-color: #ffffff !important;
}

.stDataFrame td {
    background-color: #ffffff !important;
    color: #000000 !important;
}

.stDataFrame th {
    background-color: #f0f0f0 !important;
    color: #000000 !important;
    font-weight: bold !important;
}

/* Override Streamlit's default dataframe styling */
[data-testid="stDataFrame"] {
    background-color: #ffffff !important;
}

[data-testid="stDataFrame"] td {
    color: #000000 !important;
}

</style>
""", unsafe_allow_html=True)


class ResumeScreenerApp:
    """Main application class for resume screening."""
    
    def __init__(self):
        """Initialize the application with configuration and models."""
        self.config = Config()
        self.extractor = PDFExtractor(self.config)
        self.embedder = Embedder(self.config)
        self.scorer = Scorer(self.config)
        
        # Initialize session state
        if 'results' not in st.session_state:
            st.session_state.results = None
        if 'processing_complete' not in st.session_state:
            st.session_state.processing_complete = False
    
    def render_header(self) -> None:
        """Render application header and title."""
        st.title("AI Resume Screening System")
        st.markdown("""
        Upload job description and candidate resumes to get AI-powered ranking 
        based on semantic similarity and keyword matching.
        """)
        
        st.markdown("""
        <div class="info-hint">
            <strong>Pro Tip:</strong> For best results, provide a detailed job description with 
            50+ characters including requirements, skills, and responsibilities.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
    
    def render_input_section(self) -> tuple[Optional[str], Optional[List], bool, bool]:
        """
        Render input controls for job description and resume uploads.
        
        Returns:
            Tuple of (job_description, uploaded_files, hybrid_mode, enable_ocr)
        """
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Job Description")
            st.caption("Recommended: 50+ characters for optimal AI matching")
            
            job_desc_option = st.radio(
                "Input method:",
                ["Paste text", "Upload .txt file"],
                horizontal=True,
                label_visibility="collapsed"
            )
            
            job_description = None
            if job_desc_option == "Paste text":
                job_description = st.text_area(
                    "Job Description",
                    height=300,
                    placeholder="Paste the job description here...\n\nExample: We are seeking a Senior Python Developer with 5+ years experience...",
                    help="Provide detailed job requirements for better matching. Minimum 50 characters recommended."
                )
                
                if job_description:
                    char_count = len(job_description.strip())
                    if char_count < 50:
                        st.warning(f"Current: {char_count} characters. Add {50 - char_count} more for better results.")
                    elif char_count < 100:
                        st.info(f"Current: {char_count} characters. Good! More details = better matching.")
                    else:
                        st.success(f"Excellent! {char_count} characters - detailed description for best results.")
            else:
                jd_file = st.file_uploader(
                    "Upload Job Description",
                    type=['txt'],
                    help="Upload a .txt file containing the job description (50+ characters recommended)"
                )
                if jd_file:
                    job_description = jd_file.read().decode('utf-8')
                    char_count = len(job_description.strip())
                    st.success(f"Loaded: {jd_file.name} ({char_count} characters)")
                    
                    if char_count < 50:
                        st.warning(f"Short description ({char_count} chars). Consider adding more details for better results.")
        
        with col2:
            st.subheader("Resume Upload")
            uploaded_files = st.file_uploader(
                "Upload PDF Resumes",
                type=['pdf'],
                accept_multiple_files=True,
                help=f"Upload 1-{self.config.max_files} PDF resumes (max {self.config.max_file_size_mb}MB each)"
            )
            
            if uploaded_files:
                st.info(f"{len(uploaded_files)} file(s) uploaded")
        
        st.markdown("---")
        
        # Options
        col_opt1, col_opt2, col_opt3 = st.columns(3)
        
        with col_opt1:
            hybrid_mode = st.checkbox(
                "Enable Hybrid Mode",
                value=True,
                help="Boost scores for resumes matching must-have keywords"
            )
        
        with col_opt2:
            enable_ocr = st.checkbox(
                "Enable OCR Fallback",
                value=False,
                help="Use OCR for scanned PDFs (slower, requires Tesseract)"
            )
        
        with col_opt3:
            show_preview = st.checkbox(
                "Show Text Previews",
                value=False,
                help="Display extracted resume text in results"
            )
        
        return job_description, uploaded_files, hybrid_mode, enable_ocr
    
    def validate_inputs(
        self, 
        job_description: Optional[str], 
        uploaded_files: Optional[List]
    ) -> tuple[bool, Optional[str]]:
        """
        Validate user inputs.
        
        Args:
            job_description: The job description text
            uploaded_files: List of uploaded files
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not job_description or len(job_description.strip()) == 0:
            return False, "Please provide a job description."
        
        char_count = len(job_description.strip())
        if char_count < 50:
            st.warning(f"Your job description is short ({char_count} characters). For optimal AI matching, we recommend 50+ characters with detailed requirements, skills, and responsibilities.")
        
        if not uploaded_files or len(uploaded_files) == 0:
            return False, "Please upload at least one resume PDF."
        
        if len(uploaded_files) > self.config.max_files:
            return False, f"Maximum {self.config.max_files} files allowed. You uploaded {len(uploaded_files)}."
        
        for file in uploaded_files:
            is_valid, error = validate_file(file, self.config)
            if not is_valid:
                return False, f"{file.name}: {error}"
        
        return True, None
    
    def process_resumes(
        self,
        job_description: str,
        uploaded_files: List,
        hybrid_mode: bool,
        enable_ocr: bool
    ) -> List[Dict[str, Any]]:
        """
        Process all resumes and return ranked results.
        
        Args:
            job_description: The job description text
            uploaded_files: List of uploaded PDF files
            hybrid_mode: Whether to enable keyword boosting
            enable_ocr: Whether to enable OCR fallback
            
        Returns:
            List of result dictionaries
        """
        start_time = time.time()
        results = []
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Extracting text from resumes...")
        resume_data = []
        
        for idx, file in enumerate(uploaded_files):
            progress = (idx + 1) / len(uploaded_files) / 3  # First 33%
            progress_bar.progress(progress)
            
            file_status = st.empty()
            file_status.text(f"Extracting: {file.name}")
            
            try:
                file_bytes = BytesIO(file.read())
                safe_filename = sanitize_filename(file.name)
                
                extracted_text, method = self.extractor.extract_with_fallback(
                    file_bytes,
                    safe_filename,
                    enable_ocr=enable_ocr
                )
                
                if extracted_text and len(extracted_text) >= 50:
                    resume_data.append({
                        'filename': safe_filename,
                        'text': extracted_text,
                        'method': method,
                        'status': 'success',
                        'error': None
                    })
                    logger.info(f"Extracted {safe_filename} using {method}")
                else:
                    resume_data.append({
                        'filename': safe_filename,
                        'text': '',
                        'method': method,
                        'status': 'error',
                        'error': 'Insufficient text extracted (< 50 chars)'
                    })
                    logger.warning(f"Insufficient text from {safe_filename}")
                    
            except Exception as e:
                logger.error(f"Error processing {file.name}: {str(e)}")
                resume_data.append({
                    'filename': sanitize_filename(file.name),
                    'text': '',
                    'method': 'none',
                    'status': 'error',
                    'error': str(e)
                })
            
            file_status.empty()
        
        status_text.text("Computing semantic embeddings...")
        
        try:
            valid_resumes = [r for r in resume_data if r['status'] == 'success']
            
            if not valid_resumes:
                st.error("No resumes were successfully extracted. Please check file quality.")
                return []
            
            valid_texts = [r['text'] for r in valid_resumes]
            
            jd_embedding = self.embedder.encode_text(job_description)
            
            resume_embeddings = self.embedder.encode_batch(
                valid_texts,
                batch_size=self.config.embedding_batch_size
            )
            
            progress_bar.progress(0.66)
            
        except Exception as e:
            logger.error(f"Embedding error: {str(e)}")
            st.error(f"Error computing embeddings: {str(e)}")
            return []
        
        status_text.text("Scoring and ranking candidates...")
        
        try:
            for idx, resume in enumerate(valid_resumes):
                similarity = self.scorer.compute_similarity(
                    jd_embedding,
                    resume_embeddings[idx]
                )
                
                if hybrid_mode:
                    boost, matched_keywords = self.scorer.compute_keyword_boost(
                        job_description,
                        resume['text']
                    )
                    final_score = min(similarity + boost, 1.0)
                else:
                    final_score = similarity
                    matched_keywords = []
                
                top_keywords = self.scorer.extract_top_keywords(
                    resume['text'],
                    n_keywords=5
                )
                
                snippet = resume['text'][:200].replace('\n', ' ').strip() + "..."
                
                results.append({
                    'filename': resume['filename'],
                    'score': final_score,
                    'similarity': similarity,
                    'keywords': ', '.join(top_keywords),
                    'snippet': snippet,
                    'matched_must_haves': ', '.join(matched_keywords) if matched_keywords else 'None',
                    'extraction_method': resume['method'],
                    'status': 'success',
                    'error': None,
                    'full_text': resume['text']
                })
            
            for resume in resume_data:
                if resume['status'] == 'error':
                    results.append({
                        'filename': resume['filename'],
                        'score': 0.0,
                        'similarity': 0.0,
                        'keywords': 'N/A',
                        'snippet': 'Error: ' + resume['error'],
                        'matched_must_haves': 'N/A',
                        'extraction_method': resume['method'],
                        'status': 'error',
                        'error': resume['error'],
                        'full_text': ''
                    })
            
            results.sort(key=lambda x: x['score'], reverse=True)
            
            for idx, result in enumerate(results, 1):
                result['rank'] = idx
            
            progress_bar.progress(1.0)
            
        except Exception as e:
            logger.error(f"Scoring error: {str(e)}")
            st.error(f"Error during scoring: {str(e)}")
            return []
        
        processing_time = time.time() - start_time
        logger.info({
            'event': 'processing_complete',
            'total_files': len(uploaded_files),
            'successful': len(valid_resumes),
            'failed': len(resume_data) - len(valid_resumes),
            'processing_time_seconds': round(processing_time, 2)
        })
        
        status_text.text(f"Complete! Processed {len(uploaded_files)} resumes in {processing_time:.1f}s")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return results
    
    def render_results(self, results: List[Dict[str, Any]]) -> None:
        """
        Render the results table with styling and download options.
        
        Args:
            results: List of result dictionaries
        """
        if not results:
            return
        
        st.markdown("---")
        st.subheader("Screening Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'error']
        
        col1.metric("Total Resumes", len(results))
        col2.metric("Successfully Processed", len(successful))
        col3.metric("Failed", len(failed))
        if successful:
            col4.metric("Top Score", f"{successful[0]['score']*100:.2f}%")
        
        display_data = []
        for result in results:
            display_data.append({
                'Rank': result['rank'],
                'File Name': result['filename'],
                'Match Score': f"{result['score']*100:.2f}%",
                'Top Keywords': result['keywords'],
                'Snippet': result['snippet'][:100] + "..." if len(result['snippet']) > 100 else result['snippet'],
                'Status': 'PASS' if result['status'] == 'success' else 'FAIL'
            })
        
        df_display = pd.DataFrame(display_data)
        
        def highlight_top_3(row):
            if row['Rank'] <= 3 and row['Status'] == 'PASS':
                return ['background-color: #0e1117; color: #ffffff; font-weight: bold'] * len(row)
            elif row['Status'] == 'FAIL':
                return ['background-color: #f8d7da; color: #ffffff'] * len(row)
            return ['background-color: #ffffff; color: #000000'] * len(row)

        
        styled_df = df_display.style.apply(highlight_top_3, axis=1)
        
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        st.markdown("### Download Results")
        
        col_dl1, col_dl2 = st.columns(2)
        
        with col_dl1:
            csv_data = self.prepare_csv_export(results)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"resume_screening_results_{int(time.time())}.csv",
                mime="text/csv",
                help="Download results as CSV file"
            )
        
        with col_dl2:
            json_data = self.prepare_json_export(results)
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"resume_screening_results_{int(time.time())}.json",
                mime="application/json",
                help="Download results as JSON for integrations"
            )
        
        if st.checkbox("Show Full Text Previews"):
            st.markdown("---")
            for result in successful[:5]:  # Show top 5
                with st.expander(f"Preview: {result['filename']} (Score: {result['score']*100:.2f}%)"):
                    st.text_area(
                        "Extracted Text",
                        result['full_text'][:2000],
                        height=200,
                        key=f"preview_{result['filename']}"
                    )
    
    def prepare_csv_export(self, results: List[Dict[str, Any]]) -> str:
        """
        Prepare CSV export data.
        
        Args:
            results: List of result dictionaries
            
        Returns:
            CSV string
        """
        df = pd.DataFrame([{
            'Rank': r['rank'],
            'File Name': r['filename'],
            'Match Score': r['score'],
            'Similarity Score': r['similarity'],
            'Top Keywords': r['keywords'],
            'Matched Must-Haves': r['matched_must_haves'],
            'Extraction Method': r['extraction_method'],
            'Status': r['status'],
            'Error': r['error'] if r['error'] else '',
            'Snippet': r['snippet']
        } for r in results])
        
        return df.to_csv(index=False)
    
    def prepare_json_export(self, results: List[Dict[str, Any]]) -> str:
        """
        Prepare JSON export data.
        
        Args:
            results: List of result dictionaries
            
        Returns:
            JSON string
        """
        export_data = {
            'timestamp': time.time(),
            'total_candidates': len(results),
            'successful': len([r for r in results if r['status'] == 'success']),
            'failed': len([r for r in results if r['status'] == 'error']),
            'results': [{
                'rank': r['rank'],
                'filename': r['filename'],
                'match_score': r['score'],
                'similarity_score': r['similarity'],
                'top_keywords': r['keywords'].split(', ') if r['keywords'] != 'N/A' else [],
                'matched_must_haves': r['matched_must_haves'].split(', ') if r['matched_must_haves'] != 'None' else [],
                'extraction_method': r['extraction_method'],
                'status': r['status'],
                'error': r['error']
            } for r in results]
        }
        
        return json.dumps(export_data, indent=2)
    
    def run(self) -> None:
        """Main application run loop."""
        self.render_header()
        
        with st.sidebar:
            st.header("About")
            st.markdown("""
            **AI Resume Screener** uses semantic similarity 
            to match resumes against job descriptions.
            
            **Features:**
            - Semantic matching via embeddings
            - Keyword boost for must-haves
            - Multi-PDF batch processing
            - Fallback extraction (OCR support)
            
            **Privacy:** All processing happens in-memory. 
            No data is stored or sent to third parties.
            """)
            
            st.markdown("---")
            st.markdown("### Job Description Tips")
            st.markdown("""
            **For best results, include:**
            - Required skills & technologies
            - Years of experience needed
            - Education requirements
            - Specific certifications
            - Key responsibilities
            
            **Recommended:** 50+ characters
            """)
            
            st.markdown("---")
            st.markdown("**Version:** 1.0.0")
            st.markdown("**License:** MIT")
        
        job_description, uploaded_files, hybrid_mode, enable_ocr = self.render_input_section()
        
        if st.button("Run AI Screening", type="primary", use_container_width=True):
            is_valid, error_message = self.validate_inputs(job_description, uploaded_files)
            
            if not is_valid:
                st.error(error_message)
                return
            
            with st.spinner("Processing resumes..."):
                try:
                    results = self.process_resumes(
                        job_description,
                        uploaded_files,
                        hybrid_mode,
                        enable_ocr
                    )
                    
                    if results:
                        st.session_state.results = results
                        st.session_state.processing_complete = True
                        st.success("Processing complete!")
                    else:
                        st.error("Processing failed. Check logs for details.")
                        
                except Exception as e:
                    logger.error(f"Application error: {str(e)}", exc_info=True)
                    st.error(f"An unexpected error occurred: {str(e)}")
        
        if st.session_state.processing_complete and st.session_state.results:
            self.render_results(st.session_state.results)


def main():
    """Application entry point."""
    try:
        app = ResumeScreenerApp()
        app.run()
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}", exc_info=True)
        st.error("A critical error occurred. Please refresh the page and try again.")


if __name__ == "__main__":
    main()