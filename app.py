"""
AI Resume Screening Streamlit Application - Enhanced Version
Uses multi-factor scoring beyond simple semantic similarity.
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
from scoring.scorer import EnhancedScorer
from utils.config import Config
from utils.logging_config import setup_logging
from utils.sanitizers import sanitize_filename
from utils.file_helpers import validate_file

# Setup logging
logger = setup_logging()

# Page configuration
st.set_page_config(
    page_title="AI Resume Screener - Enhanced",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    .score-breakdown {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #0066cc;
    }
    .high-score {
        color: #28a745;
        font-weight: bold;
    }
    .medium-score {
        color: #ffc107;
        font-weight: bold;
    }
    .low-score {
        color: #dc3545;
        font-weight: bold;
    }
    .missing-skill {
        background-color: #fff3cd;
        padding: 2px 6px;
        border-radius: 3px;
        margin: 2px;
        display: inline-block;
    }
    .matched-skill {
        background-color: #d4edda;
        padding: 2px 6px;
        border-radius: 3px;
        margin: 2px;
        display: inline-block;
    }
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
    *:focus {
        outline: 3px solid #0066cc !important;
        outline-offset: 2px !important;
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
</style>
""", unsafe_allow_html=True)


class EnhancedResumeScreenerApp:
    """Enhanced resume screening with multi-factor scoring."""
    
    def __init__(self):
        """Initialize application."""
        self.config = Config()
        self.extractor = PDFExtractor(self.config)
        self.embedder = Embedder(self.config)
        self.scorer = EnhancedScorer(self.config)
        
        if 'results' not in st.session_state:
            st.session_state.results = None
        if 'processing_complete' not in st.session_state:
            st.session_state.processing_complete = False
        if 'detailed_results' not in st.session_state:
            st.session_state.detailed_results = None
    
    def render_header(self) -> None:
        """Render application header."""
        
        st.markdown("""
        <div class="info-hint">
            <strong>Pro Tip:</strong> Be specific in your job description! Include exact years of experience needed,
            specific skills required, and education level. Example: "10 years operations experience, procurement expertise, 
            supply chain management, bachelor's degree required"
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
    
    def render_input_section(self) -> tuple:
        """Render input controls."""
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📋 Job Description")
            
            job_description = st.text_area(
                "Enter detailed job requirements:",
                height=300,
                placeholder="""Example:
                
We are seeking an Operations Manager with 10+ years of experience in logistics and supply chain management.

Required Skills:
- Procurement and vendor management
- Inventory optimization
- ERP systems (SAP/Oracle)
- Team leadership (15+ direct reports)
- Budget management

Education: Bachelor's degree in Business Administration or related field

Certifications: Six Sigma preferred""",
                help="Include specific requirements: years needed, skills required, education level"
            )
            
            if job_description:
                # Preview extracted requirements
                with st.expander("🔍 Preview Extracted Requirements"):
                    reqs = self.scorer.extract_requirements(job_description)
                    
                    st.write(f"**Years Required:** {reqs['required_years'] or 'Not specified'}")
                    st.write(f"**Education:** {reqs['education_level'] or 'Not specified'}")
                    
                    if reqs['required_skills']:
                        st.write("**Detected Skills:**")
                        skills_html = " ".join([f'<span class="matched-skill">{s}</span>' for s in reqs['required_skills']])
                        st.markdown(skills_html, unsafe_allow_html=True)
                    else:
                        st.write("*No specific skills detected. Consider adding technical or domain skills.*")
        
        with col2:
            st.subheader("📄 Resume Upload")
            uploaded_files = st.file_uploader(
                "Upload PDF Resumes",
                type=['pdf'],
                accept_multiple_files=True,
                help=f"Upload 1-{self.config.max_files} PDF resumes"
            )
            
            if uploaded_files:
                st.success(f"✅ {len(uploaded_files)} file(s) ready for analysis")
        
        st.markdown("---")
        
        # Scoring weight customization
        with st.expander("⚙️ Customize Scoring Weights (Advanced)"):
            st.markdown("Adjust how much each factor contributes to the final score:")
            
            col_w1, col_w2, col_w3 = st.columns(3)
            
            with col_w1:
                w_semantic = st.slider("Semantic Similarity", 0.0, 1.0, 0.30, 0.05)
                w_experience = st.slider("Experience Match", 0.0, 1.0, 0.25, 0.05)
            
            with col_w2:
                w_skills = st.slider("Skills Coverage", 0.0, 1.0, 0.25, 0.05)
                w_keywords = st.slider("Keyword Match", 0.0, 1.0, 0.15, 0.05)
            
            with col_w3:
                w_education = st.slider("Education Match", 0.0, 1.0, 0.05, 0.05)
            
            total_weight = w_semantic + w_experience + w_skills + w_keywords + w_education
            
            if abs(total_weight - 1.0) > 0.01:
                st.warning(f"⚠️ Weights sum to {total_weight:.2f}. They should sum to 1.0 for proper scoring.")
            else:
                st.success("✅ Weights properly configured")
                # Update scorer weights
                self.scorer.weights = {
                    'semantic_similarity': w_semantic,
                    'experience_match': w_experience,
                    'skills_coverage': w_skills,
                    'keyword_match': w_keywords,
                    'education_match': w_education
                }
        
        enable_ocr = st.checkbox(
            "Enable OCR Fallback",
            value=False,
            help="Use OCR for scanned PDFs (slower)"
        )
        
        return job_description, uploaded_files, enable_ocr
    
    def validate_inputs(self, job_description: Optional[str], uploaded_files: Optional[List]) -> tuple:
        """Validate user inputs."""
        if not job_description or len(job_description.strip()) < 20:
            return False, "Please provide a job description (minimum 20 characters)."
        
        if not uploaded_files:
            return False, "Please upload at least one resume PDF."
        
        if len(uploaded_files) > self.config.max_files:
            return False, f"Maximum {self.config.max_files} files allowed."
        
        for file in uploaded_files:
            is_valid, error = validate_file(file, self.config)
            if not is_valid:
                return False, f"{file.name}: {error}"
        
        return True, None
    
    def process_resumes(
        self,
        job_description: str,
        uploaded_files: List,
        enable_ocr: bool
    ) -> List[Dict[str, Any]]:
        """Process resumes with enhanced multi-factor scoring."""
        start_time = time.time()
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Extract job requirements once (cache for all resumes)
        status_text.text("Analyzing job requirements...")
        cached_requirements = self.scorer.extract_requirements(job_description)
        
        # Step 1: Extract text from all resumes
        status_text.text("Extracting text from resumes...")
        resume_data = []
        
        for idx, file in enumerate(uploaded_files):
            progress_bar.progress((idx + 1) / len(uploaded_files) / 3)
            
            try:
                file_bytes = BytesIO(file.read())
                safe_filename = sanitize_filename(file.name)
                
                extracted_text, method = self.extractor.extract_with_fallback(
                    file_bytes, safe_filename, enable_ocr=enable_ocr
                )
                
                if extracted_text and len(extracted_text) >= 50:
                    resume_data.append({
                        'filename': safe_filename,
                        'text': extracted_text,
                        'method': method,
                        'status': 'success',
                        'error': None
                    })
                else:
                    resume_data.append({
                        'filename': safe_filename,
                        'text': '',
                        'method': method,
                        'status': 'error',
                        'error': 'Insufficient text extracted'
                    })
            except Exception as e:
                logger.error(f"Error processing {file.name}: {str(e)}")
                resume_data.append({
                    'filename': sanitize_filename(file.name),
                    'text': '',
                    'method': 'none',
                    'status': 'error',
                    'error': str(e)
                })
        
        # Step 2: Compute embeddings
        status_text.text("Computing semantic embeddings...")
        
        try:
            valid_resumes = [r for r in resume_data if r['status'] == 'success']
            
            if not valid_resumes:
                st.error("No resumes were successfully extracted.")
                return []
            
            valid_texts = [r['text'] for r in valid_resumes]
            jd_embedding = self.embedder.encode_text(job_description)
            resume_embeddings = self.embedder.encode_batch(
                valid_texts, batch_size=self.config.embedding_batch_size
            )
            
            progress_bar.progress(0.66)
            
        except Exception as e:
            logger.error(f"Embedding error: {str(e)}")
            st.error(f"Embedding computation failed: {str(e)}")
            return []
        
        # Step 3: Enhanced multi-factor scoring
        status_text.text("Performing comprehensive candidate analysis...")
        
        try:
            for idx, resume in enumerate(valid_resumes):
                # Get comprehensive score breakdown
                score_result = self.scorer.compute_comprehensive_score(
                    job_description,
                    resume['text'],
                    jd_embedding,
                    resume_embeddings[idx],
                    cached_requirements=cached_requirements
                )
                
                # Extract top keywords for display
                top_keywords = self.scorer.extract_top_keywords(resume['text'], n_keywords=5)
                snippet = resume['text'][:200].replace('\n', ' ').strip() + "..."
                
                results.append({
                    'filename': resume['filename'],
                    'score': score_result['final_score'],
                    'component_scores': score_result['component_scores'],
                    'candidate_info': score_result['candidate_info'],
                    'matched_skills': score_result['matched_skills'],
                    'missing_skills': score_result['missing_skills'],
                    'matched_keywords': score_result['matched_keywords'],
                    'experience_gap': score_result['experience_gap'],
                    'recommendations': score_result['recommendations'],
                    'keywords': ', '.join(top_keywords),
                    'snippet': snippet,
                    'extraction_method': resume['method'],
                    'status': 'success',
                    'error': None,
                    'full_text': resume['text']
                })
            
            # Add failed resumes
            for resume in resume_data:
                if resume['status'] == 'error':
                    results.append({
                        'filename': resume['filename'],
                        'score': 0.0,
                        'component_scores': {},
                        'candidate_info': {},
                        'matched_skills': [],
                        'missing_skills': [],
                        'matched_keywords': [],
                        'experience_gap': None,
                        'recommendations': ['Extraction failed'],
                        'keywords': 'N/A',
                        'snippet': f'Error: {resume["error"]}',
                        'extraction_method': resume['method'],
                        'status': 'error',
                        'error': resume['error'],
                        'full_text': ''
                    })
            
            # Sort by final score
            results.sort(key=lambda x: x['score'], reverse=True)
            
            for idx, result in enumerate(results, 1):
                result['rank'] = idx
            
            progress_bar.progress(1.0)
            
        except Exception as e:
            logger.error(f"Scoring error: {str(e)}")
            st.error(f"Scoring failed: {str(e)}")
            return []
        
        processing_time = time.time() - start_time
        status_text.text(f"Complete! Analyzed {len(uploaded_files)} resumes in {processing_time:.1f}s")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return results
    
    def render_results(self, results: List[Dict[str, Any]]) -> None:
        """Render enhanced results with score breakdown."""
        if not results:
            return
        
        st.markdown("---")
        st.subheader("📊 Screening Results")
        
        # Summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'error']
        
        col1.metric("Total Resumes", len(results))
        col2.metric("Processed", len(successful))
        col3.metric("Failed", len(failed))
        if successful:
            col4.metric("Top Score", f"{successful[0]['score']*100:.1f}%")
            avg_score = sum(r['score'] for r in successful) / len(successful)
            col5.metric("Average Score", f"{avg_score*100:.1f}%")
        
        # Main results table
        st.markdown("### 🏆 Ranked Candidates")
        
        display_data = []
        for result in results:
            if result['status'] == 'success':
                # Color code score
                score_pct = result['score'] * 100
                if score_pct >= 70:
                    score_class = "🟢"
                elif score_pct >= 50:
                    score_class = "🟡"
                else:
                    score_class = "🔴"
                
                # Experience info
                years = result['candidate_info'].get('years_experience', 0)
                exp_gap = result.get('experience_gap')
                if exp_gap:
                    exp_info = f"{years}y (needs +{exp_gap}y)"
                else:
                    exp_info = f"{years}y ✓"
                
                # Skills coverage
                total_required = len(result.get('matched_skills', [])) + len(result.get('missing_skills', []))
                if total_required > 0:
                    skills_pct = len(result.get('matched_skills', [])) / total_required * 100
                    skills_info = f"{skills_pct:.0f}% ({len(result.get('matched_skills', []))}/{total_required})"
                else:
                    skills_info = "N/A"
                
                display_data.append({
                    'Rank': result['rank'],
                    'Candidate': result['filename'],
                    'Overall Score': f"{score_class} {score_pct:.1f}%",
                    'Experience': exp_info,
                    'Skills Match': skills_info,
                    'Semantic Fit': f"{result['component_scores'].get('semantic_similarity', 0)*100:.1f}%",
                    'Status': '✅ PASS'
                })
            else:
                display_data.append({
                    'Rank': result['rank'],
                    'Candidate': result['filename'],
                    'Overall Score': "❌ 0%",
                    'Experience': "N/A",
                    'Skills Match': "N/A",
                    'Semantic Fit': "N/A",
                    'Status': '❌ FAIL'
                })
        
        df_display = pd.DataFrame(display_data)
        st.dataframe(df_display, use_container_width=True, height=400)
        
        # Detailed breakdown for each candidate
        st.markdown("### 📋 Detailed Candidate Analysis")
        
        for result in successful[:10]:  # Show top 10
            with st.expander(f"#{result['rank']} - {result['filename']} ({result['score']*100:.1f}%)"):
                col_d1, col_d2 = st.columns(2)
                
                with col_d1:
                    st.markdown("**Score Breakdown:**")
                    for component, score in result['component_scores'].items():
                        bar_html = f"""
                        <div style="margin: 5px 0;">
                            <span>{component.replace('_', ' ').title()}: {score*100:.1f}%</span>
                            <div style="background: #e0e0e0; border-radius: 10px; height: 20px;">
                                <div style="background: {'#28a745' if score > 0.7 else '#ffc107' if score > 0.5 else '#dc3545'}; 
                                            width: {score*100}%; height: 100%; border-radius: 10px;"></div>
                            </div>
                        </div>
                        """
                        st.markdown(bar_html, unsafe_allow_html=True)
                    
                    st.markdown("**Candidate Profile:**")
                    st.write(f"- Years Experience: {result['candidate_info'].get('years_experience', 0)}")
                    st.write(f"- Education: {result['candidate_info'].get('education', 'Not detected')}")
                    if result['candidate_info'].get('job_titles'):
                        st.write(f"- Recent Roles: {', '.join(result['candidate_info']['job_titles'][:3])}")
                
                with col_d2:
                    st.markdown("**Skills Analysis:**")
                    
                    if result['matched_skills']:
                        st.write("✅ **Matched Skills:**")
                        skills_html = " ".join([f'<span class="matched-skill">{s}</span>' for s in result['matched_skills']])
                        st.markdown(skills_html, unsafe_allow_html=True)
                    
                    if result['missing_skills']:
                        st.write("⚠️ **Missing Required Skills:**")
                        skills_html = " ".join([f'<span class="missing-skill">{s}</span>' for s in result['missing_skills']])
                        st.markdown(skills_html, unsafe_allow_html=True)
                    
                    if result['recommendations']:
                        st.markdown("**💡 Recommendations:**")
                        for rec in result['recommendations']:
                            st.write(f"- {rec}")
        
        # Download options
        st.markdown("### 💾 Export Results")
        
        col_dl1, col_dl2 = st.columns(2)
        
        with col_dl1:
            csv_data = self.prepare_csv_export(results)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"enhanced_screening_results_{int(time.time())}.csv",
                mime="text/csv"
            )
        
        with col_dl2:
            json_data = self.prepare_json_export(results)
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=f"enhanced_screening_results_{int(time.time())}.json",
                mime="application/json"
            )
    
    def prepare_csv_export(self, results: List[Dict[str, Any]]) -> str:
        """Prepare CSV export."""
        export_rows = []
        for r in results:
            row = {
                'Rank': r['rank'],
                'File Name': r['filename'],
                'Overall Score': r['score'],
                'Experience Years': r.get('candidate_info', {}).get('years_experience', 0),
                'Experience Gap': r.get('experience_gap') or 0,
                'Skills Matched': len(r.get('matched_skills', [])),
                'Skills Missing': len(r.get('missing_skills', [])),
                'Semantic Score': r.get('component_scores', {}).get('semantic_similarity', 0),
                'Experience Score': r.get('component_scores', {}).get('experience_match', 0),
                'Skills Score': r.get('component_scores', {}).get('skills_coverage', 0),
                'Education': r.get('candidate_info', {}).get('education', ''),
                'Matched Skills': ', '.join(r.get('matched_skills', [])),
                'Missing Skills': ', '.join(r.get('missing_skills', [])),
                'Recommendations': ' | '.join(r.get('recommendations', [])),
                'Status': r['status']
            }
            export_rows.append(row)
        
        return pd.DataFrame(export_rows).to_csv(index=False)
    
    def prepare_json_export(self, results: List[Dict[str, Any]]) -> str:
        """Prepare JSON export."""
        export_data = {
            'timestamp': time.time(),
            'total_candidates': len(results),
            'successful': len([r for r in results if r['status'] == 'success']),
            'results': []
        }
        
        for r in results:
            export_data['results'].append({
                'rank': r['rank'],
                'filename': r['filename'],
                'overall_score': r['score'],
                'component_scores': r.get('component_scores', {}),
                'candidate_info': r.get('candidate_info', {}),
                'matched_skills': r.get('matched_skills', []),
                'missing_skills': r.get('missing_skills', []),
                'experience_gap': r.get('experience_gap'),
                'recommendations': r.get('recommendations', []),
                'status': r['status']
            })
        
        return json.dumps(export_data, indent=2)
    
    def run(self) -> None:
        """Main application loop."""
        self.render_header()
        
        with st.sidebar:
            st.header("ℹ️ About Enhanced Scoring")
            st.markdown("""
            **Why this is better than simple text matching:**
            
            🎯 **Multi-Factor Analysis**
            - Not just "does this text sound similar"
            - Actually checks if requirements are met
            
            📊 **Transparent Scoring**
            - See exactly WHY a candidate scored high/low
            - Identify skill gaps and experience mismatches
            
            ⚙️ **Customizable**
            - Adjust weights for what matters most
            - Focus on experience vs skills vs semantics
            
            ---
            
            **Scoring Components:**
            - **Semantic (30%)**: Context understanding
            - **Experience (25%)**: Years match
            - **Skills (25%)**: Required skills coverage
            - **Keywords (15%)**: Must-have terms
            - **Education (5%)**: Degree requirements
            """)
        
        job_description, uploaded_files, enable_ocr = self.render_input_section()
        
        if st.button("🚀 Run Enhanced Analysis", type="primary", use_container_width=True):
            is_valid, error_message = self.validate_inputs(job_description, uploaded_files)
            
            if not is_valid:
                st.error(error_message)
                return
            
            with st.spinner("Performing intelligent candidate analysis..."):
                try:
                    results = self.process_resumes(job_description, uploaded_files, enable_ocr)
                    
                    if results:
                        st.session_state.results = results
                        st.session_state.processing_complete = True
                        st.success("✅ Analysis complete!")
                    else:
                        st.error("Processing failed.")
                        
                except Exception as e:
                    logger.error(f"Application error: {str(e)}", exc_info=True)
                    st.error(f"Error: {str(e)}")
        
        if st.session_state.processing_complete and st.session_state.results:
            self.render_results(st.session_state.results)


def main():
    """Entry point."""
    try:
        app = EnhancedResumeScreenerApp()
        app.run()
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}", exc_info=True)
        st.error("Critical error. Please refresh the page.")


if __name__ == "__main__":
    main()

