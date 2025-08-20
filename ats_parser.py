import os
import re
import json
from typing import Dict, List, Tuple, Set
import PyPDF2
import pandas as pd
from collections import Counter, defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.tree import Tree
import warnings
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime
import textstat
from textblob import TextBlob
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('chunkers/maxent_ne_chunker')
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')

class SmartATSMatcher:
    def __init__(self):
        # Try to load spacy model, fallback if not available
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.use_spacy = True
        except OSError:
            print("SpaCy model not found. Install with: python -m spacy download en_core_web_sm")
            print("Falling back to NLTK-only processing...")
            self.use_spacy = False
            
        self.stop_words = set(stopwords.words('english'))
        
        # Common patterns for better extraction
        self.degree_patterns = [
            r'\b(?:bachelor|master|phd|doctorate|mba|ms|ma|bs|ba|btech|mtech|be|me|bca|mca|diploma)(?:\s+(?:of|in|degree))?\s+[^.\n]{1,50}',
            r'\b(?:b\.?s\.?|m\.?s\.?|b\.?a\.?|m\.?a\.?|ph\.?d\.?|m\.?b\.?a\.?)\s+(?:in\s+)?[^.\n]{1,50}'
        ]
        
        self.experience_patterns = [
            r'(?:(\d+)(?:\+|\s*to\s*\d+)?)\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)',
            r'(?:experience|exp)(?:d)?\s*(?:of\s*)?(?:(\d+)(?:\+|\s*to\s*\d+)?)\s*(?:years?|yrs?)',
            r'(\d+)(?:\+|\s*to\s*\d+)?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:relevant\s*)?(?:work\s*)?(?:experience|exp)'
        ]
        
        self.skill_context_words = [
            'experience', 'skilled', 'proficient', 'expertise', 'knowledge',
            'familiar', 'working', 'hands-on', 'strong', 'solid', 'advanced',
            'intermediate', 'beginner', 'expert', 'competent', 'ability',
            'understanding', 'background', 'training', 'certification'
        ]
        
        # Modern tech trends and market-relevant skills (2024-2025)
        self.trending_skills = {
            'ai_ml': ['chatgpt', 'claude', 'llm', 'transformer', 'bert', 'gpt', 'langchain', 'llamaindex', 
                     'vector database', 'rag', 'fine-tuning', 'prompt engineering', 'huggingface', 'openai',
                     'anthropic', 'stable diffusion', 'midjourney', 'generative ai', 'neural networks'],
            'cloud_devops': ['kubernetes', 'docker', 'aws', 'azure', 'gcp', 'terraform', 'ansible', 
                           'jenkins', 'github actions', 'ci/cd', 'microservices', 'serverless', 'lambda'],
            'data_engineering': ['apache spark', 'kafka', 'airflow', 'dbt', 'snowflake', 'databricks', 
                               'bigquery', 'redshift', 'elasticsearch', 'mongodb', 'postgres'],
            'modern_frontend': ['react', 'vue', 'angular', 'nextjs', 'typescript', 'tailwind', 'svelte', 
                              'remix', 'nuxt', 'vite', 'webpack'],
            'backend_modern': ['fastapi', 'django', 'flask', 'nodejs', 'express', 'nestjs', 'graphql', 
                             'rest api', 'grpc', 'websockets'],
            'security': ['oauth', 'jwt', 'ssl/tls', 'cybersecurity', 'penetration testing', 'zero trust',
                        'encryption', 'vulnerability assessment'],
            'blockchain_web3': ['ethereum', 'smart contracts', 'solidity', 'web3', 'defi', 'nft', 'dao']
        }
        
        # Industry-specific high-value certifications
        self.valuable_certifications = [
            'aws certified', 'azure certified', 'google cloud', 'certified kubernetes', 'docker certified',
            'pmp', 'scrum master', 'product owner', 'agile', 'six sigma', 'itil', 'cissp', 'ceh',
            'tensorflow developer', 'data scientist certified', 'machine learning engineer',
            'certified data professional', 'tableau certified', 'power bi certified'
        ]
        
        # Leadership and impact keywords
        self.leadership_indicators = [
            'led team', 'managed', 'mentored', 'supervised', 'coordinated', 'drove initiative',
            'spearheaded', 'pioneered', 'established', 'founded', 'launched', 'scaled',
            'increased revenue', 'reduced costs', 'improved efficiency', 'optimized performance'
        ]
        
        # Quantifiable achievement patterns
        self.achievement_patterns = [
            r'(\d+(?:\.\d+)?)\s*%\s*(?:increase|improvement|reduction|growth)',
            r'(?:\$|USD)\s*(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:million|k|thousand)?',
            r'(\d+(?:,\d{3})*)\s*(?:users|customers|clients|downloads|views)',
            r'(\d+(?:\.\d+)?)\s*(?:x|times)\s*(?:faster|improvement|increase)'
        ]

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Enhanced PDF text extraction with better formatting"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    # Clean up common PDF extraction issues
                    page_text = re.sub(r'\s+', ' ', page_text)  # Multiple spaces to single
                    page_text = re.sub(r'\n\s*\n', '\n', page_text)  # Multiple newlines
                    text += page_text + "\n"
                return text
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {str(e)}")
            return ""

    def extract_skills_from_jd(self, jd_text: str) -> Dict[str, List[str]]:
        """Dynamically extract required skills from job description"""
        jd_lower = jd_text.lower()
        
        # Technical skills patterns
        tech_patterns = [
            r'(?:experience|skilled|proficient|knowledge)\s+(?:in|with|of)\s+([^.;\n]{5,100})',
            r'(?:must|should)\s+(?:have|know|understand)\s+([^.;\n]{5,100})',
            r'(?:required|preferred|desired)(?:\s+skills?)?[:\s]+([^.;\n]{5,150})',
            r'(?:technologies?|tools?|languages?|frameworks?|platforms?)[:\s]+([^.;\n]{5,150})',
            r'(?:strong|solid|good)\s+(?:background|experience|knowledge|understanding)\s+(?:in|with|of)\s+([^.;\n]{5,100})'
        ]
        
        extracted_skills = set()
        for pattern in tech_patterns:
            matches = re.finditer(pattern, jd_lower, re.IGNORECASE)
            for match in matches:
                skill_text = match.group(1).strip()
                # Clean and split skills
                skills = self._clean_and_split_skills(skill_text)
                extracted_skills.update(skills)
        
        # Education requirements
        education_reqs = self._extract_education_requirements(jd_lower)
        
        # Experience requirements
        experience_reqs = self._extract_experience_requirements(jd_lower)
        
        # Soft skills
        soft_skills = self._extract_soft_skills(jd_lower)
        
        # Certifications
        certifications = self._extract_certifications(jd_lower)
        
        return {
            'technical_skills': list(extracted_skills),
            'education_requirements': education_reqs,
            'experience_requirements': experience_reqs,
            'soft_skills': soft_skills,
            'certifications': certifications
        }

    def _clean_and_split_skills(self, skill_text: str) -> List[str]:
        """Clean and split skill text into individual skills"""
        # Remove common non-skill words
        skill_text = re.sub(r'\b(?:experience|knowledge|understanding|skills?|ability|proficiency)\b', '', skill_text)
        skill_text = re.sub(r'\b(?:strong|solid|good|excellent|advanced|intermediate|basic)\b', '', skill_text)
        skill_text = re.sub(r'\b(?:working|hands-on|practical|theoretical)\b', '', skill_text)
        
        # Split on common delimiters
        skills = re.split(r'[,;&/\n\t]+|(?:\s+and\s+)|(?:\s+or\s+)', skill_text)
        
        cleaned_skills = []
        for skill in skills:
            skill = skill.strip().strip('.,;()[]{}')
            # Filter out very short or very long skills, and common non-skill words
            if (3 <= len(skill) <= 50 and 
                not re.match(r'^\d+$', skill) and
                skill not in ['the', 'and', 'or', 'with', 'for', 'in', 'on', 'at', 'to', 'from']):
                cleaned_skills.append(skill)
        
        return cleaned_skills

    def _extract_education_requirements(self, text: str) -> List[str]:
        """Extract education requirements from job description"""
        education_reqs = []
        
        # Look for degree requirements
        for pattern in self.degree_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                education_reqs.append(match.group().strip())
        
        # Look for general education mentions
        edu_patterns = [
            r'(?:bachelor|master|phd|degree)\s+(?:in|of)\s+[^.\n]{5,50}',
            r'(?:education|qualification)[:\s]+[^.\n]{5,100}'
        ]
        
        for pattern in edu_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                education_reqs.append(match.group().strip())
        
        return list(set(education_reqs))

    def _extract_experience_requirements(self, text: str) -> Dict[str, any]:
        """Extract experience requirements"""
        experience_years = []
        experience_types = []
        
        # Extract years of experience
        for pattern in self.experience_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if match.group(1):
                    try:
                        years = int(match.group(1))
                        experience_years.append(years)
                    except ValueError:
                        continue
        
        # Extract types of experience
        exp_type_patterns = [
            r'(?:experience|background)\s+(?:in|with)\s+([^.\n]{5,80})',
            r'(?:proven|demonstrated)\s+(?:experience|track record)\s+(?:in|with)\s+([^.\n]{5,80})'
        ]
        
        for pattern in exp_type_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                exp_type = match.group(1).strip()
                if len(exp_type) > 5:
                    experience_types.append(exp_type)
        
        return {
            'years_required': max(experience_years) if experience_years else 0,
            'types': list(set(experience_types))
        }

    def _extract_soft_skills(self, text: str) -> List[str]:
        """Extract soft skills from job description"""
        soft_skill_patterns = [
            r'\b(?:excellent|strong|good)\s+(?:communication|leadership|analytical|problem[\s-]?solving|teamwork|interpersonal)\s+(?:skills?|abilities?)\b',
            r'\b(?:ability to|capable of|able to)\s+([^.\n]{10,80})',
            r'\b(?:self[\s-]?motivated|detail[\s-]?oriented|results[\s-]?driven|team[\s-]?player)\b'
        ]
        
        soft_skills = []
        for pattern in soft_skill_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                soft_skills.append(match.group().strip())
        
        return list(set(soft_skills))

    def _extract_certifications(self, text: str) -> List[str]:
        """Extract certification requirements"""
        cert_patterns = [
            r'\b(?:certification|certified|certificate)\s+(?:in|of)\s+([^.\n]{5,50})',
            r'\b[A-Z]{2,6}(?:\s+[A-Z]{2,6})?\s+(?:certification|certified|certificate)\b',
            r'\b(?:aws|azure|google cloud|cisco|microsoft|oracle|pmp|scrum|agile)\s+(?:certification|certified|certificate)\b'
        ]
        
        certifications = []
        for pattern in cert_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                certifications.append(match.group().strip())
        
        return list(set(certifications))

    def extract_resume_info(self, resume_text: str) -> Dict[str, any]:
        """Extract comprehensive information from resume"""
        resume_lower = resume_text.lower()
        
        # Extract skills (look for skill-related sections and context)
        skills = self._extract_resume_skills(resume_text)
        
        # Extract education
        education = self._extract_resume_education(resume_text)
        
        # Extract experience
        experience = self._extract_resume_experience(resume_text)
        
        # Extract projects
        projects = self._extract_resume_projects(resume_text)
        
        # Extract certifications
        certifications = self._extract_resume_certifications(resume_text)
        
        # Work authorization indicators
        work_auth = self._check_work_authorization(resume_text)
        
        return {
            'skills': skills,
            'education': education,
            'experience': experience,
            'projects': projects,
            'certifications': certifications,
            'work_authorization': work_auth
        }

    def _extract_resume_skills(self, text: str) -> List[str]:
        """Extract skills from resume"""
        skills = []
        
        # Look for skills sections
        skills_sections = re.finditer(r'(?:technical\s+)?skills?[:\s]*([^.\n]{20,300})', text, re.IGNORECASE)
        for match in skills_sections:
            skill_text = match.group(1)
            extracted = self._clean_and_split_skills(skill_text)
            skills.extend(extracted)
        
        # Look for skills in context
        skill_contexts = re.finditer(r'(?:proficient|experienced|skilled|expertise)\s+(?:in|with)\s+([^.\n]{10,100})', text, re.IGNORECASE)
        for match in skill_contexts:
            skill_text = match.group(1)
            extracted = self._clean_and_split_skills(skill_text)
            skills.extend(extracted)
        
        return list(set(skills))

    def _extract_resume_education(self, text: str) -> List[Dict[str, str]]:
        """Extract education information from resume"""
        education_info = []
        
        for pattern in self.degree_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                education_info.append({
                    'degree': match.group().strip(),
                    'raw_text': match.group().strip()
                })
        
        return education_info

    def _extract_resume_experience(self, text: str) -> Dict[str, any]:
        """Extract experience information from resume"""
        # Look for work experience section
        exp_section_match = re.search(r'(?:work\s+)?experience[:\s]*(.*?)(?:education|projects|skills|$)', text, re.IGNORECASE | re.DOTALL)
        
        experience_entries = []
        total_years = 0
        
        if exp_section_match:
            exp_text = exp_section_match.group(1)
            
            # Extract individual experience entries
            # Look for patterns like "Company Name | Position | Date"
            exp_patterns = [
                r'([A-Za-z\s&]+)\s*[|\-]\s*([^|\n]+)\s*[|\-]\s*([^|\n]+)',
                r'([A-Za-z\s&]+)\n([^|\n]+)\n([^|\n]+)'
            ]
            
            for pattern in exp_patterns:
                matches = re.finditer(pattern, exp_text)
                for match in matches:
                    experience_entries.append({
                        'company': match.group(1).strip(),
                        'position': match.group(2).strip(),
                        'duration': match.group(3).strip()
                    })
        
        # Try to calculate total years of experience
        date_patterns = [
            r'(\d{4})\s*[-–]\s*(\d{4}|present|current)',
            r'(\d{1,2})/(\d{4})\s*[-–]\s*(\d{1,2})/(\d{4})|present|current'
        ]
        
        current_year = datetime.now().year
        for entry in experience_entries:
            duration = entry.get('duration', '').lower()
            for pattern in date_patterns:
                match = re.search(pattern, duration, re.IGNORECASE)
                if match:
                    start_year = int(match.group(1))
                    end_year = current_year if 'present' in duration or 'current' in duration else int(match.group(2))
                    total_years += max(0, end_year - start_year)
                    break
        
        return {
            'entries': experience_entries,
            'total_years': total_years,
            'count': len(experience_entries)
        }

    def _extract_resume_projects(self, text: str) -> List[Dict[str, str]]:
        """Extract projects from resume"""
        projects = []
        
        # Look for projects section
        project_section_match = re.search(r'projects?[:\s]*(.*?)(?:education|experience|skills|$)', text, re.IGNORECASE | re.DOTALL)
        
        if project_section_match:
            project_text = project_section_match.group(1)
            
            # Split by bullet points or line breaks
            project_lines = re.split(r'[•\-\*\n]', project_text)
            for line in project_lines:
                line = line.strip()
                if len(line) > 20:  # Meaningful project description
                    projects.append({
                        'description': line,
                        'technologies': self._extract_technologies_from_text(line)
                    })
        
        return projects

    def _extract_technologies_from_text(self, text: str) -> List[str]:
        """Extract technologies/tools mentioned in text"""
        # Common technology patterns
        tech_indicators = ['using', 'with', 'in', 'built', 'developed', 'implemented']
        technologies = []
        
        text_lower = text.lower()
        
        # Look for technology mentions after indicators
        for indicator in tech_indicators:
            pattern = rf'{indicator}\s+([^.,;]+)'
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                tech_text = match.group(1).strip()
                techs = self._clean_and_split_skills(tech_text)
                technologies.extend(techs)
        
        return list(set(technologies))

    def _extract_resume_certifications(self, text: str) -> List[str]:
        """Extract certifications from resume"""
        return self._extract_certifications(text.lower())

    def _check_work_authorization(self, text: str) -> Dict[str, bool]:
        """Check for work authorization indicators"""
        text_lower = text.lower()
        
        auth_indicators = {
            'opt': any(term in text_lower for term in ['opt', 'optional practical training']),
            'cpt': any(term in text_lower for term in ['cpt', 'curricular practical training']),
            'h1b': any(term in text_lower for term in ['h1b', 'h-1b', 'h1-b']),
            'citizen': 'citizen' in text_lower,
            'green_card': any(term in text_lower for term in ['green card', 'permanent resident']),
            'authorized': any(term in text_lower for term in ['authorized to work', 'work authorization', 'eligible to work'])
        }
        
        return auth_indicators

    def calculate_skill_match(self, jd_skills: List[str], resume_skills: List[str]) -> Dict[str, any]:
        """Calculate skill matching with fuzzy matching"""
        if not jd_skills:
            return {'score': 50, 'matched': [], 'missing': [], 'match_details': {}}
        
        matched_skills = []
        match_details = {}
        
        # Exact matches
        for jd_skill in jd_skills:
            jd_skill_clean = jd_skill.lower().strip()
            for resume_skill in resume_skills:
                resume_skill_clean = resume_skill.lower().strip()
                
                # Exact match
                if jd_skill_clean == resume_skill_clean:
                    matched_skills.append(jd_skill)
                    match_details[jd_skill] = {'type': 'exact', 'matched_with': resume_skill}
                    break
                # Partial match (one contains the other)
                elif jd_skill_clean in resume_skill_clean or resume_skill_clean in jd_skill_clean:
                    if len(jd_skill_clean) > 3 and len(resume_skill_clean) > 3:  # Avoid short word false positives
                        matched_skills.append(jd_skill)
                        match_details[jd_skill] = {'type': 'partial', 'matched_with': resume_skill}
                        break
        
        # Calculate TF-IDF similarity for unmatched skills
        unmatched_jd_skills = [skill for skill in jd_skills if skill not in matched_skills]
        if unmatched_jd_skills and resume_skills:
            try:
                vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
                all_skills = unmatched_jd_skills + resume_skills
                tfidf_matrix = vectorizer.fit_transform(all_skills)
                
                jd_matrix = tfidf_matrix[:len(unmatched_jd_skills)]
                resume_matrix = tfidf_matrix[len(unmatched_jd_skills):]
                
                similarities = cosine_similarity(jd_matrix, resume_matrix)
                
                for i, jd_skill in enumerate(unmatched_jd_skills):
                    max_sim_idx = np.argmax(similarities[i])
                    max_similarity = similarities[i][max_sim_idx]
                    
                    if max_similarity > 0.3:  # Threshold for semantic similarity
                        matched_skills.append(jd_skill)
                        match_details[jd_skill] = {
                            'type': 'semantic', 
                            'matched_with': resume_skills[max_sim_idx],
                            'similarity': float(max_similarity)
                        }
            except Exception as e:
                print(f"Warning: TF-IDF similarity calculation failed: {e}")
        
        missing_skills = [skill for skill in jd_skills if skill not in matched_skills]
        score = (len(matched_skills) / len(jd_skills)) * 100 if jd_skills else 0
        
        return {
            'score': round(score, 2),
            'matched': matched_skills,
            'missing': missing_skills,
            'match_details': match_details
        }

    def calculate_experience_match(self, jd_requirements: Dict, resume_experience: Dict) -> Dict[str, any]:
        """Calculate experience matching score"""
        score = 0
        details = {}
        
        required_years = jd_requirements.get('years_required', 0)
        resume_years = resume_experience.get('total_years', 0)
        
        # Years of experience scoring
        if required_years == 0:
            score += 50  # No specific requirement
        elif resume_years >= required_years:
            score += 60  # Meets or exceeds requirement
            if resume_years > required_years * 1.5:
                score += 20  # Significantly exceeds
        elif resume_years >= required_years * 0.7:
            score += 40  # Close to requirement
        elif resume_years > 0:
            score += 20  # Some experience
        
        # Experience type matching
        if jd_requirements.get('types') and resume_experience.get('entries'):
            jd_types = [t.lower() for t in jd_requirements['types']]
            resume_text = ' '.join([
                f"{entry.get('position', '')} {entry.get('company', '')}"
                for entry in resume_experience['entries']
            ]).lower()
            
            type_matches = sum(1 for jd_type in jd_types if any(word in resume_text for word in jd_type.split()))
            if jd_types:
                type_score = (type_matches / len(jd_types)) * 40
                score += type_score
                details['type_match_score'] = type_score
        
        details.update({
            'required_years': required_years,
            'resume_years': resume_years,
            'experience_entries': len(resume_experience.get('entries', []))
        })
        
        return {
            'score': min(score, 100),
            'details': details
        }

    def calculate_education_match(self, jd_education: List[str], resume_education: List[Dict]) -> Dict[str, any]:
        """Calculate education matching score"""
        if not jd_education:
            return {'score': 50, 'details': {'message': 'No specific education requirements'}}
        
        score = 0
        details = {}
        
        if not resume_education:
            return {'score': 0, 'details': {'message': 'No education information found in resume'}}
        
        jd_education_lower = [edu.lower() for edu in jd_education]
        resume_education_text = ' '.join([edu.get('raw_text', '').lower() for edu in resume_education])
        
        # Check for degree level matches
        degree_levels = {
            'bachelor': ['bachelor', 'bs', 'ba', 'b.s', 'b.a', 'btech', 'be'],
            'master': ['master', 'ms', 'ma', 'm.s', 'm.a', 'mtech', 'me', 'mba'],
            'phd': ['phd', 'ph.d', 'doctorate']
        }
        
        required_level = None
        for level, patterns in degree_levels.items():
            if any(pattern in ' '.join(jd_education_lower) for pattern in patterns):
                required_level = level
                break
        
        has_required_level = False
        if required_level:
            patterns = degree_levels[required_level]
            has_required_level = any(pattern in resume_education_text for pattern in patterns)
        
        if has_required_level:
            score += 70
            details['degree_level_match'] = True
        elif resume_education:
            score += 30  # Has some degree
            details['degree_level_match'] = False
        
        # Field of study matching
        jd_fields = []
        for edu_req in jd_education_lower:
            # Extract field of study
            field_match = re.search(r'(?:in|of)\s+([^.\n]{5,50})', edu_req)
            if field_match:
                jd_fields.append(field_match.group(1).strip())
        
        field_match_found = False
        if jd_fields:
            for field in jd_fields:
                if any(word in resume_education_text for word in field.split() if len(word) > 3):
                    field_match_found = True
                    break
        
        if field_match_found:
            score += 30
            details['field_match'] = True
        elif jd_fields:
            details['field_match'] = False
        
        return {
            'score': min(score, 100),
            'details': details
        }

    def calculate_overall_match(self, job_desc_text: str, resume_text: str, all_resume_texts: List[str] = None) -> Dict[str, any]:
        """Calculate comprehensive matching score with advanced market analysis"""
        print("Analyzing job description...")
        jd_requirements = self.extract_skills_from_jd(job_desc_text)
        
        print("Analyzing resume...")
        resume_info = self.extract_resume_info(resume_text)
        
        print("Calculating matches...")
        
        # Core matching (60% weight)
        tech_match = self.calculate_skill_match(
            jd_requirements['technical_skills'], 
            resume_info['skills']
        )
        
        exp_match = self.calculate_experience_match(
            jd_requirements['experience_requirements'],
            resume_info['experience']
        )
        
        edu_match = self.calculate_education_match(
            jd_requirements['education_requirements'],
            resume_info['education']
        )
        
        # Advanced market analysis (40% weight)
        print("Analyzing market relevance...")
        market_analysis = self.analyze_market_relevance(resume_text)
        
        print("Analyzing leadership and impact...")
        leadership_analysis = self.analyze_leadership_impact(resume_text)
        
        print("Analyzing resume quality...")
        quality_analysis = self.analyze_resume_quality(resume_text)
        
        print("Analyzing certifications...")
        cert_analysis = self.analyze_certification_value(resume_text)
        
        # Differentiation analysis (if multiple resumes provided)
        differentiation_analysis = None
        if all_resume_texts:
            print("Analyzing differentiation...")
            differentiation_analysis = self.calculate_differentiation_score(resume_text, all_resume_texts)
        
        # Enhanced scoring algorithm
        # Core skills (25% weight)
        core_score = (
            tech_match['score'] * 0.4 +
            exp_match['score'] * 0.35 +
            edu_match['score'] * 0.25
        )
        
        # Market relevance (20% weight)
        market_score = market_analysis['overall_market_score']
        
        # Leadership & Impact (15% weight)
        leadership_score = (
            leadership_analysis['leadership_score'] * 0.6 +
            leadership_analysis['achievement_score'] * 0.4
        )
        
        # Resume quality (15% weight)
        quality_score = quality_analysis['overall_quality_score']
        
        # Certifications (10% weight)
        certification_score = cert_analysis['certification_score']
        
        # Differentiation bonus (15% weight)
        differentiation_score = differentiation_analysis['differentiation_score'] if differentiation_analysis else 50
        
        # Calculate final weighted score
        overall_score = (
            core_score * 0.25 +
            market_score * 0.20 +
            leadership_score * 0.15 +
            quality_score * 0.15 +
            certification_score * 0.10 +
            differentiation_score * 0.15
        )
        
        # Additional factors for backward compatibility
        additional_factors = {
            'projects_score': min(len(resume_info['projects']) * 10, 60),
            'certifications': cert_analysis,
            'work_authorization': resume_info['work_authorization']
        }
        
        return {
            'overall_score': round(overall_score, 1),
            'technical_skills': tech_match,
            'experience': exp_match,
            'education': edu_match,
            'market_analysis': market_analysis,
            'leadership_analysis': leadership_analysis,
            'quality_analysis': quality_analysis,
            'certification_analysis': cert_analysis,
            'differentiation_analysis': differentiation_analysis,
            'additional_factors': additional_factors,
            'jd_requirements': jd_requirements,
            'resume_analysis': resume_info,
            'score_breakdown': {
                'core_skills': round(core_score, 1),
                'market_relevance': round(market_score, 1),
                'leadership_impact': round(leadership_score, 1),
                'resume_quality': round(quality_score, 1),
                'certifications': round(certification_score, 1),
                'differentiation': round(differentiation_score, 1)
            },
            'match_summary': {
                'strengths': self._identify_advanced_strengths(
                    tech_match, exp_match, edu_match, market_analysis, 
                    leadership_analysis, quality_analysis, cert_analysis
                ),
                'improvements': self._identify_advanced_improvements(
                    tech_match, exp_match, edu_match, market_analysis,
                    leadership_analysis, quality_analysis, overall_score
                )
            }
        }

    def _identify_strengths(self, tech_match, exp_match, edu_match, additional_details) -> List[str]:
        """Identify candidate's strengths"""
        strengths = []
        
        if tech_match['score'] >= 70:
            strengths.append(f"Strong technical skills match ({tech_match['score']}%)")
        
        if exp_match['score'] >= 70:
            strengths.append(f"Relevant work experience ({exp_match['score']}%)")
        
        if edu_match['score'] >= 70:
            strengths.append(f"Educational background aligns well ({edu_match['score']}%)")
        
        if additional_details.get('projects_score', 0) >= 40:
            strengths.append("Strong project portfolio demonstrates practical skills")
        
        if additional_details.get('work_authorization', {}).get('authorized', False):
            strengths.append("Work authorization clearly stated")
        
        if additional_details.get('certifications', {}).get('score', 0) >= 50:
            strengths.append("Relevant certifications found")
        
        return strengths

    def _identify_improvements(self, tech_match, exp_match, edu_match, overall_score) -> List[str]:
        """Identify areas for improvement"""
        improvements = []
        
        if tech_match['score'] < 50:
            missing_skills = tech_match['missing'][:5]  # Top 5 missing skills
            improvements.append(f"Add key technical skills: {', '.join(missing_skills)}")
        
        if exp_match['score'] < 40:
            improvements.append("Highlight relevant work experience, internships, or projects more prominently")
        
        if edu_match['score'] < 50:
            improvements.append("Better align educational background with job requirements")
        
        if overall_score < 60:
            improvements.append("Incorporate more job-relevant keywords throughout resume")
            improvements.append("Quantify achievements and impact in previous roles/projects")
        
        return improvements

    def _identify_advanced_strengths(self, tech_match, exp_match, edu_match, market_analysis, 
                                   leadership_analysis, quality_analysis, cert_analysis) -> List[str]:
        """Identify candidate's advanced strengths for competitive market"""
        strengths = []
        
        # Core competency strengths
        if tech_match['score'] >= 70:
            strengths.append(f"Excellent technical skills match ({tech_match['score']:.1f}%)")
        elif tech_match['score'] >= 50:
            strengths.append(f"Good technical foundation ({tech_match['score']:.1f}%)")
        
        if exp_match['score'] >= 70:
            strengths.append(f"Strong relevant experience ({exp_match['score']:.1f}%)")
        
        if edu_match['score'] >= 70:
            strengths.append(f"Educational background aligns well ({edu_match['score']:.1f}%)")
        
        # Market relevance strengths
        if market_analysis['overall_market_score'] >= 15:
            top_categories = sorted(market_analysis['category_scores'].items(), 
                                  key=lambda x: x[1]['score'], reverse=True)
            if top_categories[0][1]['score'] > 20:
                strengths.append(f"Strong {top_categories[0][0].replace('_', ' ')} expertise ({top_categories[0][1]['score']:.1f}%)")
        
        # Leadership and impact
        if leadership_analysis['leadership_score'] >= 60:
            strengths.append(f"Demonstrated leadership experience ({leadership_analysis['leadership_count']} indicators)")
        
        if leadership_analysis['achievement_score'] >= 40:
            strengths.append(f"Quantified achievements ({leadership_analysis['achievement_count']} metrics)")
        
        # Resume quality
        if quality_analysis['overall_quality_score'] >= 75:
            strengths.append("Professional, well-structured resume")
        
        if quality_analysis['action_verb_score'] >= 70:
            strengths.append("Strong action-oriented language")
        
        if quality_analysis['technical_depth_score'] >= 60:
            strengths.append("Good technical depth and terminology")
        
        # Certifications
        if cert_analysis['certification_score'] >= 50:
            strengths.append(f"Valuable industry certifications ({cert_analysis['certification_count']})")
        
        return strengths

    def _identify_advanced_improvements(self, tech_match, exp_match, edu_match, market_analysis,
                                      leadership_analysis, quality_analysis, overall_score) -> List[str]:
        """Identify advanced improvement recommendations for competitive advantage"""
        improvements = []
        
        # Technical skills gaps
        if tech_match['score'] < 50:
            missing_skills = tech_match['missing'][:3]
            improvements.append(f"Priority skills to acquire: {', '.join(missing_skills)}")
        
        # Market relevance improvements
        weak_categories = [(cat, data) for cat, data in market_analysis['category_scores'].items() 
                          if data['score'] < 10]
        if weak_categories:
            top_weak = min(weak_categories, key=lambda x: x[1]['score'])
            improvements.append(f"Consider learning trending {top_weak[0].replace('_', ' ')} technologies")
        
        # Leadership development
        if leadership_analysis['leadership_score'] < 30:
            improvements.append("Highlight leadership roles, team collaboration, or mentoring experience")
        
        if leadership_analysis['achievement_score'] < 20:
            improvements.append("Add quantifiable metrics and business impact to achievements")
        
        # Resume quality improvements
        if quality_analysis['action_verb_score'] < 50:
            improvements.append("Use more action verbs to demonstrate impact and initiative")
        
        if quality_analysis['technical_depth_score'] < 40:
            improvements.append("Include more technical details about tools, frameworks, and methodologies")
        
        if quality_analysis['readability_score'] < 60:
            improvements.append("Improve resume clarity and professional writing style")
        
        # Experience enhancement
        if exp_match['score'] < 40:
            improvements.append("Emphasize relevant projects, internships, or freelance work")
        
        # Overall competitive positioning
        if overall_score < 70:
            improvements.append("Focus on skills that differentiate you from other candidates")
            improvements.append("Consider contributing to open source projects or building a portfolio")
        
        return improvements

    def _identify_competitive_advantages(self, match_result: Dict) -> List[str]:
        """Identify what makes this candidate stand out in competitive market"""
        advantages = []
        
        # Top quartile performances
        if match_result['score_breakdown']['market_relevance'] >= 20:
            advantages.append("Strong alignment with current market trends")
        
        if match_result['leadership_analysis']['leadership_score'] >= 70:
            advantages.append("Proven leadership experience")
        
        if match_result['quality_analysis']['overall_quality_score'] >= 80:
            advantages.append("Exceptionally well-crafted resume")
        
        if match_result['differentiation_analysis'] and match_result['differentiation_analysis']['differentiation_score'] >= 70:
            advantages.append("Unique skills that differentiate from other candidates")
        
        # Special combinations
        tech_score = match_result['technical_skills']['score']
        market_score = match_result['score_breakdown']['market_relevance']
        
        if tech_score >= 60 and market_score >= 15:
            advantages.append("Strong combination of technical depth and market relevance")
        
        if match_result['leadership_analysis']['achievement_count'] >= 3:
            advantages.append("Multiple quantified achievements demonstrate impact")
        
        return advantages

    def _calculate_market_positioning(self, match_result: Dict) -> Dict[str, str]:
        """Calculate how candidate positions in current job market"""
        overall_score = match_result['overall_score']
        market_score = match_result['score_breakdown']['market_relevance']
        quality_score = match_result['score_breakdown']['resume_quality']
        
        # Market tier classification
        if overall_score >= 80:
            tier = "Top Tier"
            description = "Exceptional candidate with strong competitive advantage"
        elif overall_score >= 65:
            tier = "High Potential"
            description = "Strong candidate with good market positioning"
        elif overall_score >= 50:
            tier = "Competitive"
            description = "Solid candidate who could benefit from targeted improvements"
        elif overall_score >= 35:
            tier = "Developing"
            description = "Candidate with potential but needs significant skill development"
        else:
            tier = "Entry Level"
            description = "Candidate requiring substantial development for this role"
        
        # Market readiness assessment
        if market_score >= 20 and quality_score >= 70:
            readiness = "Market Ready"
        elif market_score >= 10 and quality_score >= 60:
            readiness = "Nearly Ready"
        elif market_score >= 5 or quality_score >= 50:
            readiness = "Needs Preparation"
        else:
            readiness = "Significant Development Required"
        
        return {
            'tier': tier,
            'description': description,
            'market_readiness': readiness,
            'recommendation': self._get_positioning_recommendation(tier, readiness)
        }
    
    def _get_positioning_recommendation(self, tier: str, readiness: str) -> str:
        """Get specific positioning recommendation"""
        if tier == "Top Tier":
            return "Prioritize for immediate interview - strong hire potential"
        elif tier == "High Potential":
            return "Schedule interview - good candidate with competitive skills"
        elif tier == "Competitive" and readiness in ["Market Ready", "Nearly Ready"]:
            return "Consider for interview - assess cultural fit and growth potential"
        elif tier == "Developing":
            return "Consider for junior roles or internship programs"
        else:
            return "Recommend skill development before consideration"

    def _calculate_additional_factors_score(self, additional_factors: Dict) -> float:
        """Calculate a numeric score from additional factors"""
        total_score = 0
        
        if isinstance(additional_factors, dict):
            for key, value in additional_factors.items():
                if isinstance(value, (int, float)):
                    total_score += value
                elif isinstance(value, dict) and 'score' in value:
                    total_score += value['score']
                elif isinstance(value, bool) and value:
                    total_score += 10  # Bonus for boolean factors
        
        return total_score

    def analyze_market_relevance(self, resume_text: str) -> Dict[str, any]:
        """Analyze how well resume aligns with current market trends"""
        resume_lower = resume_text.lower()
        
        market_scores = {}
        total_trending_found = 0
        
        # Check trending skills by category
        for category, skills in self.trending_skills.items():
            found_skills = []
            for skill in skills:
                if skill.lower() in resume_lower:
                    found_skills.append(skill)
                    total_trending_found += 1
            
            category_score = (len(found_skills) / len(skills)) * 100
            market_scores[category] = {
                'score': round(category_score, 1),
                'found_skills': found_skills,
                'total_possible': len(skills)
            }
        
        # Overall market relevance score
        all_trending_skills = [skill for skills in self.trending_skills.values() for skill in skills]
        overall_market_score = (total_trending_found / len(all_trending_skills)) * 100
        
        return {
            'overall_market_score': round(overall_market_score, 1),
            'category_scores': market_scores,
            'total_trending_skills_found': total_trending_found
        }

    def analyze_leadership_impact(self, resume_text: str) -> Dict[str, any]:
        """Analyze leadership experience and quantifiable impact"""
        resume_lower = resume_text.lower()
        
        # Find leadership indicators
        leadership_found = []
        for indicator in self.leadership_indicators:
            if indicator.lower() in resume_lower:
                leadership_found.append(indicator)
        
        leadership_score = min((len(leadership_found) / 5) * 100, 100)  # Max score for 5+ indicators
        
        # Find quantifiable achievements
        achievements = []
        for pattern in self.achievement_patterns:
            matches = re.finditer(pattern, resume_text, re.IGNORECASE)
            for match in matches:
                achievements.append(match.group())
        
        achievement_score = min(len(achievements) * 20, 100)  # 20 points per quantified achievement
        
        return {
            'leadership_score': round(leadership_score, 1),
            'achievement_score': round(achievement_score, 1),
            'leadership_indicators': leadership_found,
            'quantified_achievements': achievements,
            'leadership_count': len(leadership_found),
            'achievement_count': len(achievements)
        }

    def analyze_resume_quality(self, resume_text: str) -> Dict[str, any]:
        """Analyze overall resume quality and readability"""
        
        # Text quality metrics
        word_count = len(resume_text.split())
        sentence_count = len(sent_tokenize(resume_text))
        
        # Readability (prefer professional but accessible writing)
        try:
            flesch_score = textstat.flesch_reading_ease(resume_text)
            grade_level = textstat.flesch_kincaid_grade(resume_text)
        except Exception as e:
            print(f"Warning: Readability analysis failed: {e}")
            flesch_score = 65  # Default neutral score
            grade_level = 12
        
        # Optimal readability for professional documents: 60-70 Flesch score
        readability_score = 100 - abs(65 - flesch_score)  # Penalty for deviation from optimal
        readability_score = max(0, min(100, readability_score))
        
        # Check for action verbs (important for impact)
        action_verbs = [
            'achieved', 'developed', 'implemented', 'created', 'designed', 'built', 'improved',
            'optimized', 'automated', 'streamlined', 'delivered', 'executed', 'launched',
            'maintained', 'managed', 'collaborated', 'analyzed', 'researched', 'tested'
        ]
        
        action_verb_count = sum(1 for verb in action_verbs if verb.lower() in resume_text.lower())
        action_verb_score = min((action_verb_count / 10) * 100, 100)  # 10+ action verbs = 100%
        
        # Check for technical depth indicators
        technical_depth_indicators = [
            'architecture', 'framework', 'algorithm', 'optimization', 'scalability',
            'performance', 'security', 'integration', 'deployment', 'testing',
            'debugging', 'troubleshooting', 'documentation', 'monitoring'
        ]
        
        tech_depth_count = sum(1 for term in technical_depth_indicators if term.lower() in resume_text.lower())
        tech_depth_score = min((tech_depth_count / 8) * 100, 100)  # 8+ terms = 100%
        
        # Length assessment (1-2 pages is optimal)
        length_score = 100
        if word_count < 300:
            length_score = 40  # Too short
        elif word_count > 1500:
            length_score = 70  # Too long
        
        return {
            'overall_quality_score': round((readability_score + action_verb_score + tech_depth_score + length_score) / 4, 1),
            'readability_score': round(readability_score, 1),
            'action_verb_score': round(action_verb_score, 1),
            'technical_depth_score': round(tech_depth_score, 1),
            'length_score': round(length_score, 1),
            'word_count': word_count,
            'sentence_count': sentence_count,
            'flesch_reading_ease': round(flesch_score, 1),
            'grade_level': round(grade_level, 1),
            'action_verbs_found': action_verb_count,
            'tech_depth_indicators': tech_depth_count
        }

    def analyze_certification_value(self, resume_text: str) -> Dict[str, any]:
        """Analyze the market value of certifications mentioned"""
        resume_lower = resume_text.lower()
        
        valuable_certs_found = []
        for cert in self.valuable_certifications:
            if cert.lower() in resume_lower:
                valuable_certs_found.append(cert)
        
        # Higher score for more valuable certifications
        cert_score = min(len(valuable_certs_found) * 25, 100)  # 25 points per valuable cert
        
        return {
            'certification_score': round(cert_score, 1),
            'valuable_certifications': valuable_certs_found,
            'certification_count': len(valuable_certs_found)
        }

    def calculate_differentiation_score(self, resume_text: str, all_resume_texts: List[str]) -> Dict[str, any]:
        """Calculate how much this resume stands out from others"""
        
        if len(all_resume_texts) <= 1:
            return {'differentiation_score': 50, 'unique_elements': []}
        
        # Extract unique skills/terms that appear in this resume but not in others
        resume_words = set(word_tokenize(resume_text.lower()))
        resume_words = {word for word in resume_words if len(word) > 3 and word not in self.stop_words}
        
        # Compare with other resumes
        other_words = set()
        for other_resume in all_resume_texts:
            if other_resume != resume_text:
                other_words.update(word_tokenize(other_resume.lower()))
        
        other_words = {word for word in other_words if len(word) > 3 and word not in self.stop_words}
        
        # Find unique elements
        unique_words = resume_words - other_words
        unique_score = min((len(unique_words) / 20) * 100, 100)  # 20+ unique terms = 100%
        
        # Check for rare/specialized skills
        specialized_terms = []
        for word in unique_words:
            if any(word in category_skills for category_skills in self.trending_skills.values() for skill in category_skills if word in skill.lower()):
                specialized_terms.append(word)
        
        specialization_bonus = min(len(specialized_terms) * 10, 50)  # Up to 50 bonus points
        
        final_score = min(unique_score + specialization_bonus, 100)
        
        return {
            'differentiation_score': round(final_score, 1),
            'unique_elements': list(unique_words)[:10],  # Top 10 unique elements
            'specialized_terms': specialized_terms,
            'uniqueness_ratio': round(len(unique_words) / len(resume_words) * 100, 1) if resume_words else 0
        }

    def process_resumes_batch(self, job_desc_folder: str, resumes_folder: str, output_file: str = "smart_ats_results.json"):
        """Process multiple resumes against job description with enhanced analysis"""
        
        if not os.path.exists(job_desc_folder):
            print(f"❌ Job description folder '{job_desc_folder}' not found!")
            return
            
        if not os.path.exists(resumes_folder):
            print(f"❌ Resumes folder '{resumes_folder}' not found!")
            return
        
        # Get job description
        jd_files = [f for f in os.listdir(job_desc_folder) if f.endswith('.pdf')]
        if not jd_files:
            print("❌ No job description PDF found!")
            return
            
        print(f"📄 Loading job description: {jd_files[0]}")
        job_desc_path = os.path.join(job_desc_folder, jd_files[0])
        job_description = self.extract_text_from_pdf(job_desc_path)
        
        if not job_description.strip():
            print("❌ Could not extract text from job description PDF!")
            return
        
        # Get all resume files
        resume_files = [f for f in os.listdir(resumes_folder) if f.endswith('.pdf')]
        if not resume_files:
            print("❌ No resume PDFs found!")
            return
            
        print(f"📊 Processing {len(resume_files)} resumes...")
        print("=" * 60)
        
        results = []
        processing_errors = []
        all_resume_texts = []
        
        # First pass: extract all resume texts for differentiation analysis
        print("📊 Pre-processing resumes for advanced analysis...")
        for resume_file in resume_files:
            try:
                resume_path = os.path.join(resumes_folder, resume_file)
                resume_text = self.extract_text_from_pdf(resume_path)
                if resume_text.strip():
                    all_resume_texts.append(resume_text)
            except Exception as e:
                print(f"  ⚠️  Pre-processing failed for {resume_file}: {e}")
        
        # Second pass: detailed analysis with differentiation
        for i, resume_file in enumerate(resume_files, 1):
            try:
                print(f"🔍 Processing ({i}/{len(resume_files)}): {resume_file}")
                
                resume_path = os.path.join(resumes_folder, resume_file)
                resume_text = self.extract_text_from_pdf(resume_path)
                
                if not resume_text.strip():
                    processing_errors.append(f"Could not extract text from {resume_file}")
                    print(f"  ⚠️  Text extraction failed")
                    continue
                
                # Calculate comprehensive match with advanced analysis
                match_result = self.calculate_overall_match(job_description, resume_text, all_resume_texts)
                
                candidate_result = {
                    'candidate_name': resume_file.replace('.pdf', ''),
                    'file_name': resume_file,
                    'overall_score': match_result['overall_score'],
                    'score_breakdown': {
                        'core_skills': match_result['score_breakdown']['core_skills'],
                        'market_relevance': match_result['score_breakdown']['market_relevance'],
                        'leadership_impact': match_result['score_breakdown']['leadership_impact'],
                        'resume_quality': match_result['score_breakdown']['resume_quality'],
                        'certifications': match_result['score_breakdown']['certifications'],
                        'differentiation': match_result['score_breakdown']['differentiation']
                    },
                    'detailed_analysis': {
                        'technical_skills': {
                            'matched': match_result['technical_skills']['matched'],
                            'missing': match_result['technical_skills']['missing'],
                            'match_details': match_result['technical_skills']['match_details']
                        },
                        'market_analysis': match_result['market_analysis'],
                        'leadership_analysis': match_result['leadership_analysis'],
                        'quality_analysis': match_result['quality_analysis'],
                        'certification_analysis': match_result['certification_analysis'],
                        'differentiation_analysis': match_result['differentiation_analysis'],
                        'experience_analysis': match_result['experience']['details'],
                        'education_analysis': match_result['education']['details']
                    },
                    'competitive_advantages': self._identify_competitive_advantages(match_result),
                    'market_positioning': self._calculate_market_positioning(match_result),
                    'strengths': match_result['match_summary']['strengths'],
                    'recommendations': match_result['match_summary']['improvements'],
                    'extracted_info': {
                        'skills_found': len(match_result['resume_analysis']['skills']),
                        'experience_years': match_result['resume_analysis']['experience']['total_years'],
                        'education_count': len(match_result['resume_analysis']['education']),
                        'projects_count': len(match_result['resume_analysis']['projects']),
                        'certifications_count': len(match_result['resume_analysis']['certifications']),
                        'leadership_indicators': match_result['leadership_analysis']['leadership_count'],
                        'quantified_achievements': match_result['leadership_analysis']['achievement_count']
                    }
                }
                
                results.append(candidate_result)
                print(f"  ✅ Score: {match_result['overall_score']:.1f}% | Market: {match_result['score_breakdown']['market_relevance']:.1f}% | Quality: {match_result['score_breakdown']['resume_quality']:.1f}%")
                
            except Exception as e:
                error_msg = f"Error processing {resume_file}: {str(e)}"
                processing_errors.append(error_msg)
                print(f"  ❌ Error: {str(e)}")
                continue
        
        if not results:
            print("❌ No resumes were successfully processed!")
            return
        
        # Sort by overall score (highest first)
        results.sort(key=lambda x: x['overall_score'], reverse=True)
        
        # Add ranking
        for i, result in enumerate(results, 1):
            result['rank'] = i
        
        # Generate comprehensive results
        analysis_results = {
            'job_description_file': jd_files[0],
            'processing_timestamp': datetime.now().isoformat(),
            'total_candidates': len(results),
            'processing_errors': processing_errors,
            'job_requirements_extracted': results[0]['detailed_analysis'] if results else None,
            'candidates': results,
            'statistics': {
                'score_distribution': {
                    'excellent_80_plus': len([r for r in results if r['overall_score'] >= 80]),
                    'good_60_to_79': len([r for r in results if 60 <= r['overall_score'] < 80]),
                    'fair_40_to_59': len([r for r in results if 40 <= r['overall_score'] < 60]),
                    'poor_below_40': len([r for r in results if r['overall_score'] < 40])
                },
                'average_score': round(sum(r['overall_score'] for r in results) / len(results), 1) if results else 0,
                'median_score': round(sorted([r['overall_score'] for r in results])[len(results)//2], 1) if results else 0
            }
        }
        
        # Save comprehensive results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 80)
        print("🎯 SMART ATS ANALYSIS COMPLETE")
        print("=" * 80)
        
        self.print_comprehensive_summary(results, analysis_results['statistics'])
        
        # Generate CSV summary for easy analysis
        self.generate_csv_summary(results, output_file)
        
        print(f"\n📁 Detailed results saved to: {output_file}")
        
        return results

    def print_comprehensive_summary(self, results: List[Dict], statistics: Dict):
        """Print detailed analysis summary"""
        
        print(f"📊 PROCESSING SUMMARY:")
        print(f"   📈 Average Score: {statistics['average_score']}%")
        print(f"   📊 Median Score: {statistics['median_score']}%")
        print(f"   📋 Total Candidates: {len(results)}")
        
        print(f"\n📈 SCORE DISTRIBUTION:")
        dist = statistics['score_distribution']
        print(f"   🟢 Excellent (80%+): {dist['excellent_80_plus']} candidates")
        print(f"   🟡 Good (60-79%):    {dist['good_60_to_79']} candidates") 
        print(f"   🟠 Fair (40-59%):    {dist['fair_40_to_59']} candidates")
        print(f"   🔴 Poor (<40%):      {dist['poor_below_40']} candidates")
        
        print(f"\n🏆 TOP 10 CANDIDATES:")
        for i, result in enumerate(results[:10], 1):
            score = result['overall_score']
            name = result['candidate_name']
            
            if score >= 80:
                emoji = "🟢"
            elif score >= 60:
                emoji = "🟡"
            elif score >= 40:
                emoji = "🟠"
            else:
                emoji = "🔴"
                
            # Show key strengths
            strengths = result['strengths'][:2]  # Top 2 strengths
            strength_text = f" | {'; '.join(strengths)}" if strengths else ""
            
            print(f"   {i:2d}. {emoji} {name:<35} {score:5.1f}%{strength_text}")
        
        # Recommendations for top candidates
        excellent_candidates = [r for r in results if r['overall_score'] >= 80]
        good_candidates = [r for r in results if r['overall_score'] >= 60]
        
        if excellent_candidates:
            print(f"\n💼 RECOMMENDED FOR IMMEDIATE INTERVIEW:")
            for candidate in excellent_candidates[:5]:
                print(f"   • {candidate['candidate_name']} ({candidate['overall_score']:.1f}%)")
        elif good_candidates:
            print(f"\n💼 RECOMMENDED FOR INTERVIEW:")
            for candidate in good_candidates[:3]:
                print(f"   • {candidate['candidate_name']} ({candidate['overall_score']:.1f}%)")
        else:
            # Show top candidates even if scores are low
            print(f"\n💼 TOP CANDIDATES FOR CONSIDERATION:")
            for candidate in results[:3]:
                positioning = candidate.get('market_positioning', {})
                tier = positioning.get('tier', 'Unknown')
                print(f"   • {candidate['candidate_name']} ({candidate['overall_score']:.1f}%) - {tier}")
        
        # Show common gaps safely
        print(f"\n🔍 COMMON SKILL GAPS:")
        all_missing_skills = []
        for result in results:
            # Safely access missing skills
            tech_analysis = result.get('detailed_analysis', {}).get('technical_skills', {})
            missing = tech_analysis.get('missing', [])
            all_missing_skills.extend(missing)
        
        if all_missing_skills:
            skill_counts = Counter(all_missing_skills)
            top_missing = skill_counts.most_common(5)
            for skill, count in top_missing:
                percentage = (count / len(results)) * 100
                print(f"   • {skill}: Missing in {count}/{len(results)} candidates ({percentage:.1f}%)")
        else:
            print("   No common skill gaps identified")

    def generate_csv_summary(self, results: List[Dict], output_file: str):
        """Generate CSV summary for easy spreadsheet analysis"""
        csv_data = []
        
        for result in results:
            csv_data.append({
                'Rank': result['rank'],
                'Candidate_Name': result['candidate_name'],
                'Overall_Score': result['overall_score'],
                'Core_Skills_Score': result['score_breakdown'].get('core_skills', 0),
                'Market_Relevance_Score': result['score_breakdown'].get('market_relevance', 0),
                'Leadership_Impact_Score': result['score_breakdown'].get('leadership_impact', 0),
                'Resume_Quality_Score': result['score_breakdown'].get('resume_quality', 0),
                'Certifications_Score': result['score_breakdown'].get('certifications', 0),
                'Differentiation_Score': result['score_breakdown'].get('differentiation', 0),
                'Skills_Found_Count': result['extracted_info']['skills_found'],
                'Experience_Years': result['extracted_info']['experience_years'],
                'Projects_Count': result['extracted_info']['projects_count'],
                'Leadership_Indicators': result['extracted_info'].get('leadership_indicators', 0),
                'Quantified_Achievements': result['extracted_info'].get('quantified_achievements', 0),
                'Market_Tier': result.get('market_positioning', {}).get('tier', 'Unknown'),
                'Market_Readiness': result.get('market_positioning', {}).get('market_readiness', 'Unknown'),
                'Top_Strength': result['strengths'][0] if result['strengths'] else 'None',
                'Top_Competitive_Advantage': result.get('competitive_advantages', ['None'])[0] if result.get('competitive_advantages') else 'None',
                'Top_Recommendation': result['recommendations'][0] if result['recommendations'] else 'None',
                'Interview_Recommendation': result.get('market_positioning', {}).get('recommendation', 'Review profile')
            })
        
        df = pd.DataFrame(csv_data)
        csv_file = output_file.replace('.json', '.csv')
        df.to_csv(csv_file, index=False)
        print(f"📊 Enhanced CSV summary saved to: {csv_file}")

    def analyze_single_resume(self, job_desc_path: str, resume_path: str) -> Dict:
        """Analyze a single resume against job description"""
        
        print("📄 Loading job description...")
        job_description = self.extract_text_from_pdf(job_desc_path)
        
        print("📄 Loading resume...")
        resume_text = self.extract_text_from_pdf(resume_path)
        
        if not job_description.strip() or not resume_text.strip():
            print("❌ Could not extract text from PDF files!")
            return None
        
        print("🔍 Analyzing match...")
        match_result = self.calculate_overall_match(job_description, resume_text)
        
        print(f"\n🎯 ANALYSIS RESULTS")
        print("=" * 50)
        print(f"Overall Match Score: {match_result['overall_score']:.1f}%")
        print(f"Technical Skills: {match_result['technical_skills']['score']:.1f}%")
        print(f"Experience: {match_result['experience']['score']:.1f}%")
        print(f"Education: {match_result['education']['score']:.1f}%")
        
        if match_result['match_summary']['strengths']:
            print(f"\n✅ STRENGTHS:")
            for strength in match_result['match_summary']['strengths']:
                print(f"   • {strength}")
        
        if match_result['match_summary']['improvements']:
            print(f"\n🔧 RECOMMENDATIONS:")
            for improvement in match_result['match_summary']['improvements']:
                print(f"   • {improvement}")
        
        return match_result


def main():
    """Main function with enhanced CLI"""
    print("🚀 ADVANCED ATS RESUME MATCHER")
    print("=" * 60)
    print("AI-powered resume screening optimized for today's competitive market")
    print("✨ Enhanced with market trend analysis & competitive positioning")
    print()
    
    matcher = SmartATSMatcher()
    
    # Configuration
    job_desc_folder = "jD"
    resumes_folder = "Candidate_resume"
    output_file = "advanced_ats_analysis.json"
    
    print(f"📁 Configuration:")
    print(f"   Job Descriptions: {job_desc_folder}")
    print(f"   Resumes: {resumes_folder}")
    print(f"   Output: {output_file}")
    print()
    
    # Check if folders exist
    folders_exist = True
    if not os.path.exists(job_desc_folder):
        print(f"⚠️  Create folder: {job_desc_folder}")
        folders_exist = False
    if not os.path.exists(resumes_folder):
        print(f"⚠️  Create folder: {resumes_folder}")
        folders_exist = False
    
    if not folders_exist:
        print("\n📝 SETUP INSTRUCTIONS:")
        print("1. Create the required folders")
        print("2. Place ONE job description PDF in job_descriptions/")
        print("3. Place resume PDFs in resumes/")
        print("4. Run this script again")
        return
    
    try:
        print("🔄 Starting advanced analysis...")
        results = matcher.process_resumes_batch(job_desc_folder, resumes_folder, output_file)
        
        if results:
            print(f"\n🎉 Advanced analysis completed successfully!")
            print(f"   📊 {len(results)} candidates analyzed with market intelligence")
            print(f"   🏆 Top candidate: {results[0]['candidate_name']} ({results[0]['overall_score']:.1f}%)")
            print(f"   📈 Market positioning: {results[0]['market_positioning']['tier']}")
            print(f"   📁 Detailed results: {output_file}")
        else:
            print("❌ No results generated. Check your input files.")
            
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        print("Please check your input files and try again.")

if __name__ == "__main__":
    main()