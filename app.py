from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for, make_response
import os
import json
import zipfile
from werkzeug.utils import secure_filename
from datetime import datetime
import tempfile
import shutil
from ats_parser import SmartATSMatcher
import pandas as pd
from io import BytesIO
import base64

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'job_descriptions'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'resumes'), exist_ok=True)

ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Homepage with feature showcase"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint for deployment monitoring"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'ATS Resume Matcher'
    })

@app.route('/upload')
def upload_page():
    """Upload page for job descriptions and resumes"""
    return render_template('upload.html')

@app.route('/demo')
def demo_page():
    """Demo page showing sample analysis"""
    try:
        # Get sample data directly from function
        sample_data = {
            "job_description_file": "AI Engineer - Job Description.pdf",
            "job_title": "AI Engineer - Machine Learning",
            "processing_timestamp": datetime.now().isoformat(),
            "total_candidates": 7,
            "processing_time": "1.8 minutes",
            "processing_errors": [],
            "candidates": [
                {
                    "rank": 1,
                    "candidate_name": "lateset_shine_res",
                    "overall_score": 44.1,
                    "score_breakdown": {
                        "core_skills": 27.6,
                        "market_relevance": 17.7,
                        "leadership_impact": 36.0,
                        "resume_quality": 71.9,
                        "certifications": 25,
                        "differentiation": 100
                    },
                    "market_positioning": {
                        "tier": "Developing",
                        "market_readiness": "Nearly Ready",
                        "recommendation": "Consider for junior roles or internship programs"
                    },
                    "strengths": [
                        "Strong AI/ML expertise (42.1%)",
                        "Demonstrated leadership experience (3 indicators)",
                        "Good technical depth and terminology"
                    ],
                    "competitive_advantages": [
                        "Unique skills that differentiate from other candidates",
                        "Strong alignment with current market trends"
                    ]
                },
                {
                    "rank": 2,
                    "candidate_name": "fullstack",
                    "overall_score": 40.0,
                    "score_breakdown": {
                        "core_skills": 27.6,
                        "market_relevance": 24.1,
                        "leadership_impact": 0.0,
                        "resume_quality": 71.7,
                        "certifications": 25,
                        "differentiation": 100
                    },
                    "market_positioning": {
                        "tier": "Developing",
                        "market_readiness": "Market Ready",
                        "recommendation": "Consider for junior roles or internship programs"
                    },
                    "strengths": [
                        "Strong Cloud/DevOps expertise (38.5%)",
                        "Good technical depth and terminology"
                    ],
                    "competitive_advantages": [
                        "Strong alignment with current market trends",
                        "Versatile technology stack"
                    ]
                },
                {
                    "rank": 3,
                    "candidate_name": "java_resume",
                    "overall_score": 40.0,
                    "score_breakdown": {
                        "core_skills": 28.7,
                        "market_relevance": 20.3,
                        "leadership_impact": 24.0,
                        "resume_quality": 67.6,
                        "certifications": 0,
                        "differentiation": 100
                    },
                    "market_positioning": {
                        "tier": "Developing",
                        "market_readiness": "Nearly Ready",
                        "recommendation": "Consider for junior roles or internship programs"
                    },
                    "strengths": [
                        "Strong Cloud/DevOps expertise (38.5%)",
                        "Strong action-oriented language"
                    ],
                    "competitive_advantages": [
                        "Strong alignment with current market trends"
                    ]
                }
            ],
            "statistics": {
                "score_distribution": {
                    "excellent_80_plus": 0,
                    "good_60_to_79": 0,
                    "fair_40_to_59": 3,
                    "poor_below_40": 4
                },
                "average_score": 37.2,
                "median_score": 36.2
            },
            "market_gaps": [
                "Generative AI applications",
                "Vector databases", 
                "Time series analysis"
            ]
        }
        return render_template('demo.html', results=sample_data)
    except Exception as e:
        print(f"Demo page error: {e}")
        # Return empty demo page
        return render_template('demo.html', results=None)

@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Handle file uploads via API"""
    try:
        # Clear previous uploads
        upload_folder = app.config['UPLOAD_FOLDER']
        if os.path.exists(upload_folder):
            shutil.rmtree(upload_folder)
        os.makedirs(upload_folder, exist_ok=True)
        os.makedirs(os.path.join(upload_folder, 'job_descriptions'), exist_ok=True)
        os.makedirs(os.path.join(upload_folder, 'resumes'), exist_ok=True)
        
        # Handle job description upload
        if 'job_description' not in request.files:
            return jsonify({'error': 'No job description file provided'}), 400
        
        jd_file = request.files['job_description']
        if jd_file.filename == '':
            return jsonify({'error': 'No job description file selected'}), 400
        
        if not allowed_file(jd_file.filename):
            return jsonify({'error': 'Job description must be a PDF file'}), 400
        
        # Save job description
        jd_filename = secure_filename(jd_file.filename)
        jd_path = os.path.join(upload_folder, 'job_descriptions', jd_filename)
        jd_file.save(jd_path)
        
        # Handle resume uploads
        if 'resumes' not in request.files:
            return jsonify({'error': 'No resume files provided'}), 400
        
        resume_files = request.files.getlist('resumes')
        if not resume_files or all(f.filename == '' for f in resume_files):
            return jsonify({'error': 'No resume files selected'}), 400
        
        saved_resumes = []
        for resume_file in resume_files:
            if resume_file.filename != '' and allowed_file(resume_file.filename):
                resume_filename = secure_filename(resume_file.filename)
                resume_path = os.path.join(upload_folder, 'resumes', resume_filename)
                resume_file.save(resume_path)
                saved_resumes.append(resume_filename)
        
        if not saved_resumes:
            return jsonify({'error': 'No valid resume files uploaded'}), 400
        
        return jsonify({
            'success': True,
            'message': f'Successfully uploaded job description and {len(saved_resumes)} resumes',
            'job_description': jd_filename,
            'resumes': saved_resumes
        })
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_resumes():
    """Run ATS analysis on uploaded files"""
    try:
        upload_folder = app.config['UPLOAD_FOLDER']
        jd_folder = os.path.join(upload_folder, 'job_descriptions')
        resume_folder = os.path.join(upload_folder, 'resumes')
        
        # Check if folders exist and have files
        if not os.path.exists(jd_folder) or not os.listdir(jd_folder):
            return jsonify({'error': 'No job description found. Please upload a job description first.'}), 400
        
        if not os.path.exists(resume_folder) or not os.listdir(resume_folder):
            return jsonify({'error': 'No resumes found. Please upload resume files first.'}), 400
        
        # Initialize ATS matcher
        matcher = SmartATSMatcher()
        
        # Generate unique output filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'web_analysis_{timestamp}.json'
        
        # Run analysis
        results = matcher.process_resumes_batch(jd_folder, resume_folder, output_file)
        
        if not results:
            return jsonify({'error': 'Analysis failed. Please check your files and try again.'}), 500
        
        # Load the detailed results
        with open(output_file, 'r', encoding='utf-8') as f:
            detailed_results = json.load(f)
        
        # Clean up output file
        if os.path.exists(output_file):
            os.remove(output_file)
        
        return jsonify({
            'success': True,
            'results': detailed_results,
            'summary': {
                'total_candidates': len(results),
                'top_candidate': results[0]['candidate_name'] if results else None,
                'top_score': results[0]['overall_score'] if results else 0,
                'average_score': sum(r['overall_score'] for r in results) / len(results) if results else 0
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/download-results')
def download_results():
    """Download analysis results as CSV"""
    try:
        # Check if we have recent analysis results
        sample_file = 'advanced_ats_analysis.csv'
        if os.path.exists(sample_file):
            return send_file(sample_file, as_attachment=True, download_name='ats_analysis_results.csv')
        else:
            return jsonify({'error': 'No analysis results available for download'}), 404
            
    except Exception as e:
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

@app.route('/results')
def results_page():
    """Display analysis results"""
    try:
        # Get sample data directly for demo purposes
        sample_data = {
            "job_description_file": "AI Engineer - Job Description.pdf",
            "job_title": "AI Engineer - Machine Learning",
            "processing_timestamp": datetime.now().isoformat(),
            "total_candidates": 7,
            "processing_time": "1.8 minutes",
            "processing_errors": [],
            "candidates": [
                {
                    "rank": 1,
                    "candidate_name": "lateset_shine_res",
                    "overall_score": 44.1,
                    "score_breakdown": {
                        "core_skills": 27.6,
                        "market_relevance": 17.7,
                        "leadership_impact": 36.0,
                        "resume_quality": 71.9,
                        "certifications": 25,
                        "differentiation": 100
                    },
                    "market_positioning": {
                        "tier": "Developing",
                        "market_readiness": "Nearly Ready",
                        "recommendation": "Consider for junior roles or internship programs"
                    },
                    "strengths": [
                        "Strong AI/ML expertise (42.1%)",
                        "Demonstrated leadership experience (3 indicators)",
                        "Good technical depth and terminology"
                    ],
                    "competitive_advantages": [
                        "Unique skills that differentiate from other candidates",
                        "Strong alignment with current market trends"
                    ],
                    "experience_years": "2-3",
                    "education": "Master's in Computer Science"
                },
                {
                    "rank": 2,
                    "candidate_name": "fullstack",
                    "overall_score": 40.0,
                    "score_breakdown": {
                        "core_skills": 27.6,
                        "market_relevance": 24.1,
                        "leadership_impact": 0.0,
                        "resume_quality": 71.7,
                        "certifications": 25,
                        "differentiation": 100
                    },
                    "market_positioning": {
                        "tier": "Developing",
                        "market_readiness": "Market Ready",
                        "recommendation": "Consider for junior roles or internship programs"
                    },
                    "strengths": [
                        "Strong Cloud/DevOps expertise (38.5%)",
                        "Good technical depth and terminology",
                        "Full-stack development experience"
                    ],
                    "competitive_advantages": [
                        "Strong alignment with current market trends",
                        "Versatile technology stack"
                    ],
                    "experience_years": "1-2",
                    "education": "Bachelor's in Software Engineering"
                },
                {
                    "rank": 3,
                    "candidate_name": "java_resume",
                    "overall_score": 40.0,
                    "score_breakdown": {
                        "core_skills": 28.7,
                        "market_relevance": 20.3,
                        "leadership_impact": 24.0,
                        "resume_quality": 67.6,
                        "certifications": 0,
                        "differentiation": 100
                    },
                    "market_positioning": {
                        "tier": "Developing",
                        "market_readiness": "Nearly Ready",
                        "recommendation": "Consider for junior roles or internship programs"
                    },
                    "strengths": [
                        "Strong Cloud/DevOps expertise (38.5%)",
                        "Strong action-oriented language",
                        "Java programming expertise"
                    ],
                    "competitive_advantages": [
                        "Strong alignment with current market trends",
                        "Enterprise development background"
                    ],
                    "experience_years": "2+",
                    "education": "Bachelor's in Computer Science"
                }
            ],
            "statistics": {
                "score_distribution": {
                    "excellent_80_plus": 0,
                    "good_60_to_79": 0,
                    "fair_40_to_59": 3,
                    "poor_below_40": 4
                },
                "average_score": 37.2,
                "median_score": 36.2
            },
            "market_gaps": [
                "Generative AI applications",
                "Vector databases", 
                "Time series analysis"
            ]
        }
        return render_template('results.html', results=sample_data)
    except Exception as e:
        print(f"Results page error: {e}")
        # Return empty results page
        return render_template('results.html', results=None)

@app.route('/api/sample-analysis')
def sample_analysis():
    """Return sample analysis data for demo"""
    sample_data = {
        "job_description_file": "AI Engineer - Job Description.pdf",
        "job_title": "AI Engineer - Machine Learning",
        "processing_timestamp": datetime.now().isoformat(),
        "total_candidates": 7,
        "processing_time": "1.8 minutes",
        "processing_errors": [],
        "candidates": [
            {
                "rank": 1,
                "candidate_name": "lateset_shine_res",
                "overall_score": 44.1,
                "score_breakdown": {
                    "core_skills": 27.6,
                    "market_relevance": 17.7,
                    "leadership_impact": 36.0,
                    "resume_quality": 71.9,
                    "certifications": 25,
                    "differentiation": 100
                },
                "market_positioning": {
                    "tier": "Developing",
                    "market_readiness": "Nearly Ready",
                    "recommendation": "Consider for junior roles or internship programs"
                },
                "strengths": [
                    "Strong AI/ML expertise (42.1%)",
                    "Demonstrated leadership experience (3 indicators)",
                    "Good technical depth and terminology",
                    "Strong action-oriented language"
                ],
                "competitive_advantages": [
                    "Unique skills that differentiate from other candidates",
                    "Strong alignment with current market trends"
                ]
            },
            {
                "rank": 2,
                "candidate_name": "fullstack",
                "overall_score": 40.0,
                "score_breakdown": {
                    "core_skills": 27.6,
                    "market_relevance": 24.1,
                    "leadership_impact": 0.0,
                    "resume_quality": 71.7,
                    "certifications": 25,
                    "differentiation": 100
                },
                "market_positioning": {
                    "tier": "Developing",
                    "market_readiness": "Market Ready",
                    "recommendation": "Consider for junior roles or internship programs"
                },
                "strengths": [
                    "Strong Cloud/DevOps expertise (38.5%)",
                    "Good technical depth and terminology",
                    "Full-stack development experience"
                ],
                "competitive_advantages": [
                    "Strong alignment with current market trends",
                    "Versatile technology stack"
                ]
            },
            {
                "rank": 3,
                "candidate_name": "java_resume",
                "overall_score": 40.0,
                "score_breakdown": {
                    "core_skills": 28.7,
                    "market_relevance": 20.3,
                    "leadership_impact": 24.0,
                    "resume_quality": 67.6,
                    "certifications": 0,
                    "differentiation": 100
                },
                "market_positioning": {
                    "tier": "Developing",
                    "market_readiness": "Nearly Ready",
                    "recommendation": "Consider for junior roles or internship programs"
                },
                "strengths": [
                    "Strong Cloud/DevOps expertise (38.5%)",
                    "Strong action-oriented language",
                    "Java programming expertise"
                ],
                "competitive_advantages": [
                    "Strong alignment with current market trends",
                    "Enterprise development background"
                ]
            }
        ],
        "statistics": {
            "score_distribution": {
                "excellent_80_plus": 0,
                "good_60_to_79": 0,
                "fair_40_to_59": 3,
                "poor_below_40": 4
            },
            "average_score": 37.2,
            "median_score": 36.2
        },
        "market_gaps": [
            "Generative AI applications",
            "Vector databases", 
            "Time series analysis"
        ]
    }
    
    return jsonify(sample_data)

@app.route('/api/export')
def export_data():
    """Export analysis results in various formats"""
    format_type = request.args.get('format', 'pdf')
    
    if format_type == 'pdf':
        # Generate PDF report
        return jsonify({'message': 'PDF export functionality coming soon'}), 501
    elif format_type == 'xlsx':
        # Generate Excel file
        return jsonify({'message': 'Excel export functionality coming soon'}), 501
    elif format_type == 'json':
        # Return JSON data
        sample_data = sample_analysis().get_json()
        response = make_response(json.dumps(sample_data, indent=2))
        response.headers['Content-Type'] = 'application/json'
        response.headers['Content-Disposition'] = 'attachment; filename=analysis_results.json'
        return response
    else:
        return jsonify({'error': 'Unsupported format'}), 400

@app.route('/api/download-report')
def download_report():
    """Download comprehensive analysis report"""
    format_type = request.args.get('format', 'pdf')
    return jsonify({'message': f'{format_type.upper()} report download functionality coming soon'}), 501

@app.route('/api/export-candidate')
def export_candidate():
    """Export individual candidate profile"""
    candidate_name = request.args.get('name')
    format_type = request.args.get('format', 'pdf')
    
    if not candidate_name:
        return jsonify({'error': 'Candidate name required'}), 400
    
    return jsonify({'message': f'Individual candidate export for {candidate_name} coming soon'}), 501

@app.errorhandler(404)
def not_found_error(error):
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error. Please try again later.'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    app.run(debug=debug, host='0.0.0.0', port=port)
