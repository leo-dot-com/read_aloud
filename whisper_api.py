# whisper_api.py - Local Whisper Transcription API
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import whisper
from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
import logging
import json
from datetime import datetime
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global model variable
model = None

def load_model_once():
    """Load Whisper model only once when the API starts"""
    global model
    
    if model is not None:
        return
    
    logger.info("Loading Whisper tiny model...")
    try:
        # Using tiny model for faster inference on Railway
        model = whisper.load_model("tiny")
        logger.info("Whisper model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {str(e)}")
        raise

def analyze_reading_performance(transcription, original_text, audio_duration):
    """Analyze reading performance based on transcription"""
    analysis = {
        'word_accuracy': 0,
        'reading_pace_wpm': 0,
        'hesitation_count': 0,
        'repetition_count': 0,
        'self_correction_count': 0,
        'difficulty_word_accuracy': 0,
        'overall_score': 0,
        'dyslexia_likelihood': 'low',
        'audio_duration': audio_duration
    }
    
    try:
        # Convert to lowercase for comparison
        original_lower = original_text.lower()
        transcription_lower = transcription.lower()
        
        # Split into words
        original_words = [word.strip('.,!?;:') for word in original_lower.split()]
        transcribed_words = [word.strip('.,!?;:') for word in transcription_lower.split()]
        
        # Calculate word accuracy
        matching_words = set(original_words) & set(transcribed_words)
        analysis['word_accuracy'] = len(matching_words) / len(original_words) * 100 if original_words else 0
        
        # Calculate reading pace (words per minute)
        if audio_duration > 0:
            analysis['reading_pace_wpm'] = (len(transcribed_words) / audio_duration) * 60
        
        # Detect hesitations
        hesitation_patterns = ['um', 'uh', 'er', 'ah', 'hm', 'hmm']
        analysis['hesitation_count'] = sum(1 for word in transcribed_words if word in hesitation_patterns)
        
        # Detect repetitions
        analysis['repetition_count'] = count_repetitions(transcribed_words)
        
        # Detect self-corrections
        analysis['self_correction_count'] = count_self_corrections(transcription)
        
        # Calculate difficulty word accuracy
        difficulty_words = extract_difficulty_words(original_text)
        analysis['difficulty_word_accuracy'] = calculate_difficulty_word_accuracy(transcription, difficulty_words)
        
        # Calculate overall score
        analysis['overall_score'] = calculate_overall_score(analysis)
        
        # Determine dyslexia likelihood
        analysis['dyslexia_likelihood'] = determine_dyslexia_likelihood(analysis)
        
    except Exception as e:
        logger.error(f"Error in reading analysis: {str(e)}")
    
    return analysis

def count_repetitions(words):
    """Count consecutive word repetitions"""
    repetitions = 0
    for i in range(1, len(words)):
        if words[i] == words[i-1]:
            repetitions += 1
    return repetitions

def count_self_corrections(text):
    """Count self-correction patterns"""
    patterns = [
        r'\b(\w+)\s+(\1)\b',  # Immediate repetition
        r'\b(\w+)\s+no\s+\1\b',  # "word no word" pattern
        r'\b(\w+)\s+I\s+mean\s+\w+\b',  # "I mean" pattern
        r'\b(\w+)\s+sorry\s+\w+\b',  # "sorry" correction
    ]
    
    import re
    count = 0
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        count += len(matches)
    return count

def extract_difficulty_words(text):
    """Extract potentially difficult words from text"""
    # Simple heuristic: words with 7+ letters or with complex patterns
    words = text.split()
    difficulty_words = []
    
    for word in words:
        clean_word = word.strip('.,!?;:').lower()
        if len(clean_word) >= 7 or has_complex_pattern(clean_word):
            difficulty_words.append(clean_word)
    
    return list(set(difficulty_words))  # Remove duplicates

def has_complex_pattern(word):
    """Check if word has complex phonetic patterns"""
    complex_patterns = [
        'ough', 'tion', 'sion', 'cious', 'tious', 'cial', 'tial',
        'phy', 'ology', 'graph', 'spect', 'struct'
    ]
    return any(pattern in word for pattern in complex_patterns)

def calculate_difficulty_word_accuracy(transcription, difficulty_words):
    """Calculate accuracy on difficult words"""
    if not difficulty_words:
        return 100
    
    transcription_lower = transcription.lower()
    found_count = 0
    
    for word in difficulty_words:
        if word in transcription_lower:
            found_count += 1
    
    return (found_count / len(difficulty_words)) * 100

def calculate_overall_score(analysis):
    """Calculate overall reading score"""
    weights = {
        'word_accuracy': 0.35,
        'reading_pace_wpm': 0.20,
        'hesitation_count': 0.15,
        'repetition_count': 0.15,
        'difficulty_word_accuracy': 0.15
    }
    
    # Normalize values
    word_accuracy = analysis['word_accuracy']
    
    # Normalize reading pace (assume 150 WPM is excellent, 50 WPM is poor)
    reading_pace = min(max(analysis['reading_pace_wpm'], 50), 150)
    pace_score = ((reading_pace - 50) / 100) * 100
    
    # Normalize hesitation count (more hesitations = lower score)
    hesitation_score = max(0, 100 - (analysis['hesitation_count'] * 10))
    
    # Normalize repetition count (more repetitions = lower score)
    repetition_score = max(0, 100 - (analysis['repetition_count'] * 15))
    
    difficulty_accuracy = analysis['difficulty_word_accuracy']
    
    weighted_score = (
        word_accuracy * weights['word_accuracy'] +
        pace_score * weights['reading_pace_wpm'] +
        hesitation_score * weights['hesitation_count'] +
        repetition_score * weights['repetition_count'] +
        difficulty_accuracy * weights['difficulty_word_accuracy']
    )
    
    return min(100, max(0, weighted_score))

def determine_dyslexia_likelihood(analysis):
    """Determine dyslexia likelihood based on analysis"""
    score = analysis['overall_score']
    
    if score >= 80:
        return 'low'
    elif score >= 60:
        return 'moderate'
    elif score >= 40:
        return 'high'
    else:
        return 'very_high'

def convert_audio_to_wav(input_path, output_path):
    """Convert any audio format to WAV using ffmpeg"""
    try:
        cmd = [
            'ffmpeg', '-i', input_path,
            '-acodec', 'pcm_s16le',
            '-ac', '1',
            '-ar', '16000',
            '-y',  # Overwrite output file
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"FFmpeg conversion failed: {result.stderr}")
            raise Exception(f"Audio conversion failed: {result.stderr}")
            
        logger.info("Audio converted successfully to WAV")
        return True
        
    except Exception as e:
        logger.error(f"Error in audio conversion: {str(e)}")
        raise

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """Transcribe audio and analyze reading performance"""
    try:
        load_model_once()
        
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file uploaded"}), 400
        
        if 'original_text' not in request.form:
            return jsonify({"error": "No original text provided"}), 400
        
        audio_file = request.files['audio']
        original_text = request.form['original_text']
        
        if audio_file.filename == '':
            return jsonify({"error": "No audio file selected"}), 400
        
        # Save uploaded file temporarily
        input_ext = os.path.splitext(audio_file.filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=input_ext) as tmp_input:
            audio_file.save(tmp_input.name)
            input_path = tmp_input.name
        
        # Create output path for converted audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_output:
            output_path = tmp_output.name
        
        try:
            # Convert audio to WAV format for better compatibility
            logger.info("Converting audio to WAV format...")
            convert_audio_to_wav(input_path, output_path)
            
            # Transcribe audio using Whisper
            logger.info("Transcribing audio...")
            result = model.transcribe(output_path)
            transcription = result["text"].strip()
            
            # Get audio duration from Whisper result
            audio_duration = result.get('duration', 30)
            
            # Analyze reading performance
            analysis = analyze_reading_performance(transcription, original_text, audio_duration)
            
            logger.info(f"Transcription successful: {len(transcription)} characters")
            
            return jsonify({
                "success": True,
                "transcription": transcription,
                "analysis": analysis,
                "audio_duration": audio_duration
            })
            
        finally:
            # Clean up temporary files
            for temp_file in [input_path, output_path]:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                
    except Exception as e:
        logger.error(f"Error in transcribe_audio: {str(e)}")
        return jsonify({"error": str(e)}), 500
        
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        load_model_once()
        return jsonify({
            "status": "healthy", 
            "model_loaded": model is not None,
            "ready": True,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/')
def home():
    return jsonify({
        "message": "Whisper Transcription API", 
        "status": "running",
        "endpoints": ["/transcribe", "/health"]
    })

if __name__ == '__main__':
    logger.info("Starting Whisper Transcription API...")
    port = int(os.environ.get('PORT', 5000))
    
    # Pre-load model
    try:
        load_model_once()
    except Exception as e:
        logger.warning(f"Initial model load failed: {str(e)}")
    
    app.run(host='0.0.0.0', port=port, debug=False)
