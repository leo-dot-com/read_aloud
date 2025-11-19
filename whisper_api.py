# whisper_api.py - OPTIMIZED with faster-whisper
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from faster_whisper import WhisperModel
from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global model variable
model = None

def load_model_once():
    """Load optimized Whisper model"""
    global model
    
    if model is not None:
        return
    
    logger.info("Loading optimized Whisper tiny model...")
    try:
        # Use faster-whisper with int8 quantization - MUCH smaller
        model = WhisperModel(
            "tiny", 
            device="cpu",  # Use CPU to reduce size (no CUDA dependencies)
            compute_type="int8",  # Quantized for smaller size
            download_root="/tmp/whisper-models"  # Cache in tmp
        )
        logger.info("Optimized Whisper model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {str(e)}")
        raise

def analyze_reading_performance(transcription, original_text, audio_duration):
    """Analyze reading performance (same as before but optimized)"""
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
        original_lower = original_text.lower()
        transcription_lower = transcription.lower()
        
        original_words = [word.strip('.,!?;:') for word in original_lower.split()]
        transcribed_words = [word.strip('.,!?;:') for word in transcription_lower.split()]
        
        # Word accuracy
        matching_words = set(original_words) & set(transcribed_words)
        analysis['word_accuracy'] = len(matching_words) / len(original_words) * 100 if original_words else 0
        
        # Reading pace
        if audio_duration > 0:
            analysis['reading_pace_wpm'] = (len(transcribed_words) / audio_duration) * 60
        
        # Hesitations
        hesitation_patterns = ['um', 'uh', 'er', 'ah', 'hm']
        analysis['hesitation_count'] = sum(1 for word in transcribed_words if word in hesitation_patterns)
        
        # Repetitions
        analysis['repetition_count'] = count_repetitions(transcribed_words)
        
        # Self-corrections
        analysis['self_correction_count'] = count_self_corrections(transcription)
        
        # Difficulty words
        difficulty_words = extract_difficulty_words(original_text)
        analysis['difficulty_word_accuracy'] = calculate_difficulty_word_accuracy(transcription, difficulty_words)
        
        # Overall score
        analysis['overall_score'] = calculate_overall_score(analysis)
        
        # Dyslexia likelihood
        analysis['dyslexia_likelihood'] = determine_dyslexia_likelihood(analysis)
        
    except Exception as e:
        logger.error(f"Error in reading analysis: {str(e)}")
    
    return analysis

def count_repetitions(words):
    repetitions = 0
    for i in range(1, len(words)):
        if words[i] == words[i-1]:
            repetitions += 1
    return repetitions

def count_self_corrections(text):
    import re
    patterns = [
        r'\b(\w+)\s+(\1)\b',
        r'\b(\w+)\s+no\s+\1\b',
        r'\b(\w+)\s+I\s+mean\s+\w+\b',
    ]
    count = 0
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        count += len(matches)
    return count

def extract_difficulty_words(text):
    words = text.split()
    difficulty_words = []
    complex_patterns = ['ough', 'tion', 'sion', 'cious', 'tious']
    
    for word in words:
        clean_word = word.strip('.,!?;:').lower()
        if len(clean_word) >= 7 or any(pattern in clean_word for pattern in complex_patterns):
            difficulty_words.append(clean_word)
    
    return list(set(difficulty_words))

def calculate_difficulty_word_accuracy(transcription, difficulty_words):
    if not difficulty_words:
        return 100
    
    transcription_lower = transcription.lower()
    found_count = sum(1 for word in difficulty_words if word in transcription_lower)
    return (found_count / len(difficulty_words)) * 100

def calculate_overall_score(analysis):
    weights = {
        'word_accuracy': 0.35,
        'reading_pace_wpm': 0.20,
        'hesitation_count': 0.15,
        'repetition_count': 0.15,
        'difficulty_word_accuracy': 0.15
    }
    
    word_accuracy = analysis['word_accuracy']
    reading_pace = min(max(analysis['reading_pace_wpm'], 50), 150)
    pace_score = ((reading_pace - 50) / 100) * 100
    hesitation_score = max(0, 100 - (analysis['hesitation_count'] * 10))
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
    score = analysis['overall_score']
    if score >= 80: return 'low'
    elif score >= 60: return 'moderate'
    elif score >= 40: return 'high'
    else: return 'very_high'

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
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
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            audio_file.save(tmp_file.name)
            audio_path = tmp_file.name
        
        try:
            logger.info("Transcribing audio with optimized model...")
            segments, info = model.transcribe(audio_path, beam_size=5)
            
            transcription = " ".join(segment.text for segment in segments)
            audio_duration = info.duration
            
            analysis = analyze_reading_performance(transcription, original_text, audio_duration)
            
            logger.info(f"Transcription successful: {len(transcription)} characters")
            
            return jsonify({
                "success": True,
                "transcription": transcription,
                "analysis": analysis,
                "audio_duration": audio_duration
            })
            
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
                
    except Exception as e:
        logger.error(f"Error in transcribe_audio: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
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
        "message": "Optimized Whisper Transcription API", 
        "status": "running",
        "size": "lightweight"
    })

if __name__ == '__main__':
    logger.info("Starting OPTIMIZED Whisper API...")
    port = int(os.environ.get('PORT', 5000))
    
    try:
        load_model_once()
    except Exception as e:
        logger.warning(f"Initial model load failed: {str(e)}")
    
    app.run(host='0.0.0.0', port=port, debug=False)
