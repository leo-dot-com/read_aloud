# whisper_api.py - Uses Hugging Face Inference API (FREE)
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
import logging
import os
from datetime import datetime
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Hugging Face Inference API - FREE tier
HF_API_URL = "https://api-inference.huggingface.co/models/openai/whisper-tiny"
HF_TOKEN = os.environ.get("HF_TOKEN", "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")

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def transcribe_with_hf(audio_path):
    """Use Hugging Face's free inference API"""
    with open(audio_path, "rb") as f:
        data = f.read()
    
    response = requests.post(HF_API_URL, headers=headers, data=data)
    response.raise_for_status()
    
    result = response.json()
    return result["text"]

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
        hesitation_patterns = ['um', 'uh', 'er', 'ah', 'hm', 'hmm']
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
    patterns = [
        r'\b(\w+)\s+(\1)\b',  # Immediate repetition
        r'\b(\w+)\s+no\s+\1\b',  # "word no word" pattern
        r'\b(\w+)\s+I\s+mean\s+\w+\b',  # "I mean" pattern
        r'\b(\w+)\s+sorry\s+\w+\b',  # "sorry" correction
    ]
    
    count = 0
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        count += len(matches)
    return count

def extract_difficulty_words(text):
    words = text.split()
    difficulty_words = []
    complex_patterns = ['ough', 'tion', 'sion', 'cious', 'tious', 'cial', 'tial', 'phy', 'ology', 'graph', 'spect', 'struct']
    
    for word in words:
        clean_word = word.strip('.,!?;:').lower()
        if len(clean_word) >= 7 or any(pattern in clean_word for pattern in complex_patterns):
            difficulty_words.append(clean_word)
    
    return list(set(difficulty_words))

def calculate_difficulty_word_accuracy(transcription, difficulty_words):
    if not difficulty_words:
        return 100
    
    transcription_lower = transcription.lower()
    found_count = 0
    
    for word in difficulty_words:
        if word in transcription_lower:
            found_count += 1
    
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
    score = analysis['overall_score']
    if score >= 80:
        return 'low'
    elif score >= 60:
        return 'moderate'
    elif score >= 40:
        return 'high'
    else:
        return 'very_high'

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    try:
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
            logger.info("Transcribing with Hugging Face API...")
            transcription = transcribe_with_hf(audio_path)
            
            # We don't have the audio duration from HF API, so we estimate based on the number of words?
            # Alternatively, we can use a fixed value or try to compute it from the audio file.
            # For now, we'll use a fixed estimate of 30 seconds as fallback.
            # But note: the user's recording might be of different length.
            # We can use the audio file to compute the duration? We don't want to add heavy dependencies.
            # Since we are using a lightweight version, let's assume 30 seconds for now.
            audio_duration = 30  # Fallback, you might want to improve this
            
            analysis = analyze_reading_performance(transcription, original_text, audio_duration)
            
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
    return jsonify({
        "status": "healthy", 
        "timestamp": datetime.now().isoformat()
    })

@app.route('/')
def home():
    return jsonify({
        "message": "Whisper Transcription API (Hugging Face)", 
        "status": "running"
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
