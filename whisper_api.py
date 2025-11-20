# whisper_local.py - Local Whisper implementation
import os
import torch
import torchaudio
from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
import logging
import librosa
import soundfile as sf
from datetime import datetime
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Try to use transformers first, fall back to openai-whisper
try:
    from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
    TRANSFORMERS_AVAILABLE = True
    logger.info("Using transformers implementation")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.info("Transformers not available, trying openai-whisper")
    try:
        import whisper
        WHISPER_AVAILABLE = True
    except ImportError:
        WHISPER_AVAILABLE = False
        logger.error("Neither transformers nor whisper available")

# Model configuration - using smaller models for Railway compatibility
MODEL_CONFIGS = {
    "tiny": {"params": "39M", "recommended": True},
    "base": {"params": "74M", "recommended": True},
    "small": {"params": "244M", "recommended": False},
    "medium": {"params": "769M", "recommended": False},
}

# Use tiny model for Railway to avoid memory issues
SELECTED_MODEL = "small"
logger.info(f"Using model: {SELECTED_MODEL}")

def load_whisper_model():
    """Load Whisper model based on available libraries"""
    try:
        if TRANSFORMERS_AVAILABLE:
            logger.info("Loading model with transformers...")
            model_id = f"openai/whisper-{SELECTED_MODEL}"
            
            # Use pipeline for simplicity
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device="cuda:0" if torch.cuda.is_available() else "cpu",
            )
            return {"type": "transformers", "model": pipe}
        
        elif WHISPER_AVAILABLE:
            logger.info("Loading model with openai-whisper...")
            model = whisper.load_model(SELECTED_MODEL)
            return {"type": "whisper", "model": model}
        else:
            raise Exception("No Whisper implementation available")
            
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise

# Global model variable
whisper_model = None

def transcribe_local(audio_path):
    """Transcribe audio using local Whisper model"""
    global whisper_model
    
    if whisper_model is None:
        whisper_model = load_whisper_model()
    
    # Get audio duration
    try:
        audio_duration = librosa.get_duration(filename=audio_path)
        logger.info(f"Audio duration: {audio_duration:.2f} seconds")
    except Exception as e:
        logger.warning(f"Could not get audio duration: {e}")
        audio_duration = 30
    
    try:
        if whisper_model["type"] == "transformers":
            # Using transformers pipeline
            result = whisper_model["model"](
                audio_path,
                generate_kwargs={"language": "english", "task": "transcribe"}
            )
            transcription = result["text"]
            
        else:  # Using openai-whisper
            result = whisper_model["model"].transcribe(audio_path, language="english")
            transcription = result["text"]
        
        logger.info(f"Transcription successful: {len(transcription)} characters")
        return transcription, audio_duration
        
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise

def convert_audio_format(audio_path):
    """Convert audio to 16kHz WAV format for better compatibility"""
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Create temporary file for converted audio
        converted_path = audio_path + "_converted.wav"
        sf.write(converted_path, y, sr, format='WAV', subtype='PCM_16')
        
        logger.info("Audio conversion successful")
        return converted_path
    except Exception as e:
        logger.warning(f"Audio conversion failed: {e}, using original file")
        return audio_path

# Keep your existing analysis functions (they don't need to change)
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
        
        # Clean and prepare word lists
        original_words = [re.sub(r'[^\w\s]', '', word) for word in original_lower.split()]
        transcribed_words = [re.sub(r'[^\w\s]', '', word) for word in transcription_lower.split()]
        
        # Remove empty strings
        original_words = [word for word in original_words if word]
        transcribed_words = [word for word in transcribed_words if word]
        
        # Word accuracy using sequence matching
        analysis['word_accuracy'] = calculate_sequence_accuracy(original_words, transcribed_words)
        
        # Reading pace
        if audio_duration > 0:
            analysis['reading_pace_wpm'] = (len(transcribed_words) / audio_duration) * 60
        
        # Hesitations
        hesitation_patterns = ['um', 'uh', 'er', 'ah', 'hm', 'hmm', 'eh', 'mm']
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
        # Set default values
        analysis['word_accuracy'] = 50
        analysis['reading_pace_wpm'] = 100
        analysis['overall_score'] = 50
    
    return analysis

def calculate_sequence_accuracy(original, transcribed):
    """Calculate accuracy based on word sequence matching"""
    if not original:
        return 0
    
    matches = 0
    min_len = min(len(original), len(transcribed))
    
    for i in range(min_len):
        if original[i] == transcribed[i]:
            matches += 1
    
    return (matches / len(original)) * 100

def count_repetitions(words):
    repetitions = 0
    for i in range(1, len(words)):
        if words[i] == words[i-1]:
            repetitions += 1
    return repetitions

def count_self_corrections(text):
    patterns = [
        r'\b(\w+)\s+(\1)\b',
        r'\b(\w+)\s+no\s+\1\b',
        r'\b(\w+)\s+I\s+mean\s+\w+\b',
        r'\b(\w+)\s+sorry\s+\w+\b',
    ]
    
    count = 0
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        count += len(matches)
    return count

def extract_difficulty_words(text):
    words = text.split()
    difficulty_words = []
    
    for word in words:
        clean_word = re.sub(r'[^\w\s]', '', word).lower()
        if len(clean_word) >= 7:
            difficulty_words.append(clean_word)
    
    return list(set(difficulty_words))[:5]

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
    
    # Normalize reading pace
    reading_pace = min(max(analysis['reading_pace_wpm'], 50), 150)
    pace_score = ((reading_pace - 50) / 100) * 100
    
    # Normalize hesitation count
    hesitation_score = max(0, 100 - (analysis['hesitation_count'] * 10))
    
    # Normalize repetition count
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
            logger.info("Starting local transcription...")
            
            # Convert audio if needed
            converted_path = convert_audio_format(audio_path)
            
            # Transcribe using local model
            transcription, audio_duration = transcribe_local(converted_path)
            
            logger.info("Transcription completed, starting analysis...")
            analysis = analyze_reading_performance(transcription, original_text, audio_duration)
            
            return jsonify({
                "success": True,
                "transcription": transcription,
                "analysis": analysis,
                "audio_duration": audio_duration,
                "model_used": f"local-whisper-{SELECTED_MODEL}"
            })
            
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            return jsonify({"error": f"Transcription failed: {str(e)}"}), 500
            
        finally:
            # Clean up temporary files
            if os.path.exists(audio_path):
                os.unlink(audio_path)
            converted_path = audio_path + "_converted.wav"
            if os.path.exists(converted_path):
                os.unlink(converted_path)
                
    except Exception as e:
        logger.error(f"Error in transcribe_audio: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    model_status = "loaded" if whisper_model is not None else "not loaded"
    return jsonify({
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "model_status": model_status,
        "model_type": SELECTED_MODEL
    })

@app.route('/')
def home():
    return jsonify({
        "message": "Local Whisper Transcription API", 
        "status": "running",
        "model": SELECTED_MODEL
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
