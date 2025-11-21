# whisper_api.py - Local Whisper Transcription API with FFmpeg
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import whisper
from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
import logging
import json
from datetime import datetime
import subprocess
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global model variable
model = None

def check_ffmpeg():
    """Check if FFmpeg is available"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("FFmpeg is available")
            return True
        else:
            logger.error("FFmpeg check failed")
            return False
    except Exception as e:
        logger.error(f"FFmpeg not found: {e}")
        return False

def load_model_once():
    """Load Whisper model only once when the API starts"""
    global model
    
    if model is not None:
        return
    
    logger.info("Loading Whisper tiny model...")
    try:
        # Using tiny model for faster inference
        model = whisper.load_model("tiny")
        logger.info("Whisper model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {str(e)}")
        raise

# Update the convert_audio_to_wav function to be more robust
def convert_audio_to_wav(input_path, output_path):
    """Convert any audio format to WAV using ffmpeg with better error handling"""
    try:
        # Check if input file exists and has content
        if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
            logger.error(f"Input file is empty or doesn't exist: {input_path}")
            return False
        
        cmd = [
            'ffmpeg', '-i', input_path,
            '-acodec', 'pcm_s16le',
            '-ac', '1',
            '-ar', '16000',
            '-y',  # Overwrite output file
            output_path
        ]
        
        logger.info(f"Converting audio: {input_path} -> {output_path}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg conversion failed: {result.stderr}")
            return False
        
        # Check if output file was created and has content
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            logger.error("FFmpeg conversion resulted in empty output file")
            return False
            
        logger.info("Audio converted successfully to WAV")
        return True
        
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg conversion timed out")
        return False
    except Exception as e:
        logger.error(f"Error in audio conversion: {str(e)}")
        return False

def analyze_reading_performance(transcription, original_text, audio_duration, 
                              pauses=None, hesitations=None, fluency_metrics=None, text_comparison=None):
    """Enhanced reading performance analysis"""
    
    # Initialize with default values
    pauses = pauses or []
    hesitations = hesitations or []
    fluency_metrics = fluency_metrics or {}
    text_comparison = text_comparison or {}
    
    analysis = {
        'word_accuracy': text_comparison.get('word_accuracy', 0),
        'reading_pace_wpm': fluency_metrics.get('words_per_minute', 0),
        'hesitation_count': len(hesitations),
        'repetition_count': 0,  # You can add repetition detection
        'self_correction_count': 0,
        'difficulty_word_accuracy': 0,
        'pause_count': len(pauses),
        'total_pause_duration': sum(pause['gap_duration'] for pause in pauses),
        'fluency_consistency': fluency_metrics.get('speaking_rate_consistency', 100),
        'error_count': text_comparison.get('total_errors', 0),
        'overall_score': 0,
        'dyslexia_likelihood': 'low',
        'audio_duration': audio_duration
    }
    
    # Calculate overall score with enhanced factors
    analysis['overall_score'] = calculate_enhanced_score(analysis, text_comparison)
    analysis['dyslexia_likelihood'] = determine_dyslexia_likelihood(analysis)
    
    return analysis

def calculate_enhanced_score(analysis, text_comparison):
    """Calculate enhanced overall score considering dyslexia indicators"""
    weights = {
        'word_accuracy': 0.25,
        'fluency_consistency': 0.20,
        'hesitation_count': 0.15,
        'pause_count': 0.10,
        'error_count': 0.20,
        'reading_pace_wpm': 0.10
    }
    
    # Normalize values
    word_accuracy = analysis['word_accuracy']
    fluency_consistency = analysis['fluency_consistency']
    
    # Normalize hesitation count (more hesitations = lower score)
    hesitation_score = max(0, 100 - (analysis['hesitation_count'] * 8))
    
    # Normalize pause count (more pauses = lower score)
    pause_score = max(0, 100 - (analysis['pause_count'] * 5))
    
    # Normalize error count
    max_expected_errors = text_comparison.get('original_word_count', 50) * 0.3  # Allow 30% errors
    error_score = max(0, 100 - (analysis['error_count'] / max_expected_errors * 100)) if max_expected_errors > 0 else 100
    
    # Normalize reading pace (optimal range 100-150 WPM for children)
    reading_pace = min(max(analysis['reading_pace_wpm'], 50), 200)
    if reading_pace < 80:
        pace_score = (reading_pace / 80) * 100
    elif reading_pace > 150:
        pace_score = max(0, 100 - ((reading_pace - 150) / 50 * 100))
    else:
        pace_score = 100
    
    weighted_score = (
        word_accuracy * weights['word_accuracy'] +
        fluency_consistency * weights['fluency_consistency'] +
        hesitation_score * weights['hesitation_count'] +
        pause_score * weights['pause_count'] +
        error_score * weights['error_count'] +
        pace_score * weights['reading_pace_wpm']
    )
    
    return min(100, max(0, weighted_score))

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

# Add this function to handle the new transcription method
def transcribe_with_timestamps(audio_file_path):
    """Transcribe audio with word-level timestamps with fallback"""
    try:
        # Try to get word-level timestamps
        result = model.transcribe(audio_file_path, word_timestamps=True)
        
        # Extract words with timestamps
        words_with_timestamps = []
        for segment in result.get("segments", []):
            for word_info in segment.get("words", []):
                words_with_timestamps.append({
                    'word': word_info['word'].strip(),
                    'start': word_info['start'],
                    'end': word_info['end'],
                    'confidence': word_info.get('probability', 0)
                })
        
        return result["text"].strip(), words_with_timestamps, result.get('duration', 0)
    
    except Exception as e:
        logger.warning(f"Word-level timestamp failed, falling back to basic transcription: {str(e)}")
        # Fallback to basic transcription
        result = model.transcribe(audio_file_path)
        return result["text"].strip(), [], result.get('duration', 0)

def detect_pauses_and_hesitations(words_with_timestamps, pause_threshold=0.8, hesitation_threshold=2.0):
    """Detect pauses and hesitations between words"""
    pauses = []
    hesitations = []
    
    if len(words_with_timestamps) < 2:
        return pauses, hesitations
    
    for i in range(1, len(words_with_timestamps)):
        current_word = words_with_timestamps[i]
        previous_word = words_with_timestamps[i-1]
        
        gap = current_word['start'] - previous_word['end']
        
        # Detect pauses (unusually long gaps between words)
        if gap > pause_threshold:
            pauses.append({
                'position': i,
                'gap_duration': gap,
                'before_word': previous_word['word'],
                'after_word': current_word['word']
            })
        
        # Detect hesitations (very long gaps)
        if gap > hesitation_threshold:
            hesitations.append({
                'position': i,
                'gap_duration': gap,
                'before_word': previous_word['word'],
                'after_word': current_word['word']
            })
    
    return pauses, hesitations

def analyze_reading_fluency(words_with_timestamps, audio_duration):
    """Analyze reading fluency metrics"""
    if not words_with_timestamps or len(words_with_timestamps) < 2:
        return {
            'words_per_minute': 0,
            'avg_pause_duration': 0,
            'pause_frequency': 0,
            'speaking_rate_consistency': 100
        }
    
    # Calculate words per minute
    total_words = len(words_with_timestamps)
    words_per_minute = (total_words / audio_duration) * 60 if audio_duration > 0 else 0
    
    # Calculate pause statistics
    pause_durations = []
    for i in range(1, len(words_with_timestamps)):
        gap = words_with_timestamps[i]['start'] - words_with_timestamps[i-1]['end']
        if gap > 0.1:  # Only count gaps > 100ms as pauses
            pause_durations.append(gap)
    
    avg_pause_duration = sum(pause_durations) / len(pause_durations) if pause_durations else 0
    pause_frequency = len(pause_durations) / total_words if total_words > 0 else 0
    
    # Calculate speaking rate consistency (variation in word durations)
    word_durations = [word['end'] - word['start'] for word in words_with_timestamps]
    avg_word_duration = sum(word_durations) / len(word_durations) if word_durations else 0
    
    # Coefficient of variation for speaking rate consistency
    if avg_word_duration > 0:
        variance = sum((duration - avg_word_duration) ** 2 for duration in word_durations) / len(word_durations)
        std_dev = variance ** 0.5
        consistency = max(0, 100 - (std_dev / avg_word_duration * 100))
    else:
        consistency = 100
    
    return {
        'words_per_minute': round(words_per_minute, 2),
        'avg_pause_duration': round(avg_pause_duration, 2),
        'pause_frequency': round(pause_frequency, 4),
        'speaking_rate_consistency': round(consistency, 2)
    }

def advanced_text_comparison(transcription, original_text, words_with_timestamps):
    """Compare transcription to original text with detailed analysis"""
    
    # Normalize texts for comparison
    original_lower = original_text.lower()
    transcription_lower = transcription.lower()
    
    # Split into words, preserving some punctuation for context
    original_words = [word.strip('.,!?;:"').lower() for word in original_text.split()]
    transcribed_words = [word.strip('.,!?;:"').lower() for word in transcription.split()]
    
    # Initialize error tracking
    errors = {
        'omissions': [],
        'additions': [],
        'substitutions': [],
        'inversions': [],
        'line_jumps': 0,
        'phonetic_errors': []
    }
    
    # Simple sequence alignment for error detection
    i, j = 0, 0
    max_i, max_j = len(original_words), len(transcribed_words)
    
    while i < max_i and j < max_j:
        orig_word = original_words[i]
        trans_word = transcribed_words[j]
        
        # Exact match
        if orig_word == trans_word:
            i += 1
            j += 1
            continue
        
        # Substitution (different word in same position)
        if j < max_j and i < max_i:
            errors['substitutions'].append({
                'original': orig_word,
                'spoken': trans_word,
                'position': i
            })
            i += 1
            j += 1
            continue
        
        # Omission (missing word)
        if j >= max_j and i < max_i:
            errors['omissions'].append({
                'word': orig_word,
                'position': i
            })
            i += 1
            continue
        
        # Addition (extra word)
        if i >= max_i and j < max_j:
            errors['additions'].append({
                'word': trans_word,
                'position': j
            })
            j += 1
            continue
    
    # Detect potential inversions (check adjacent word swaps)
    for pos in range(min(len(original_words), len(transcribed_words)) - 1):
        if (pos + 1 < len(transcribed_words) and 
            original_words[pos] == transcribed_words[pos + 1] and
            original_words[pos + 1] == transcribed_words[pos]):
            errors['inversions'].append({
                'word1': original_words[pos],
                'word2': original_words[pos + 1],
                'position': pos
            })
    
    # Calculate accuracy metrics
    total_original_words = len(original_words)
    correct_words = total_original_words - len(errors['omissions']) - len(errors['substitutions'])
    word_accuracy = (correct_words / total_original_words * 100) if total_original_words > 0 else 0
    
    return {
        'word_accuracy': round(word_accuracy, 2),
        'total_errors': len(errors['omissions']) + len(errors['additions']) + 
                        len(errors['substitutions']) + len(errors['inversions']),
        'error_breakdown': errors,
        'original_word_count': total_original_words,
        'transcribed_word_count': len(transcribed_words)
    }

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

# Update the transcribe_audio function in whisper_api.py

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """Transcribe audio and analyze reading performance"""
    try:
        load_model_once()
        
        if 'audio' not in request.files:
            return jsonify({"success": False, "error": "No audio file uploaded"}), 400
        
        if 'original_text' not in request.form:
            return jsonify({"success": False, "error": "No original text provided"}), 400
        
        audio_file = request.files['audio']
        original_text = request.form['original_text']
        
        if audio_file.filename == '':
            return jsonify({"success": False, "error": "No audio file selected"}), 400
        
        # Validate file size
        audio_file.seek(0, 2)  # Seek to end to get file size
        file_size = audio_file.tell()
        audio_file.seek(0)  # Reset seek position
        
        if file_size > 50 * 1024 * 1024:  # 50MB limit
            return jsonify({"success": False, "error": "Audio file too large. Maximum size is 50MB."}), 400
        
        # Save uploaded file temporarily
        input_ext = os.path.splitext(audio_file.filename)[1].lower()
        allowed_extensions = ['.wav', '.mp3', '.m4a', '.webm', '.ogg', '.flac', '.aac']
        
        if input_ext not in allowed_extensions:
            return jsonify({"success": False, "error": f"Unsupported audio format. Allowed: {', '.join(allowed_extensions)}"}), 400
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=input_ext) as tmp_input:
            audio_file.save(tmp_input.name)
            input_path = tmp_input.name
        
        # Create output path for converted audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_output:
            output_path = tmp_output.name
        
        try:
            # Convert audio to WAV format for better compatibility
            logger.info("Converting audio to WAV format...")
            conversion_success = convert_audio_to_wav(input_path, output_path)
            
            # Use converted file if successful, otherwise try original
            audio_file_path = output_path if conversion_success else input_path
            
            # Check if converted file exists and has content
            if not os.path.exists(audio_file_path) or os.path.getsize(audio_file_path) == 0:
                return jsonify({"success": False, "error": "Audio conversion failed or resulted in empty file"}), 400
            
            # Transcribe audio using Whisper with timestamps
            logger.info("Transcribing audio with timestamps...")
            transcription, words_with_timestamps, audio_duration = transcribe_with_timestamps(audio_file_path)
            
            # Analyze pauses and hesitations
            pauses, hesitations = detect_pauses_and_hesitations(words_with_timestamps)
            
            # Analyze reading fluency
            fluency_analysis = analyze_reading_fluency(words_with_timestamps, audio_duration)
            
            # Advanced text comparison
            text_comparison = advanced_text_comparison(transcription, original_text, words_with_timestamps)
            
            # Enhanced reading performance analysis
            analysis = analyze_reading_performance(
                transcription, 
                original_text, 
                audio_duration,
                pauses=pauses,
                hesitations=hesitations,
                fluency_metrics=fluency_analysis,
                text_comparison=text_comparison
            )
            
            logger.info(f"Enhanced transcription successful: {len(transcription)} characters")
            
            response_data = {
                "success": True,
                "transcription": transcription,
                "analysis": analysis,
                "audio_duration": audio_duration,
                "detailed_analysis": {
                    "pauses": pauses,
                    "hesitations": hesitations,
                    "fluency_metrics": fluency_analysis,
                    "text_comparison": text_comparison,
                    "words_with_timestamps": words_with_timestamps
                }
            }
            
            return jsonify(response_data)
            
        except Exception as e:
            logger.error(f"Error in transcription process: {str(e)}")
            return jsonify({"success": False, "error": f"Transcription failed: {str(e)}"}), 500
            
        finally:
            # Clean up temporary files
            for temp_file in [input_path, output_path]:
                if os.path.exists(temp_file):
                    try:
                        os.unlink(temp_file)
                    except Exception as e:
                        logger.warning(f"Failed to delete temp file {temp_file}: {str(e)}")
                
    except Exception as e:
        logger.error(f"Error in transcribe_audio endpoint: {str(e)}")
        return jsonify({"success": False, "error": f"Server error: {str(e)}"}), 500




        
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        load_model_once()
        ffmpeg_available = check_ffmpeg()
        return jsonify({
            "status": "healthy", 
            "model_loaded": model is not None,
            "ffmpeg_available": ffmpeg_available,
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
    
    # Check FFmpeg availability
    check_ffmpeg()
    
    # Pre-load model
    try:
        load_model_once()
    except Exception as e:
        logger.warning(f"Initial model load failed: {str(e)}")
    
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
