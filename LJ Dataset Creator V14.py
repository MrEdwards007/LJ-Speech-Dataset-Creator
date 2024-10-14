import whisper
import nltk
import pandas as pd
from pydub import AudioSegment
import os
import datetime
import string
import time
from pathlib import Path

### You may use this program as you wish, but I ask that you include me (Waverly Edwards) in the credits

# Ensure the NLTK 'punkt' tokenizer is downloaded for sentence splitting
def download_nltk_punkt():
    """
    Checks if the NLTK 'punkt' tokenizer is available.
    Downloads it if it's not already present to enable sentence tokenization.
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt_tab')

# Call the function to ensure 'punkt' is available without displaying download messages
download_nltk_punkt()

# Function to desensitize text by removing punctuation, whitespace, and converting to lowercase
def desensitize_text(text):
    """
    Desensitizes the input text by removing spaces, punctuation, and converting to lowercase.
    This aids in comparing transcriptions by normalizing them.

    Parameters:
    - text: The string to desensitize.

    Returns:
    - A normalized string suitable for comparison.
    """
    translator = str.maketrans('', '', string.punctuation + string.whitespace)
    return text.translate(translator).lower()

# Function to split long sentences into smaller chunks based on segment duration
def split_long_sentence(words_in_sentence, min_duration, max_duration, absolute_max):
    """
    Splits a sentence into smaller chunks if its duration exceeds the target segment length.

    Conditions for starting a new chunk:
    1. Adding this word would exceed absolute max duration.
    2. The current chunk duration is >= minimum duration AND adding the word would exceed the target max duration.

    Parameters:
    - words_in_sentence: List of words in the sentence with their timestamps.
    - min_duration: Minimum segment duration (in seconds).
    - max_duration: Target maximum segment duration (in seconds).
    - absolute_max: Hard limit for the segment duration (in seconds).

    Returns:
    - List of chunks, where each chunk is a list of words that can be processed as a segment.
    """
    chunks = []
    current_chunk = []
    chunk_start_time = None
    chunk_duration = 0.0

    for word in words_in_sentence:
        word_duration = word['end'] - word['start']

        # Start new chunk if this is the first word
        if not current_chunk:
            current_chunk = [word]
            chunk_start_time = word['start']
            chunk_duration = word_duration
            continue

        potential_duration = word['end'] - chunk_start_time

        # Conditions for starting a new chunk
        if (potential_duration > absolute_max) or \
           (chunk_duration >= min_duration and potential_duration > max_duration):
            # Add current chunk to chunks list
            chunks.append({
                'words': current_chunk,
                'start': chunk_start_time,
                'end': current_chunk[-1]['end'],
                'duration': chunk_duration
            })
            # Start new chunk with current word
            current_chunk = [word]
            chunk_start_time = word['start']
            chunk_duration = word_duration
        else:
            # Add word to current chunk
            current_chunk.append(word)
            chunk_duration = word['end'] - chunk_start_time

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append({
            'words': current_chunk,
            'start': chunk_start_time,
            'end': current_chunk[-1]['end'],
            'duration': current_chunk[-1]['end'] - chunk_start_time
        })

    return chunks

# Function to process sentences and create segments based on duration constraints
def process_sentences(sentence_spans, transcript, word_list, min_duration, max_duration, absolute_max):
    """
    Processes the sentences from the transcript and creates audio segments based on duration constraints.

    Parameters:
    - sentence_spans: List of sentence spans with character offsets.
    - transcript: Full transcript text.
    - word_list: List of words with their timestamps and character positions.
    - min_duration: Minimum segment duration (in seconds).
    - max_duration: Target maximum segment duration (in seconds).
    - absolute_max: Hard limit for the segment duration (in seconds).

    Returns:
    - List of segments, each containing words and timing information.
    """
    segments = []
    current_segment = []
    segment_start_time = None
    segment_duration = 0.0

    for sent_start, sent_end in sentence_spans:
        sentence_text = transcript[sent_start:sent_end].strip()
        words_in_sentence = [w for w in word_list
                             if w['char_start'] >= sent_start and w['char_end'] <= sent_end]

        if not words_in_sentence:
            continue

        sentence_duration = words_in_sentence[-1]['end'] - words_in_sentence[0]['start']

        # If this single sentence is longer than our max duration, split it
        if sentence_duration > max_duration:
            # First, handle any existing segment
            if current_segment:
                segments.append({
                    'words': current_segment,
                    'start': segment_start_time,
                    'end': current_segment[-1]['end'],
                    'duration': segment_duration
                })
                current_segment = []
                segment_start_time = None
                segment_duration = 0.0

            # Split the long sentence
            chunks = split_long_sentence(words_in_sentence,
                                         min_duration,
                                         max_duration,
                                         absolute_max)
            segments.extend(chunks)
            continue

        # Try to add sentence to current segment
        if not current_segment:
            current_segment = words_in_sentence
            segment_start_time = words_in_sentence[0]['start']
            segment_duration = sentence_duration
        else:
            potential_duration = words_in_sentence[-1]['end'] - segment_start_time

            # Conditions for starting a new segment
            if potential_duration > absolute_max or \
               (segment_duration >= min_duration and potential_duration > max_duration):
                # Store current segment and start a new one
                segments.append({
                    'words': current_segment,
                    'start': segment_start_time,
                    'end': current_segment[-1]['end'],
                    'duration': segment_duration
                })
                current_segment = words_in_sentence
                segment_start_time = words_in_sentence[0]['start']
                segment_duration = sentence_duration
            else:
                # Add sentence to current segment
                current_segment.extend(words_in_sentence)
                segment_duration = current_segment[-1]['end'] - segment_start_time

    # Handle the last segment
    if current_segment:
        segments.append({
            'words': current_segment,
            'start': segment_start_time,
            'end': current_segment[-1]['end'],
            'duration': segment_duration
        })

    return segments

# Function to get the text from a segment's words
def get_segment_text(segment):
    """
    Concatenates the words in a segment to form the full segment text.

    Parameters:
    - segment: A dictionary containing 'words', each with 'word' keys.

    Returns:
    - A string representing the full text of the segment.
    """
    return ' '.join(word['word'] for word in segment['words'])

# Function to extract audio for a given segment
def extract_audio_for_segment(audio, start_time, end_time, segment_text, audio_duration,
                              time_offset, output_folder, audio_file_base, segment_counter,
                              transcriptions):
    """
    Extracts the audio segment from the original audio and saves it as a new file.
    Also updates the transcriptions list with segment details.

    Parameters:
    - audio: The original audio loaded using pydub.
    - start_time: Start time of the segment in seconds.
    - end_time: End time of the segment in seconds.
    - segment_text: Transcription text for the segment.
    - audio_duration: Total duration of the original audio in seconds.
    - time_offset: Time offset to adjust start and end times.
    - output_folder: Folder path to save the extracted audio segments.
    - audio_file_base: Base name for the audio files.
    - segment_counter: Counter to number the segments.
    - transcriptions: List to store transcription details.

    Returns:
    - Updated segment_counter and transcriptions list.
    """
    segment_counter += 1

    # Adjust start and end times by the offset
    adjusted_start_time = max(0, start_time - time_offset)
    adjusted_end_time = min(audio_duration, end_time + time_offset)

    # Convert start and end times to milliseconds
    start_ms = int(adjusted_start_time * 1000)
    end_ms = int(adjusted_end_time * 1000)

    # Extract the corresponding audio segment
    split_audio = audio[start_ms:end_ms]

    # Create the output filename for the audio segment
    segment_file_name = f"{audio_file_base}_segment_{segment_counter}.wav"
    segment_file_path = os.path.join(output_folder, segment_file_name)

    # Export the audio segment to the file
    split_audio.export(segment_file_path, format="wav")

    # Calculate the segment length in seconds, formatted to 3 decimal places
    segment_length = f"{(end_ms - start_ms) / 1000:.3f}"

    # Append the segment details to the transcriptions list
    transcriptions.append({
        "file_name": f"wavs/{segment_file_name}",    # Relative path to wavs folder
        "transcription": segment_text.strip(),       # Segment transcription
        "start_time": f"{adjusted_start_time:.3f}",  # Adjusted start time in seconds
        "end_time": f"{adjusted_end_time:.3f}",      # Adjusted end time in seconds
        "segment_length_seconds": segment_length     # Always 3 decimal places
    })

    return segment_counter, transcriptions

# Main function to process the audio file
def process_audio_file(audio_file_path, base_directory, model_name="large-v3",
                       min_segment_duration=6.0, max_segment_duration=8.0, max_overage=3.0,
                       time_offset=0.15, language="en", verbose=True):
    """
    Processes an audio file by splitting it into smaller segments, transcribing it with Whisper,
    and saving both the segmented audio and corresponding transcriptions. Includes revalidation and logging.

    Parameters:
    - audio_file_path: Path to the input audio file.
    - base_directory: Base directory for input and output files.
    - model_name: Whisper model to use for transcription.
    - min_segment_duration: Minimum duration for audio segments in seconds.
    - max_segment_duration: Target maximum duration for audio segments in seconds.
    - max_overage: Maximum allowed overage beyond the target maximum duration.
    - time_offset: Time offset to adjust the start and end times of segments.
    - language: Language code for transcription.
    - verbose: Whether to print verbose logs during processing.

    Returns:
    - None. Outputs are saved to files and logs are printed.
    """
    # Record the start time of the entire process
    start_time = time.time()

    # Load the specified Whisper model
    model = whisper.load_model(model_name)

    # Detect audio file format based on the file extension
    audio_format = Path(audio_file_path).suffix[1:]  # Get file extension without the dot

    # Load the audio file using pydub for various formats
    try:
        if audio_format in ['wav', 'mp3', 'flac', 'mp4', 'm4a', 'ogg']:
            audio = AudioSegment.from_file(audio_file_path, format=audio_format)
        else:
            raise ValueError(f"Unsupported audio format: {audio_format}")
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return

    # Calculate total duration of the audio file
    audio_duration = len(audio) / 1000.0  # Duration in seconds

    # Configuration constants
    MIN_SEGMENT_DURATION  = min_segment_duration
    MAX_SEGMENT_DURATION  = max_segment_duration
    MAX_OVERAGE           = max_overage
    ABSOLUTE_MAX_DURATION = MAX_SEGMENT_DURATION + MAX_OVERAGE
    TIME_OFFSET           = time_offset

    # ---------------------
    # Clipping of audio causes artifacts in the TTS model, so we must avoid this.
    #
    # TIME_OFFSET is set to a default of 0.15 seconds but can and should be adjusted per Whisper model.
    # An offset of 0.15 seconds appears to be good for "large-v3-turbo".
    #
    # Without this offset, words might be clipped at the beginning or end.
    # Since pauses between words are typically 150-200 milliseconds (0.15-0.20 seconds),
    # a small buffer is added to compensate for potential timing inaccuracies in Whisper's word-level transcription.
    # Ideally, this value should be automatically refined based on comparison with known, accurate audio timings as a warm-up run.
    # Since Whisper's timing discrepancies may not follow a linear pattern, the required offset may vary by model.
    # Thus, switching models would engage the warm-up timing exercise.
    # ---------------------

    # Prepare output directories
    audio_file_base = os.path.splitext(os.path.basename(audio_file_path))[0]
    output_folder   = os.path.join(base_directory, "Output", "wavs")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize list to store validated transcriptions for CSV
    transcriptions  = []
    segment_counter = 0

    # Transcribe the original audio file with Whisper (word-level transcription enabled)
    result = model.transcribe(audio_file_path, language=language, verbose=verbose, word_timestamps=True)

    # Build the full transcript and word list with character positions
    transcript = ''
    word_list = []

    for segment in result['segments']:
        for word in segment['words']:
            word_text = word['word'].strip()  # Strip any leading/trailing spaces
            if word_text:  # Ensure the word is not empty
                char_start = len(transcript)
                # Add a single space only if the transcript is not empty
                if transcript:
                    transcript += ' '
                transcript += word_text
                char_end = len(transcript)
                word_list.append({
                    'word': word_text,
                    'start': word['start'],
                    'end': word['end'],
                    'char_start': char_start,
                    'char_end': char_end
                })

    # Use NLTK's PunktSentenceTokenizer to get sentence spans with character offsets
    sent_tokenizer = nltk.tokenize.PunktSentenceTokenizer()
    sentence_spans = list(sent_tokenizer.span_tokenize(transcript))

    # Process sentences and create segments
    segments = process_sentences(sentence_spans, transcript, word_list,
                                 MIN_SEGMENT_DURATION, MAX_SEGMENT_DURATION, ABSOLUTE_MAX_DURATION)

    # Extract audio for each segment
    for segment in segments:
        segment_text = get_segment_text(segment)
        segment_counter, transcriptions = extract_audio_for_segment(
            audio, segment['start'], segment['end'], segment_text, audio_duration,
            TIME_OFFSET, output_folder, audio_file_base, segment_counter, transcriptions)

    # Convert the list of transcriptions to a pandas DataFrame
    df = pd.DataFrame(transcriptions)

    # ================================================
    # Validation Pass: Re-transcribe and Compare
    # ================================================

    # Prepare log file for discrepancies
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f"{audio_file_base}_discrepancy_{timestamp}.log"
    log_file_path = os.path.join(base_directory, "Output", log_file_name)

    # Initialize list to store discrepancies
    discrepancies = []

    # Iterate over each segment and validate
    for idx, row in df.iterrows():
        segment_file_path = os.path.join(base_directory, "Output", row['file_name'])
        original_transcription = row['transcription'].strip()

        # Re-transcribe the audio segment
        segment_result = model.transcribe(segment_file_path, language=language, verbose=False)
        new_transcription = segment_result['text'].strip()

        # Desensitize transcriptions
        desensitized_original = desensitize_text(original_transcription)
        desensitized_new = desensitize_text(new_transcription)

        # Compare desensitized transcriptions
        if desensitized_original != desensitized_new:
            # Record discrepancy with detailed information
            discrepancies.append({
                "file_name": row['file_name'],
                "original_transcription": original_transcription,
                "new_transcription": new_transcription,
                "desensitized_original": desensitized_original,
                "desensitized_new": desensitized_new
            })
            # Update the transcription in the DataFrame with the new transcription
            df.at[idx, 'transcription'] = new_transcription

    # Save the updated DataFrame to CSV file with "|" as the delimiter
    csv_file = os.path.join(base_directory, "Output", "Sentence_level_transcriptions.csv")
    df.to_csv(csv_file, index=False, sep='|')

    # Calculate total processing time
    end_time = time.time()
    total_processing_time = end_time - start_time  # In seconds

    # Calculate processing speed relative to real-time
    if total_processing_time > 0:
        processing_speed = audio_duration / total_processing_time
    else:
        processing_speed = 0

    # Write discrepancies to log file
    with open(log_file_path, 'w') as log_file:
        log_file.write(f"Processing completed at: {datetime.datetime.now()}\n")
        log_file.write(f"Configuration:\n")
        log_file.write(f"  Minimum segment duration: {MIN_SEGMENT_DURATION} seconds\n")
        log_file.write(f"  Target maximum duration: {MAX_SEGMENT_DURATION} seconds\n")
        log_file.write(f"  Maximum allowed overage: {MAX_OVERAGE} seconds\n")
        log_file.write(f"  Absolute maximum duration: {ABSOLUTE_MAX_DURATION} seconds\n")
        log_file.write(f"  Time offset: {TIME_OFFSET} seconds\n\n")

        if discrepancies:
            log_file.write("=== Discrepancies Found ===\n\n")
            for discrepancy in discrepancies:
                log_file.write(f"File: {discrepancy['file_name']}\n")
                log_file.write(f"Original Transcription: {discrepancy['original_transcription']}\n")
                log_file.write(f"New Transcription: {discrepancy['new_transcription']}\n")
                log_file.write(f"Desensitized Original: {discrepancy['desensitized_original']}\n")
                log_file.write(f"Desensitized New: {discrepancy['desensitized_new']}\n")
                log_file.write("-----\n")
        else:
            log_file.write("No discrepancies found between original and new transcriptions.\n")

        # Write summary statistics
        log_file.write("\n==== Summary Statistics ====\n")
        log_file.write(f"Total segments created: {len(df)}\n")
        log_file.write(f"Total discrepancies: {len(discrepancies)}\n")
        log_file.write(f"Total length of input audio file: {audio_duration:.2f} seconds\n")
        log_file.write(f"Total processing time: {total_processing_time:.2f} seconds\n")
        if processing_speed > 0:
            log_file.write(f"Processing speed: {processing_speed:.2f}x faster than real-time\n")
        else:
            log_file.write("Processing speed: N/A (processing time is zero)\n")

        # Add segment duration statistics
        segment_durations = df['segment_length_seconds'].astype(float)
        log_file.write(f"\nSegment Duration Statistics:\n")
        log_file.write(f"  Minimum: {segment_durations.min():.2f} seconds\n")
        log_file.write(f"  Maximum: {segment_durations.max():.2f} seconds\n")
        log_file.write(f"  Average: {segment_durations.mean():.2f} seconds\n")
        log_file.write(f"  Median: {segment_durations.median():.2f} seconds\n")

        # Add distribution of segment durations
        duration_bins = [0, 2, 4, 6, 8, 10, float('inf')]
        duration_labels = ['0-2s', '2-4s', '4-6s', '6-8s', '8-10s', '>10s']
        duration_counts = pd.cut(segment_durations, bins=duration_bins, labels=duration_labels).value_counts()

        log_file.write("\nSegment Duration Distribution:\n")
        for label, count in duration_counts.items():
            log_file.write(f"  {label}: {count} segments\n")

    # Output processing information to the console
    print(f"Processing complete!")
    print(f"Audio files saved to: {output_folder}")
    print(f"Transcriptions saved to: {csv_file}")
    print(f"Log file saved to: {log_file_path}")
    print(f"Total processing time: {total_processing_time:.2f} seconds")
    if processing_speed > 0:
        print(f"Processing speed: {processing_speed:.2f}x faster than real-time")
    else:
        print("Processing speed: N/A (processing time is zero)")


def process_audio_detailed(audio_file_path, base_directory, model_name):
    """
    A wrapper function to process an audio file with specific parameters.
    
    Parameters:
    - audio_file_path (str): The full path to the audio file that needs to be processed.
    - base_directory (str): The base directory where output (transcriptions and audio segments) will be saved.
    - model_name (str): The name of the Whisper model to be used for transcription.
    """

    # Explicit call to process the audio file with detailed configuration parameters:
    # - min_segment_duration: Minimum duration (in seconds) for each audio segment.
    # - max_segment_duration: Maximum duration (in seconds) for each audio segment.
    # - max_overage: Allowable overage (in seconds) beyond the max duration.
    # - time_offset: Time buffer (in seconds) to prevent clipping of audio at segment boundaries.
    # - language: The language code for transcription (English in this case).
    # - verbose: Set to True to enable detailed output from the Whisper model during transcription.
    process_audio_file(
        audio_file_path=audio_file_path,   # Path to the audio file being processed
        base_directory=base_directory,     # Directory where results will be saved
        model_name=model_name,             # Whisper model to use for transcription (e.g., "large-v3-turbo")
        min_segment_duration=6.0,          # Minimum duration of an audio segment (6 seconds)
        max_segment_duration=8.0,          # Maximum duration of an audio segment (8 seconds)
        max_overage=3.0,                   # Allowable overage of 3 seconds beyond the max segment duration
        time_offset=0.15,                  # 0.15-second buffer to avoid clipping at segment boundaries
        language="en",                     # Language code for transcription (English)
        verbose=True                       # Enable detailed output for debugging or monitoring
    )

### ---------- ####

# Example usage:
# "large-v3" is recommended, while "large-v3-turbo" is for TESTING because it is 6x faster than large-v3 but has higher word-error-rate (WER)
base_directory    = "/output/path/" # The directory you want the output to go to
audio_file_path   = "/input/path/My_Audio.mp3"   # location of the audio file to process
model_name        = "large-v3" # "large-v3-turbo" for TESTING
recommend_models  = ["large","large-v2","large-v3"] # The models downloaded due to low word-error-rate

# Implicit arguments to process the audio file
process_audio_file(audio_file_path, base_directory)

# ALTERNATIVELY
# Explicit arguments to process the audio file
#process_audio_detailed(audio_file_path, base_directory, model_name)
