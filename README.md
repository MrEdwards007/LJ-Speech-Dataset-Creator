# LJ-Speech-Dataset-Creator
Audio processing and transcription pipeline using Whisper AI. Splits audio into segments, transcribes, revalidates, and logs discrepancies, with support for multiple audio formats and customizable segment duration.

<a name="readme-top"></a>

<!-- ABOUT THE PROJECT -->
## About The Project


The purpose of this program is to process audio files by segmenting them into smaller parts based on word-level timestamps, transcribe the segments using Whisper, and save both the audio segments and their transcriptions. It is common practice to use the "LJ Dataset" format, where each entry in the CSV contains the file path to the audio file along with its corresponding transcription.

Key functions include:

1. **Audio segmentation**: The program splits the audio into smaller segments that meet specific duration criteria (with minimum, maximum, and absolute maximum durations). It adjusts the start and end times using a configurable offset to avoid clipping words at the boundaries.
2. **Transcription**: The program uses Whisper's word-level transcription to create time-aligned transcriptions of the audio segments, storing the results in a CSV file.
3. **Validation**: After creating the segments, the program re-transcribes the audio segments and compares the new transcription to the original, noting any discrepancies.
4. **Logging and analysis**: The program logs various details about the segmentation process, including the timing offsets, processing speed, discrepancies found, and statistics on segment durations.

This program is useful for audio processing workflows such as creating datasets for text-to-speech models or analyzing large audio files with detailed word-level transcriptions.

### Caveats

Re-transcription of the shorter audio segments is crucial because Whisper tends to provide more accurate transcriptions for smaller where it is **not** using the context of the surrounding audio to inform the current audio transcription. Basically, the we are forcing it to infer less about what is being said in the audio. By reprocessing the segmented audio, the second pass will likely correct inaccuracies from the initial transcription. Note, that Whisper does hallucinate, but those are reduced on the second pass, due to the more focused transcription on the shorter segment.

Potential caveats or challenges with the program:

1. **Whisper Model Dependency**: The program uses the Whisper model for transcription, and the timing of word-level transcription can vary across different models. As mentioned in the code, the time offset (set to 0.15 seconds) may not be consistent for all models. Users need to experiment with the offset for accuracy when switching models.
2. **Timing Inaccuracies**: The timing offset (TIME_OFFSET)should **ideally** be calculated automatically against a known audio sample with known timing as a warm-up. Therefore, adjusting the offset manually for different models and datasets may introduce variability in the results. It’s not clear whether Whisper's timing errors are linear or random, making it hard to perfectly compensate for this issue.
3. **Sentence Splitting Accuracy**: The use of NLTK’s PunktSentenceTokenizer for sentence splitting works on textual data, but it might not always accurately reflect natural pauses or speech patterns in audio, especially in conversational or non-standard language contexts.
4. **Model Updates**: If OpenAI releases updates to Whisper models, the behavior of the transcription (including word timings and accuracy) could change, requiring further recalibration of the program (supporting the idea of automatic calibration, see Timing Inaccuracies).
5. **Audio Quality**: The accuracy of transcriptions and segmentation's is highly dependent on the quality of the input audio. Poor-quality audio (e.g., with noise or unclear speech) may result in lower-quality transcriptions and segment splitting errors.
6. **Non-English Languages**: While the program is primarily set up for English audio, it can be adapted for other languages since Whisper supports multiple languages and NLTK provides tools for various language processing.


<!-- USAGE EXAMPLES -->
## Usage

Define 
- The input file, which may be an audio or video file.  
- The output folder for the transcription file.  
- The Whisper model you will use to transcribe.
- Other parameters may be explicitly set.

## Output

Metadata is provided in Sentence_level_transcriptions.csv. This file consists of one record per line, delimited by the pipe character (|). 

The fields are:

  - **ID**: This is the relative path and name of the corresponding .wav file containing the segmented audio.
  - **Transcription**: The transcribed words spoken by the reader in the segmented .wav file.

Additionally, the following information is also available in each row of the CSV:

  - **Start Time**: The time (in seconds) when the corresponding segment starts within the original audio file.
  - **End Time**: The time (in seconds) when the corresponding segment ends within the original audio file.
  - **Segment Length (Seconds)**: The length of the audio segment, expressed in seconds, always formatted to three decimal places.

This structure allows easy mapping between each audio segment and its corresponding transcription.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


```Python
# Example usage:
# "large-v3" is recommended, while "large-v3-turbo" is for TESTING because it is 6x faster than large-v3 but has higher word-error-rate (WER)

base_directory    = "/output/path/"                 # The directory you want the output to go to
audio_file_path   = "/input/path/My_Audio.mp3"      # location of the audio file to process
model_name        = "large-v3"                      # "large-v3-turbo" for TESTING
recommend_models  = ["large","large-v2","large-v3"] # The models recommended due to low word-error-rate

# Implicit arguments to process the audio file
process_audio_file(audio_file_path, base_directory)

# Explicit arguments to process the audio file --SEE FUNCTION for all parameters used
process_audio_detailed(audio_file_path, base_directory, model_name)
```


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [ ] Develop an automatic process for determining the TIME_OFFSET against the user's chosen Whisper model.  This would use a pre-created audio file with known alignment timing.
- [ ] Developing a GUI interface.  It doesn't need it, as it complicates the code, using Streamlit or Gradio, but it is a nice to have for making parameter adjustments and for those who dont read Python code.


<p align="right">(<a href="#readme-top">back to top</a>)</p>




<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* OpenAI
* Github Community

<p align="right">(<a href="#readme-top">back to top</a>)</p>

