import os
import time
import torch
from faster_whisper import WhisperModel
from functools import lru_cache



#detect available device...
if torch.cuda.is_available():
    DEVICE = 'cuda'
    COMPUTE_TYPE = 'float16'
    print('\nGPU detected. Device set to cuda.')
else:
    DEVICE = 'cpu'
    COMPUTE_TYPE = 'int8'
    print('\nNo GPU detected. Device set to cpu.')

#load each model once...
@lru_cache(maxsize = 7)
def get_model(asr_model: str = 'large-v3', device: str = DEVICE, compute_type: str = COMPUTE_TYPE):

    '''
    loads and caches a WhisperModel instance for the given configuration.

    Parameters
    ----------
        asr_model : str
            whisper model size, e.g. "large-v3". Default is "large-v3".
        device : str
            compute device, e.g. "cuda" or "cpu".
        compute_type : str
            quantization type, e.g. "float16" or "int8".

    Returns
    -------
        model : WhisperModel
            cached model instance for the given (asr_model, device, compute_type) key.
    '''

    model = WhisperModel(asr_model, device = device, compute_type = compute_type)
    return model


def transcribe_file(audio_path: str, asr_model: str = 'large-v3', on_segment = None, show_output: bool = False):

    '''
    transcribes a WAV audio file to text using faster-whisper.

    Parameters
    ----------
        audio_path : str
            path to the audio file to transcribe,
            e.g. "recordings/recording_20260315_143022.wav".
        asr_model : str
            whisper model size to use, e.g. "tiny", "base", "small", "medium",
            "large" (v1-3). Default is "large-v3".
        on_segment : callable or None
            optional callback invoked after each segment with the running
            list of word dicts, e.g. on_segment([{"word": "hello", "conf": 0.95}, ...]).
            Default is None.
        show_output : bool
            option to show output from function. Default is False.

    Returns
    -------
        transcript : str
            full transcribed text as a single string.
    '''

    #guard against missing file before handing off to the model...
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f'Audio file not found: {audio_path}')

    #load (or retrieve cached) whisper model...
    if show_output:
        print(f'\nTranscribing audio with "{asr_model}" Whisper...')
    model = get_model(asr_model)
    start = time.time()

    #transcribe audio with word-level timestamps for confidence scores...
    segments, _ = model.transcribe(
        audio_path,
        language = 'en',
        task = 'transcribe',
        word_timestamps = True
    )

    #accumulate words with confidence, firing the callback after each segment...
    all_words = []
    for segment in segments:
        if segment.words:
            for w in segment.words:
                all_words.append({'word': w.word, 'conf': round(w.probability, 3)})
        else:
            all_words.append({'word': segment.text, 'conf': 1.0})
        if on_segment:
            on_segment(all_words[:])

    transcript = ''.join(w['word'] for w in all_words).strip()

    #warn if nothing was transcribed — likely silent or very short audio...
    if not transcript:
        print('⚠ Warning: transcript is empty — audio may be silent or too short.')

    #return transcript...
    if show_output:
        elapsed = time.time() - start
        print(f'Transcript processing time: {elapsed // 60:.0f}m {elapsed % 60:.0f}s\n')

    return transcript
