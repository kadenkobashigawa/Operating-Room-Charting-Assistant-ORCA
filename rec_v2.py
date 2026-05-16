import time
import sys
import sounddevice as sd
import numpy as np
import os
from datetime import datetime
from scipy.io.wavfile import write



def record_audio(rec_folder_path: str = 'recordings', rec_name: str = 'recording',
                 samplerate: int = 44100, channels: int = 1,
                 stop_event = None) -> str:

    '''
    records audio from the default microphone and saves it as a WAV file.

    Parameters
    ----------
        rec_folder_path : str
            folder directory where the recording will be saved.
            Created if it does not exist. Default is 'recordings'.
        rec_name : str
            base name for the output file. A timestamp and .wav extension
            will be appended. Default is 'recording'.
        samplerate : int
            sample rate in Hz. Default is 44100.
        channels : int
            number of audio channels. 1 for mono, 2 for stereo. Default is 1.
        stop_event : threading.Event or None
            event used to signal when to stop recording. If None (CLI use),
            waits for the user to press ENTER. If a threading.Event (server
            use), records until stop_event.set() is called.

    Returns
    -------
        filepath : str
            filepath of the saved recording.

    Example
    --------
        >>> record_audio()
        Recording saved as: recordings/recording_20260315_143022.wav
    '''

    print('\nRecording audio...')
    start = time.time()
    recording = []

    #callback is called by sounddevice for each audio chunk captured...
    def callback(indata, frames, ts, status):
        if status:
            print(status, file = sys.stderr)
        recording.append(indata.copy())

    #open the input stream and block until stop condition is met...
    try:
        with sd.InputStream(samplerate = samplerate,
                            channels = channels,
                            callback = callback):
            if stop_event is None:
                input('Press ENTER to end recording: ')
            else:
                stop_event.wait()
    except sd.PortAudioError as e:
        raise RuntimeError(
            f'🅇 Cannot open audio input: {e}\n'
            'Check that a microphone is connected and system permissions allow access.'
        ) from e

    #merge all buffered chunks into a single array...
    if not recording:
        raise RuntimeError('No audio captured — check microphone and permissions.')
    audio_data = np.concatenate(recording, axis = 0)

    #save recording as a timestamped wav file...
    os.makedirs(rec_folder_path, exist_ok = True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fn = f'{rec_name}_{timestamp}.wav'
    filepath = os.path.join(rec_folder_path, fn)
    write(filepath, samplerate, audio_data)
    elapsed = time.time() - start
    print(f'✔ Recording ({elapsed // 60:.0f}m {elapsed % 60:.0f}s) saved as: {filepath}\n')
    return filepath