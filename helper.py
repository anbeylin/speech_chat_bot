import sounddevice as sd
import numpy as np

import soundfile as sf # install as pysoundfile
from pydub import AudioSegment # also need ffmpeg to be installed

# import threading
import queue
import os
import keyboard

import openai
from openai import OpenAI

from pygame import mixer  # Load the popular external library

import inspect

FILENAME = "test"
APIKEY = "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"
client = OpenAI(api_key=APIKEY)
system_message = \
"""
You are a helpful AI assistant from speech to speech. User input is coming from speech to text transcriber and may contain errors due to incorrect speech detection.
YOur answer will be encoded into speech, keep it short but conversational. Unless user specifically requests to be detailed try to keep answer under a 100 words.
"""

# recording = record_audio_fixed_length()

# recording = record_audio()  # Start recording

# sd.default.device = [1, 5]

# test_text = 'I have a new cat. She is cute and fluffy and I want to give her a good name. Please give me a few suggestions for nice cat names.\n'

# def record_audio_fixed_length(duration=5, fs=FS):
#     """
#     Record audio from the microphone for a given duration.
#     """
#     print("Recording...")
#     recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
#     sd.wait()  # Wait until recording is finished
#     print("Recording stopped.")
#     return recording


def record_audio(fs=44100):
    """
    Record audio from the microphone. Start and end recording with a spacebar.
    """
    
    # This queue will hold the recorded audio frames
    audio_queue = queue.Queue()
    
    def callback(indata, frames, time, status):
        """This is called for each audio block."""
        if status:
            print(status, file=sys.stderr)
        audio_queue.put(indata.copy())
    
    print("Press Space to start recording...")
    keyboard.wait('space')  # Wait until space is pressed to start recording

    # Open the stream with the callback and specified sample rate
    # This will store recorded audio frames in audio_queue
    with sd.InputStream(samplerate=fs, channels=1, callback=callback):
        print("Recording... Press Space to stop.")
        keyboard.wait('space')  # Wait until space is pressed again

    # Retrieve audio data from the queue
    audio_data = []
    while not audio_queue.empty():
        audio_data.append(audio_queue.get())
    
    # Concatenate all the audio chunks into one NumPy array
    audio_data = np.concatenate(audio_data)

    # result_queue.put(recording) - this line could be used instead of return to run the function in the separate thread
    return audio_data


def save_as_mp3(recording, filename, fs=44100):
    # Save the recording as a WAV file
    sf.write(filename + '.wav', recording, fs)

    # Load the WAV file with pydub and export as MP3
    audio_segment = AudioSegment.from_wav(filename + '.wav')
    audio_segment.export(filename + '.mp3', format='mp3')

    # Delete the WAV file
    os.remove(filename + '.wav')

# def save_recording_as_mp3(audio_data, fs=FS, file_name="recording.mp3"):
#     """
#     Save the recorded audio data as an MP3 file.

#     Parameters:
#     audio_data (numpy.ndarray): The recorded audio data.
#     fs (int): The sampling rate of the audio data.
#     file_name (str): The name of the file to save the recording to.
#     """
#     # Normalize the array to be in the range of -1 to 1, as required by pydub
#     audio_data = audio_data / np.max(np.abs(audio_data))

#     # Convert the normalized audio array to a format compatible with pydub
#     audio_segment = AudioSegment(
#         audio_data.astype("float32").tobytes(),
#         frame_rate=fs,
#         sample_width=audio_data.dtype.itemsize,
#         channels=1
#     )

#     # Export the audio segment to an MP3 file
#     audio_segment.export(file_name, format="mp3")
#     print(f"File saved as {file_name}")

def Mp3Transcribe(filename):
    audio_file = open(filename + ".mp3", "rb")

    parameters = {
        "model":"whisper-1", 
        "file":audio_file, 
        "response_format":"text",
        "language":"english"}

    # transcript = openai.Audio.transcribe(**parameters)
    transcript = client.audio.transcriptions.create(**parameters)
    
    return transcript

def QueryLLM(question):
    parameters = {
      'model': 'gpt-3.5-turbo', 
      'messages': [{"role": "system", "content": system_message}, 
                   {"role": "user", "content": question}, 
                   {"role": "assistant", "content": ""}]
    }

    result = client.chat.completions.create(**parameters)

    text_reply = result.choices[0].message.content

    return text_reply

def text_preprocessing(text):
    return text.replace("\n", " ").strip()

def TTS_Reply(text_reply, filename):
    response = client.audio.speech.create(
        model="tts-1-hd", # tts-1, tts-1-hd
        voice="echo", # alloy, echo, fable, onyx, nova, shimmer
        input=text_reply  #  The text to generate audio for. The maximum length is 4096 characters.
    )
    
    response.stream_to_file(filename + "_reply.mp3")

def play_reply(filename):
    reply_filename = filename + "_reply.mp3"
    # audio = AudioSegment.from_mp3(reply_filename)
    # play(audio)

    mixer.init()
    mixer.music.load(reply_filename)
    mixer.music.play()

    # os.remove(reply_filename)

def SpeechToAnswer():
    recording = record_audio()

    save_as_mp3(recording, FILENAME)

    print("Transcribing")
    transcription = Mp3Transcribe(FILENAME)
    # os.remove(filename + '.mp3')
    print(transcription)

    print("Sending question to LLM")
    answer = QueryLLM(transcription)

    print("Answer from LLM:")
    print(answer)
    
    TTS_Reply(text_preprocessing(answer), FILENAME)

    play_reply(FILENAME)
    
    return answer