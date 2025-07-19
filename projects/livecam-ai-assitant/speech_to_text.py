import logging
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO 
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def record_audio(file_path, timeout=20, phrase_time_limit=None):
    """
    Function to record audio from the microphone and save it as an MP# file.
    
    Args:
    file_path (str): Path to save the recorded audio file.
    timeout (int): Maximum time to wait for phrase to star ( in seconds).
    phrase_time_limit (int):Maximum Time for the phrase to be record audio (in seconds).
    """

    recognizer= sr.Recognizer()
    try:
        with sr.Microphone() as source:
            logging.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1) 
            logging.info("star speaking now...")


            #record the audio
            audio_data =recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            

            #Convert the recorded audio to an MP3 file
            wav_data= audio_data.get_wav_data()
            audio_segment= AudioSegment.from_wav(BytesIO(wav_data))
            audio_segment.export(file_path, format="mp3", bitrate="128k")

            logging.info(f"Audio saved to {file_path}")
    except Exception as e:
        logging.error(f"Error recording audio: {e}")

# file_path="test_speech_to_text.mp3"
# record_audio(file_path, timeout=10, phrase_time_limit=10)

def transcribe_with_groq(audio_filepath):
    GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
    client=Groq(api_key=GROQ_API_KEY)
    stt_model="whisper-large-v3"
    audio_file=open(audio_filepath, "rb")
    transcription=client.audio.transcriptions.create(
        model=stt_model,
        file=audio_file,
        language="en"
    )

    return transcription.text


# audio_filepath="test_speech_to_text.mp3"
# print(transcribe_with_groq(audio_filepath))