import os
import gradio as gr
from speech_to_text import record_audio, transcribe_with_groq
from text_to_speech import text_to_speech_with_elevenlabs , text_to_speech_with_gtts
from ai_agent import ask_agent


GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
audio_filepath="audio_question.mp3 "
