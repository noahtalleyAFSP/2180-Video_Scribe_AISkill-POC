import logging
from pathlib import Path
from typing import Any, Dict, List

# This will be installed by the user with `poetry add openai-whisper`
import whisper

logger = logging.getLogger(__name__)

class LocalTranscriber:
    """Wrapper for OpenAI Whisper transcription."""
    def __init__(self, model_name: str = "base"):
        logger.info(f"Loading local Whisper model: {model_name}")
        try:
            self.model = whisper.load_model(model_name)
            logger.info("Whisper model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Whisper model '{model_name}'. Please ensure it's a valid model name and that you have a working internet connection for the first download.")
            raise e

    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Transcribes the audio file and returns segments compatible with the project.
        """
        logger.info(f"Transcribing audio file: {audio_path}")
        result = self.model.transcribe(audio_path, word_timestamps=False)
        logger.info("Transcription complete.")
        return result["segments"] 