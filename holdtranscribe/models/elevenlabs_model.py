import os
import logging
import tempfile
import threading
from typing import Dict, List, Optional, Generator, Any, Union

try:
    from elevenlabs import ElevenLabs, VoiceSettings
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False
    ElevenLabs = None
    VoiceSettings = None

logger = logging.getLogger(__name__)


class ElevenLabsTTSModel:
    """ElevenLabs Text-to-Speech model implementation"""

    def __init__(self, api_key: Optional[str] = None, model_id: str = "eleven_multilingual_v2", **kwargs):
        """
        Initialize ElevenLabs TTS model

        Args:
            api_key: ElevenLabs API key (if None, will try to get from environment)
            model_id: Model to use for synthesis
            **kwargs: Additional configuration options
        """
        if not ELEVENLABS_AVAILABLE:
            raise ImportError("ElevenLabs package not installed. Install with: pip install elevenlabs")

        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self.model_id = model_id
        self.client = None
        self.loaded = False

        # Default settings
        self.voice_id = "21m00Tcm4TlvDq8ikWAM"  # Default voice
        self.voice_settings = VoiceSettings(
            stability=0.5,
            similarity_boost=0.5,
            style=0.0,
            use_speaker_boost=True
        )
        self.output_format = "mp3_44100_128"
        self.optimize_streaming_latency = 0

        # Streaming parameters
        self.chunk_size = 1024
        self.streaming_chunk_length_schedule = [120, 160, 250, 290]

        # Voice consistency settings
        self.voice_consistency = True

        logger.info(f"Initialized ElevenLabs TTS model with model_id: {self.model_id}")

    def load(self) -> bool:
        """Load the ElevenLabs model (initialize client)"""
        try:
            if not self.api_key:
                raise ValueError("ElevenLabs API key is required")

            self.client = ElevenLabs(api_key=self.api_key)

            # Test the connection by getting user info (optional - some API keys may not have user_read permission)
            try:
                user_info = self.client.user.get()
                logger.info(f"Successfully connected to ElevenLabs API. User: {user_info}")
            except Exception as user_error:
                # If user info fails due to permissions, that's OK - we can still use TTS
                if "missing_permissions" in str(user_error) or "user_read" in str(user_error):
                    logger.info("ElevenLabs API connected (user info not accessible due to API key permissions)")
                else:
                    # If it's a different error, log it but continue
                    logger.warning(f"Could not get user info from ElevenLabs API: {user_error}")

            self.loaded = True
            return True

        except Exception as e:
            logger.error(f"Failed to load ElevenLabs model: {e}")
            self.loaded = False
            return False

    def unload(self):
        """Unload the model"""
        self.client = None
        self.loaded = False
        logger.info("ElevenLabs model unloaded")

    def set_seed(self, seed: Optional[int] = None):
        """Set random seed for reproducible generation"""
        # ElevenLabs supports seed parameter in API calls
        self.seed = seed
        if seed is not None:
            logger.info(f"Set seed to {seed}")

    def set_voice_conditioning(self, enabled: bool = True):
        """Set voice conditioning (consistency) settings"""
        self.voice_consistency = enabled
        logger.info(f"Voice consistency set to {enabled}")

    def split_text_for_streaming(self, text: str, target_chunk_length: int = 200) -> List[str]:
        """
        Split text into chunks suitable for streaming synthesis

        Args:
            text: Input text to split
            target_chunk_length: Target length for each chunk

        Returns:
            List of text chunks
        """
        if len(text) <= target_chunk_length:
            return [text]

        chunks = []
        sentences = text.split('. ')
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk + sentence) <= target_chunk_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _apply_voice_consistency(self, text: str, previous_text: Optional[str] = None) -> Dict[str, Any]:
        """Apply voice consistency parameters"""
        params = {}
        if self.voice_consistency and previous_text:
            params["previous_text"] = previous_text
        return params

    def synthesize(self, text: str, output_path: Optional[str] = None, **kwargs) -> Optional[str]:
        """
        Synthesize text to speech

        Args:
            text: Text to synthesize
            output_path: Optional path to save audio file
            **kwargs: Additional synthesis parameters

        Returns:
            Path to generated audio file or None if failed
        """
        if not self.loaded or not self.client:
            logger.error("Model not loaded")
            return None

        try:
            # Prepare synthesis parameters
            voice_id = kwargs.get('voice_id', self.voice_id)
            model_id = kwargs.get('model_id', self.model_id)
            voice_settings = kwargs.get('voice_settings', self.voice_settings)
            output_format = kwargs.get('output_format', self.output_format)

            # Additional parameters
            seed = kwargs.get('seed', getattr(self, 'seed', None))
            previous_text = kwargs.get('previous_text')

            # Make API call
            response = self.client.text_to_speech.convert(
                voice_id=voice_id,
                text=text,
                model_id=model_id,
                voice_settings=voice_settings,
                output_format=output_format,
                seed=seed,
                previous_text=previous_text,
                optimize_streaming_latency=self.optimize_streaming_latency
            )

            # Save audio data
            if output_path is None:
                output_path = tempfile.mktemp(suffix=f".{output_format.split('_')[0]}")

            with open(output_path, 'wb') as f:
                for chunk in response:
                    f.write(chunk)

            logger.info(f"Successfully synthesized text to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return None

    def synthesize_streaming(self, text: str, **kwargs) -> Optional[Generator[bytes, None, None]]:
        """
        Synthesize text to speech with streaming

        Args:
            text: Text to synthesize
            **kwargs: Additional synthesis parameters

        Yields:
            Audio chunks as bytes
        """
        if not self.loaded or not self.client:
            logger.error("Model not loaded")
            return None

        try:
            # Prepare synthesis parameters
            voice_id = kwargs.get('voice_id', self.voice_id)
            model_id = kwargs.get('model_id', self.model_id)
            voice_settings = kwargs.get('voice_settings', self.voice_settings)
            output_format = kwargs.get('output_format', self.output_format)

            # Additional parameters
            seed = kwargs.get('seed', getattr(self, 'seed', None))
            previous_text = kwargs.get('previous_text')

            # Make streaming API call
            response = self.client.text_to_speech.stream(
                voice_id=voice_id,
                text=text,
                model_id=model_id,
                voice_settings=voice_settings,
                output_format=output_format,
                seed=seed,
                previous_text=previous_text,
                optimize_streaming_latency=self.optimize_streaming_latency
            )

            # Yield audio chunks
            if response:
                for chunk in response:
                    yield chunk

        except Exception as e:
            logger.error(f"Streaming synthesis failed: {e}")
            return None

    def synthesize_streaming_threaded(self, text: str, audio_queue, **kwargs):
        """
        Synthesize text to speech with streaming in a separate thread

        Args:
            text: Text to synthesize
            audio_queue: Queue to put audio chunks
            **kwargs: Additional synthesis parameters
        """
        def generate_worker():
            try:
                for chunk in self.synthesize_streaming(text, **kwargs):
                    if chunk:
                        audio_queue.put(chunk)
                audio_queue.put(None)  # Signal end of stream
            except Exception as e:
                logger.error(f"Threaded synthesis failed: {e}")
                audio_queue.put(None)

        thread = threading.Thread(target=generate_worker)
        thread.daemon = True
        thread.start()
        return thread

    def stream_audio_realtime(self, text_queue, audio_queue, **kwargs):
        """
        Stream audio in real-time as text becomes available

        Args:
            text_queue: Queue containing text chunks to synthesize
            audio_queue: Queue to put generated audio chunks
            **kwargs: Additional synthesis parameters
        """
        previous_text = None

        while True:
            try:
                text_chunk = text_queue.get(timeout=1.0)
                if text_chunk is None:  # End signal
                    break

                # Apply voice consistency
                consistency_params = self._apply_voice_consistency(text_chunk, previous_text)
                synthesis_kwargs = {**kwargs, **consistency_params}

                # Generate audio for this chunk
                audio_stream = self.synthesize_streaming(text_chunk, **synthesis_kwargs)
                if audio_stream:
                    for audio_chunk in audio_stream:
                        if audio_chunk:
                            audio_queue.put(audio_chunk)

                previous_text = text_chunk

            except Exception as e:
                logger.error(f"Real-time streaming error: {e}")
                break

        audio_queue.put(None)  # Signal end of audio stream

    def _synthesize_streaming(self, text_chunks: List[str], **kwargs) -> Generator[bytes, None, None]:
        """
        Internal method for streaming synthesis of multiple chunks

        Args:
            text_chunks: List of text chunks to synthesize
            **kwargs: Additional synthesis parameters

        Yields:
            Audio chunks as bytes
        """
        previous_text = None

        for i, chunk in enumerate(text_chunks):
            # Apply voice consistency
            consistency_params = self._apply_voice_consistency(chunk, previous_text)
            synthesis_kwargs = {**kwargs, **consistency_params}

            # Generate audio for this chunk
            audio_stream = self.synthesize_streaming(chunk, **synthesis_kwargs)
            if audio_stream:
                for audio_chunk in audio_stream:
                    if audio_chunk:
                        yield audio_chunk

            previous_text = chunk

    def _save_audio_data(self, audio_data: bytes, output_path: str) -> bool:
        """
        Save audio data to file

        Args:
            audio_data: Raw audio data
            output_path: Path to save the file

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(output_path, 'wb') as f:
                f.write(audio_data)
            return True
        except Exception as e:
            logger.error(f"Failed to save audio data: {e}")
            return False

    def get_available_voices(self) -> List[Dict[str, Any]]:
        """
        Get list of available voices

        Returns:
            List of voice information dictionaries
        """
        if not self.loaded or not self.client:
            logger.error("Model not loaded")
            return []

        try:
            voices_response = self.client.voices.get_all()
            voices = []

            for voice in voices_response.voices:
                voice_info = {
                    'voice_id': voice.voice_id,
                    'name': voice.name,
                    'category': voice.category,
                    'description': getattr(voice, 'description', ''),
                    'labels': getattr(voice, 'labels', {}),
                    'preview_url': getattr(voice, 'preview_url', ''),
                    'available_for_tiers': getattr(voice, 'available_for_tiers', []),
                    'settings': getattr(voice, 'settings', None)
                }
                voices.append(voice_info)

            logger.info(f"Retrieved {len(voices)} available voices")
            return voices

        except Exception as e:
            logger.error(f"Failed to get available voices: {e}")
            return []

    def set_voice_parameters(self, voice_id: str, **voice_settings):
        """
        Set voice parameters

        Args:
            voice_id: Voice ID to use
            **voice_settings: Voice settings to apply
        """
        self.voice_id = voice_id

        # Update voice settings
        if voice_settings:
            self.voice_settings = VoiceSettings(
                stability=voice_settings.get('stability', self.voice_settings.stability),
                similarity_boost=voice_settings.get('similarity_boost', self.voice_settings.similarity_boost),
                style=voice_settings.get('style', self.voice_settings.style),
                use_speaker_boost=voice_settings.get('use_speaker_boost', self.voice_settings.use_speaker_boost)
            )

        logger.info(f"Set voice to {voice_id} with settings: {voice_settings}")

    def set_streaming_parameters(self, chunk_size: int = 1024,
                                optimize_streaming_latency: int = 0,
                                chunk_length_schedule: Optional[List[int]] = None):
        """
        Set streaming parameters

        Args:
            chunk_size: Size of audio chunks
            optimize_streaming_latency: Latency optimization level (0-4)
            chunk_length_schedule: Schedule for chunk lengths
        """
        self.chunk_size = chunk_size
        self.optimize_streaming_latency = optimize_streaming_latency

        if chunk_length_schedule:
            self.streaming_chunk_length_schedule = chunk_length_schedule

        logger.info(f"Set streaming parameters: chunk_size={chunk_size}, "
                   f"optimize_latency={optimize_streaming_latency}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information

        Returns:
            Dictionary containing model information
        """
        if not self.loaded or not self.client:
            return {
                'name': 'ElevenLabs TTS',
                'loaded': False,
                'model_id': self.model_id,
                'voice_id': self.voice_id
            }

        try:
            # Get available models
            models_response = self.client.models.list()
            current_model = None

            for model in models_response.models:
                if model.model_id == self.model_id:
                    current_model = model
                    break

            return {
                'name': 'ElevenLabs TTS',
                'loaded': True,
                'model_id': self.model_id,
                'model_name': current_model.name if current_model else 'Unknown',
                'voice_id': self.voice_id,
                'voice_settings': {
                    'stability': self.voice_settings.stability,
                    'similarity_boost': self.voice_settings.similarity_boost,
                    'style': self.voice_settings.style,
                    'use_speaker_boost': self.voice_settings.use_speaker_boost
                },
                'output_format': self.output_format,
                'streaming_enabled': True,
                'api_connected': True
            }

        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {
                'name': 'ElevenLabs TTS',
                'loaded': True,
                'model_id': self.model_id,
                'voice_id': self.voice_id,
                'error': str(e)
            }

    def __str__(self) -> str:
        return f"ElevenLabsTTSModel(model_id={self.model_id}, voice_id={self.voice_id}, loaded={self.loaded})"
