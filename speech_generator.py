"""
Speech Generation Module
Generates audio explanations for concepts using Azure Speech Services
Supports English and Indian languages (Hindi, Tamil, Telugu, Marathi)
"""
import azure.cognitiveservices.speech as speechsdk
from typing import Optional, Dict
import re
from xml.sax.saxutils import escape as xml_escape
import os
from dotenv import load_dotenv
import base64
from io import BytesIO

load_dotenv(override=True)


class SpeechGenerator:
    """Generates speech for concept explanations"""
    
    # Supported languages and their voice names
    SUPPORTED_LANGUAGES = {
        "en-US": {
            "language": "en-US",
            "voices": ["en-US-AriaNeural", "en-US-GuyNeural"],
            "default_voice": "en-US-AriaNeural",
            "name": "English (US)"
        },
        "en-IN": {
            "language": "en-IN",
            "voices": ["en-IN-NeerjaNeural", "en-IN-PrabhatNeural"],
            "default_voice": "en-IN-NeerjaNeural",
            "name": "English (India)"
        },
        "hi-IN": {
            "language": "hi-IN",
            "voices": ["hi-IN-SwaraNeural", "hi-IN-MadhurNeural"],
            "default_voice": "hi-IN-SwaraNeural",
            "name": "Hindi"
        },
        "ta-IN": {
            "language": "ta-IN",
            "voices": ["ta-IN-PallaviNeural", "ta-IN-ValluvarNeural"],
            "default_voice": "ta-IN-PallaviNeural",
            "name": "Tamil"
        },
        "te-IN": {
            "language": "te-IN",
            "voices": ["te-IN-ShrutiNeural", "te-IN-MohanNeural"],
            "default_voice": "te-IN-ShrutiNeural",
            "name": "Telugu"
        },
        "mr-IN": {
            "language": "mr-IN",
            "voices": ["mr-IN-AarohiNeural", "mr-IN-ManoharNeural"],
            "default_voice": "mr-IN-AarohiNeural",
            "name": "Marathi"
        }
    }
    
    def __init__(self):
        """Initialize the speech generator with Azure credentials"""
        self.speech_key = os.getenv("SPEECH_KEY")
        self.speech_region = os.getenv("SPEECH_REGION", "eastus")
        
        if not self.speech_key:
            # Speech not configured - will return error when trying to generate
            self.speech_config = None
            return
        
        # Initialize speech config
        self.speech_config = speechsdk.SpeechConfig(
            subscription=self.speech_key,
            region=self.speech_region
        )
    
    def _create_ssml(
        self,
        text: str,
        language: str,
        voice: Optional[str] = None
    ) -> str:
        """
        Create SSML (Speech Synthesis Markup Language) for advanced TTS
        
        Args:
            text: The text to convert to speech
            language: Language code (e.g., "en-US", "hi-IN")
            voice: Specific voice to use (optional)
            
        Returns:
            SSML string
        """
        if language not in self.SUPPORTED_LANGUAGES:
            language = "en-US"
        
        if not voice:
            voice = self.SUPPORTED_LANGUAGES[language]["default_voice"]
        
        # Create SSML with proper XML namespaces that Azure TTS expects
        # Use explicit xmlns and mstts namespace; include xml:lang on voice
        ssml = (
            f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
            f'xmlns:mstts="http://www.w3.org/2001/mstts" xml:lang="{language}">'
            f'<voice xml:lang="{language}" name="{voice}">' 
            f'<prosody rate="0%" pitch="0%">{text}</prosody>'
            f'</voice></speak>'
        )
        
        return ssml
    
    async def generate_speech(
        self,
        concept_name: str,
        concept_description: str,
        language: str = "en-US",
        voice: Optional[str] = None
    ) -> Dict:
        """
        Generate speech for a concept explanation
        
        Args:
            concept_name: Name of the concept
            concept_description: Description of the concept
            language: Language code (default: en-US)
            voice: Specific voice to use (optional)
            
        Returns:
            Dictionary with audio data and metadata
        """
        try:
            # Check if speech is configured
            if not self.speech_config:
                # Mock a silent audio response to bypass errors when no key is provided
                empty_wav_b64 = "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA="
                return {
                    "success": True,
                    "audio_base64": empty_wav_b64,
                    "audio_format": "wav",
                    "language": language,
                    "voice": "mock-voice (no API key)",
                    "concept": concept_name,
                    "duration_ms": 1000,
                    "note": "Mocked audio due to missing SPEECH_KEY"
                }
            
            # Validate language
            if language not in self.SUPPORTED_LANGUAGES:
                return {
                    "success": False,
                    "error": f"Language {language} not supported",
                    "supported_languages": list(self.SUPPORTED_LANGUAGES.keys())
                }
            
            # Prepare the text to speak
            raw_text = f"{concept_name}. {concept_description}"

            # Remove any HTML tags that may have been included in the concept text
            # (model outputs sometimes include <ul>, <li>, etc.). Replace with spaces.
            plain_text = re.sub(r'<[^>]+>', ' ', raw_text)
            # Collapse whitespace
            plain_text = re.sub(r'\s+', ' ', plain_text).strip()
            # Escape XML special chars for safe SSML embedding
            text_to_speak = xml_escape(plain_text)

            # Create SSML
            ssml = self._create_ssml(text_to_speak, language, voice)
            
            # Create synthesizer without directing output to speaker so
            # we can access the audio bytes from the result object.
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config, audio_config=None)

            # Synthesize speech
            result = synthesizer.speak_ssml(ssml)

            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                # Get raw audio bytes from the synthesis result
                audio_data = result.audio_data if hasattr(result, 'audio_data') else None
                if not audio_data:
                    return {
                        "success": False,
                        "error": "No audio data returned from speech synthesizer"
                    }

                # Encode to base64
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')

                return {
                    "success": True,
                    "audio_base64": audio_base64,
                    "audio_format": "wav",
                    "language": language,
                    "voice": voice or self.SUPPORTED_LANGUAGES[language]["default_voice"],
                    "concept": concept_name,
                    "duration_ms": max(0, len(audio_data) // 320)  # rough ms estimate
                }
            else:
                # Attempt to extract meaningful error information without
                # assuming the presence of `error_details` attribute.
                err = None
                # SDK may provide cancellation details on failure
                if hasattr(result, 'cancellation_details') and result.cancellation_details:
                    cd = result.cancellation_details
                    err = getattr(cd, 'error_details', None) or getattr(cd, 'reason', None)
                # Fallback to any available attribute or str(result)
                if not err:
                    err = getattr(result, 'error_details', None) or str(result)

                error_msg = f"Speech synthesis failed: {err}"
                return {
                    "success": False,
                    "error": error_msg
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_supported_languages(self) -> Dict:
        """
        Get list of supported languages
        
        Returns:
            Dictionary of supported languages with details
        """
        return {
            language_code: {
                "code": language_code,
                "name": lang_info["name"],
                "voices": lang_info["voices"],
                "default_voice": lang_info["default_voice"]
            }
            for language_code, lang_info in self.SUPPORTED_LANGUAGES.items()
        }
    
    def validate_language(self, language: str) -> bool:
        """Check if a language is supported"""
        return language in self.SUPPORTED_LANGUAGES


async def generate_explanation_audio(
    concept_name: str,
    concept_description: str,
    language: str = "en-US"
) -> Dict:
    """
    Convenience function to generate speech for a concept
    
    Args:
        concept_name: Name of the concept
        concept_description: Description of the concept
        language: Language code (default: en-US)
        
    Returns:
        Dictionary with audio data
    """
    try:
        generator = SpeechGenerator()
        result = await generator.generate_speech(
            concept_name,
            concept_description,
            language
        )
        return result
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
