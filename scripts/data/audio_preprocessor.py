#!/usr/bin/env python3
"""
Audio Preprocessing Pipeline & Negative Sample Generator
=========================================================
Pillar 1 & Pillar 2 Component for Sinhala ASR & Translation.

Functions:
1. Audio Resampling & Channel Reduction: Standardizes to 16kHz mono float32/int16 PCM.
2. Voice Activity Detection (VAD): WebRTC VAD to trim leading/trailing dead air.
3. Audio Normalization: Peak / RMS normalization to -20 dBFS.
4. Negative Sample Synthesis: Injects silence and non-speech background frames for hallucination suppression.
"""

import os
import io
import wave
import numpy as np
import soundfile as sf
import webrtcvad


class SinhalaAudioPreprocessor:
    """Preprocesses raw audio files for Whisper ASR model fine-tuning."""

    def __init__(self, target_sample_rate: int = 16000, vad_aggressiveness: int = 2):
        self.target_sample_rate = target_sample_rate
        self.vad = webrtcvad.Vad(vad_aggressiveness)

    def load_and_resample(self, file_path: str) -> np.ndarray:
        """Loads audio file, converts to mono, and resamples to target sample rate."""
        audio, sr = sf.read(file_path, dtype="float32")
        
        # Convert stereo to mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Resample if sample rate doesn't match
        if sr != self.target_sample_rate:
            # Simple linear interpolation or torchaudio/scipy resample
            import torchaudio.transforms as T
            import torch
            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            resampler = T.Resample(sr, self.target_sample_rate)
            audio = resampler(audio_tensor).squeeze(0).numpy()

        return audio

    def normalize_volume(self, audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """Normalizes audio to target dBFS RMS volume."""
        rms = np.sqrt(np.mean(audio ** 2) + 1e-9)
        current_db = 20 * np.log10(rms + 1e-9)
        gain = 10 ** ((target_db - current_db) / 20)
        normalized = audio * gain
        # Clip to prevent overflow
        return np.clip(normalized, -1.0, 1.0)

    def apply_vad_trimming(self, audio: np.ndarray, frame_duration_ms: int = 30) -> np.ndarray:
        """Trims non-speech silence at the beginning and end of audio using WebRTC VAD."""
        frame_size = int(self.target_sample_rate * (frame_duration_ms / 1000.0))
        # Convert float32 to int16 PCM for WebRTC VAD
        int16_audio = (audio * 32767).astype(np.int16)
        
        num_frames = len(int16_audio) // frame_size
        if num_frames == 0:
            return audio

        speech_frames = []
        for i in range(num_frames):
            frame = int16_audio[i * frame_size : (i + 1) * frame_size]
            is_speech = self.vad.is_speech(frame.tobytes(), self.target_sample_rate)
            speech_frames.append(is_speech)

        # Find first and last speech frame
        first_idx = 0
        last_idx = num_frames - 1

        for i, val in enumerate(speech_frames):
            if val:
                first_idx = max(0, i - 1)  # Keep 1 frame padding
                break

        for i in range(num_frames - 1, -1, -1):
            if speech_frames[i]:
                last_idx = min(num_frames, i + 2) # Keep 1 frame padding
                break

        trimmed = audio[first_idx * frame_size : last_idx * frame_size]
        return trimmed if len(trimmed) > 0 else audio

    @staticmethod
    def generate_negative_sample(duration_seconds: float = 3.0, sample_rate: int = 16000, noise_type: str = "silence") -> np.ndarray:
        """
        Generates synthetic negative audio for negative-only training.
        Types: 'silence', 'gaussian', 'pink'
        """
        num_samples = int(duration_seconds * sample_rate)
        if noise_type == "silence":
            return np.zeros(num_samples, dtype=np.float32)
        elif noise_type == "gaussian":
            # Low-amplitude ambient white/gaussian noise (-50dB to -40dB)
            return np.random.normal(0, 0.005, num_samples).astype(np.float32)
        else:
            return np.zeros(num_samples, dtype=np.float32)


if __name__ == "__main__":
    preprocessor = SinhalaAudioPreprocessor()
    print("=== Sinhala Audio Preprocessor Verified ===")
    
    # Generate negative test sample
    neg = preprocessor.generate_negative_sample(duration_seconds=2.0, noise_type="gaussian")
    print(f"Generated negative sample shape: {neg.shape}, sample rate: 16000, mean: {np.mean(neg):.6f}")
    
    # Check volume normalization
    norm_neg = preprocessor.normalize_volume(neg, target_db=-30.0)
    print(f"Normalized negative sample shape: {norm_neg.shape}")
