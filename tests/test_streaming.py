#!/usr/bin/env python3
"""
Simple interactive streaming test for DIA TTS.
Demonstrates true real-time streaming with immediate playback.
"""

import os
import sys
import threading
import queue
import subprocess
import tempfile
import re
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from transformers import AutoProcessor, DiaForConditionalGeneration
import torch

class SimpleStreamer:
    def __init__(self):
        self.model_name = "nari-labs/Dia-1.6B-0626"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        self.audio_queue = queue.Queue(maxsize=2)

    def load(self):
        """Load the DIA model."""
        print(f"Loading DIA model on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = DiaForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        print("✅ Model loaded!")

    def split_text(self, text, max_words=25):
        """Split text into chunks for streaming."""
        lines = text.strip().splitlines()
        chunks = []
        current_chunk = []
        word_count = 0
        last_speaker = None

        for line in lines:
            if not line.strip():
                continue

            # Check for speaker change
            speaker_match = re.match(r'^\s*\[([^\]]+)\]', line)
            current_speaker = speaker_match.group(1) if speaker_match else last_speaker

            # Split on speaker change OR when max_words reached
            if (word_count >= max_words and speaker_match) or (speaker_match and current_speaker != last_speaker and word_count > 10):
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
                    # Add speaker context for continuation
                    if last_speaker and current_speaker != last_speaker:
                        current_chunk.append(f"[{last_speaker}]")
                word_count = 0

            current_chunk.append(line)
            word_count += len(line.split())

            if speaker_match:
                last_speaker = current_speaker

            # Force split at sentence boundary if chunk gets too long
            if word_count >= max_words * 1.5 and re.search(r'[.!?]\s*$', line):
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
                    if last_speaker:
                        current_chunk.append(f"[{last_speaker}]")
                    word_count = 0

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return [c for c in chunks if c.strip()]

    def generate_chunk(self, text_chunk):
        """Generate audio for one chunk."""
        # Add speaker tag if missing
        if not text_chunk.startswith("[S"):
            text_chunk = f"[S1] {text_chunk}"

        # Process and generate
        inputs = self.processor(text=[text_chunk], padding=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=3072,
                guidance_scale=3.0,
                temperature=1.8,
                top_p=0.90,
                top_k=45
            )

        # Decode and save
        decoded = self.processor.batch_decode(outputs)
        if decoded:
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_path = temp_file.name
            temp_file.close()

            self.processor.save_audio([decoded[0]], temp_path)
            return temp_path
        return None

    def play_audio(self, file_path):
        """Play audio file immediately."""
        try:
            subprocess.run(["aplay", file_path], check=True, capture_output=True)
            os.unlink(file_path)  # Delete after playing
            return True
        except:
            return False

    def generation_worker(self, chunks):
        """Background worker to generate audio chunks."""
        for i, chunk in enumerate(chunks):
            print(f"🔄 Generating chunk {i+1}/{len(chunks)}")
            audio_file = self.generate_chunk(chunk)
            if audio_file:
                self.audio_queue.put(('audio', audio_file))
            else:
                self.audio_queue.put(('error', f"Failed chunk {i+1}"))
        self.audio_queue.put(('done', None))

    def stream_text(self, text):
        """Stream text with real-time playback."""
        # Split into chunks
        chunks = self.split_text(text)
        print(f"📊 Split into {len(chunks)} chunks")

        # Start background generation
        generation_thread = threading.Thread(
            target=self.generation_worker,
            args=(chunks,),
            daemon=True
        )
        generation_thread.start()

        # Play chunks as they become available
        chunks_played = 0
        total_chunks = len(chunks)

        print("🎵 Starting real-time streaming...")

        while chunks_played < total_chunks:
            try:
                status, data = self.audio_queue.get(timeout=60)

                if status == 'audio':
                    print(f"🔊 Playing chunk {chunks_played + 1}/{total_chunks}")
                    if self.play_audio(data):
                        chunks_played += 1
                        print(f"✅ Chunk {chunks_played} played")
                    else:
                        print(f"❌ Failed to play chunk")

                elif status == 'error':
                    print(f"❌ {data}")

                elif status == 'done':
                    break

            except queue.Empty:
                print("⚠️ Timeout waiting for audio")
                break

        print(f"🎉 Streaming complete! {chunks_played}/{total_chunks} chunks played")

def main():
    print("🎵 Simple DIA Streaming Test")
    print("=" * 40)

    streamer = SimpleStreamer()
    streamer.load()

    # Test with longer dialogue to ensure multiple chunks
    test_text = """
    [S1] Welcome to our comprehensive streaming demonstration! This is a much longer piece of text designed specifically to test the real-time streaming capabilities of our text-to-speech system.
    [S2] That's absolutely fantastic! I'm the second speaker, and I want to contribute a substantial amount of dialogue to ensure we get multiple chunks during the streaming process. This should definitely create several audio segments.
    [S1] Excellent point! The beauty of this streaming approach is that you should hear my voice immediately, without waiting for the entire conversation to be processed. Each chunk generates and plays in real-time.
    [S2] I completely agree! The audio quality should remain pristine since we're not concatenating files together. Instead, each chunk plays directly from the model's output, maintaining the original fidelity.
    [S1] Furthermore, this approach allows for much better responsiveness in interactive applications. Users can start hearing the response immediately rather than waiting for complete generation.
    [S2] And the speaker consistency is maintained throughout each chunk, creating a natural flowing conversation experience that feels seamless and professional.
    """

    print("Test text:")
    print(test_text)
    print()

    streamer.stream_text(test_text)

    # Interactive mode
    print("\n" + "=" * 40)
    print("Interactive mode - enter text to stream:")
    print("(Type 'quit' to exit)")

    while True:
        try:
            user_input = input("\n💬 Text: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            if user_input:
                streamer.stream_text(user_input)
        except KeyboardInterrupt:
            break

    print("\n👋 Done!")

if __name__ == "__main__":
    main()
