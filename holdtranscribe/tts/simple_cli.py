#!/usr/bin/env python3
"""
Simple CLI for Kyutai TTS that actually works.

This is a minimal implementation that focuses on reliability over features.
"""

import argparse
import sys
import os
import tempfile
import time

def main():
    parser = argparse.ArgumentParser(description="Simple Kyutai TTS CLI")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("-o", "--output", help="Output file (if not specified, plays to speakers)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--model", default="kyutai/tts-1.6b-en_fr", help="TTS model to use")

    args = parser.parse_args()

    try:
        # Import dependencies
        import torch
        from moshi.models.loaders import CheckpointInfo
        from moshi.models.tts import TTSModel, script_to_entries
        import sphn
        import numpy as np

        print(f"Loading TTS model: {args.model}")

        # Load model
        checkpoint_info = CheckpointInfo.from_hf_repo(args.model)
        tts_model = TTSModel.from_checkpoint_info(
            checkpoint_info,
            n_q=32,
            temp=0.6,
            device=args.device
        )

        # Get voice
        voice_path = tts_model.get_voice_path("expresso/ex03-ex01_happy_001_channel1_334s.wav")
        condition_attributes = tts_model.make_condition_attributes([voice_path], cfg_coef=2.0)

        print("🔊 Kyutai TTS model loaded successfully")

        # Synthesize
        print(f"Synthesizing: '{args.text[:50]}{'...' if len(args.text) > 50 else ''}'")

        pcms = []

        def on_frame(frame):
            if (frame != -1).all():
                pcm = tts_model.mimi.decode(frame[:, 1:, :]).cpu().detach().numpy()
                pcms.append(np.clip(pcm[0, 0], -1, 1))

        # Create generation wrapper
        from moshi.models.tts import TTSGen
        gen = TTSGen(tts_model, [condition_attributes], on_frame=on_frame)

        # Process text
        with tts_model.mimi.streaming(1):
            entries = script_to_entries(
                tts_model.tokenizer,
                tts_model.machine.token_ids,
                tts_model.mimi.frame_rate,
                [args.text],
                multi_speaker=False,
                padding_between=1,
            )

            for entry in entries:
                gen.append_entry(entry)
                gen.process()
            gen.process_last()

        if pcms:
            audio_data = np.concatenate(pcms, axis=-1)

            if args.output:
                # Save to file
                sphn.write_wav(args.output, audio_data, tts_model.mimi.sample_rate)
                print(f"Audio saved to: {args.output}")
            else:
                # Play to speakers
                try:
                    import sounddevice as sd
                    print("Playing audio to speakers...")
                    sd.play(audio_data, samplerate=tts_model.mimi.sample_rate)
                    sd.wait()
                    print("Playback completed")
                except ImportError:
                    print("sounddevice not available, saving to temp file instead")
                    temp_file = tempfile.mktemp(suffix=".wav")
                    sphn.write_wav(temp_file, audio_data, tts_model.mimi.sample_rate)
                    print(f"Audio saved to: {temp_file}")
        else:
            print("No audio generated")
            return 1

        return 0

    except ImportError as e:
        print(f"Missing dependencies: {e}")
        print("Install with: pip install moshi torch sphn sounddevice")
        return 1
    except Exception as e:
        print(f"TTS error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
