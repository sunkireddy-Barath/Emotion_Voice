"""Voice recording tool for dataset collection.

Usage:
    python recorder.py --output_dir ../raw --metadata_file ../metadata/recordings.txt
                       --emotion happy --num_sentences 20
"""
from __future__ import annotations

import argparse
import csv
import os
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import List, Optional

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

import numpy as np


SAMPLE_SENTENCES = {
    "neutral": [
        "Hello everyone, welcome to the platform.",
        "The meeting is scheduled for tomorrow morning.",
        "Please review the document before the deadline.",
        "I will send you the report by end of day.",
        "The system is running normally today.",
        "Could you please pass me the salt?",
        "The train arrives at platform number three.",
        "I need to buy some groceries this afternoon.",
        "The temperature outside is around twenty degrees.",
        "Please save the file before closing the application.",
    ],
    "happy": [
        "I just got promoted! This is the best day ever!",
        "We won the championship! I am so proud of everyone!",
        "My daughter took her first steps today!",
        "The project was a massive success, congratulations to all!",
        "I cannot believe how wonderful this day has been!",
        "Finally, the weekend is here and I am ready to celebrate!",
        "This is absolutely amazing news, I am thrilled!",
        "We did it! All our hard work finally paid off!",
        "What a beautiful day! Everything is going perfectly!",
        "I am so grateful for all the love and support!",
    ],
    "sad": [
        "I miss them so much, it is really hard without them.",
        "Things did not go the way I hoped they would.",
        "I have been feeling quite down lately.",
        "It is really difficult to move forward right now.",
        "I wish things could have been different between us.",
        "The news was heartbreaking, I could not stop crying.",
        "Sometimes loneliness can be overwhelming.",
        "I tried my best but it was not enough.",
        "Everything feels so empty without them around.",
        "I just need some time to heal and recover.",
    ],
    "angry": [
        "This is absolutely unacceptable! How could you do this?",
        "I am done tolerating this behavior, it needs to stop!",
        "Why does nobody ever listen to what I say?",
        "This has gone too far and I will not stand for it!",
        "I specifically told you not to do that!",
        "This is the third time this week you have been late!",
        "I cannot believe how irresponsible this situation is!",
        "Stop making excuses and take responsibility!",
        "This is a complete waste of everyone's time and resources!",
        "I demand an explanation for this mess right now!",
    ],
    "excited": [
        "Oh my goodness, I cannot believe this is happening!",
        "This is incredible! I have been waiting for this moment!",
        "We are going to the finals! Can you believe it?",
        "I just got selected for the dream job I always wanted!",
        "The concert is tonight and I am beyond excited!",
        "They finally released the sequel and it looks amazing!",
        "I am starting my new adventure tomorrow, so pumped!",
        "We are getting a puppy! This is the best surprise ever!",
        "The results are out and I topped the examination!",
        "I just discovered something that is going to change everything!",
    ],
    "fear": [
        "I heard strange noises coming from the basement.",
        "Something does not feel right about this situation.",
        "I am terrified of what might happen next.",
        "Please do not leave me alone in this darkness.",
        "I keep having these nightmares and cannot sleep.",
        "What if something terrible happens to my family?",
        "I am scared to walk alone on this street at night.",
        "The doctor's results have me extremely worried.",
        "I cannot shake this feeling that something is wrong.",
        "My hands are shaking and I cannot calm down.",
    ],
    "serious": [
        "This matter requires your immediate and full attention.",
        "We need to address this issue without any further delay.",
        "The consequences of inaction will be severe and lasting.",
        "I want to be very clear about the gravity of this situation.",
        "This decision will impact every single person in this organization.",
        "We must proceed with utmost caution and careful consideration.",
        "The evidence points to a critical flaw in our current process.",
        "I am asking everyone to take this protocol seriously.",
        "This is not a matter to be taken lightly by anyone.",
        "Our response must be measured, deliberate, and well-planned.",
    ],
    "calm": [
        "Take a deep breath and let it go slowly.",
        "Everything will work out fine in the end.",
        "There is no need to rush, we have plenty of time.",
        "Let us approach this thoughtfully and without stress.",
        "I find peace in the simplicity of everyday moments.",
        "The rain falling softly creates such a soothing atmosphere.",
        "Breathe in deeply and release all tension from your body.",
        "We are safe and there is nothing to worry about.",
        "Focus on this present moment and nothing else.",
        "Let the gentle breeze carry your worries away.",
    ],
    "motivational": [
        "You have the strength to overcome every single obstacle!",
        "Believe in yourself because you are capable of greatness!",
        "Every failure is just one step closer to your success!",
        "Rise up, push forward, and never stop fighting!",
        "Your potential is limitless if you dare to dream big!",
        "Today is the day you transform your life forever!",
        "Champions are not born, they are made through persistence!",
        "The only limit that exists is the one in your mind!",
        "Keep going because your breakthrough is just around the corner!",
        "You have come too far to give up now, stay the course!",
    ],
    "questioning": [
        "Wait, did you really mean what you just said?",
        "Are you absolutely sure that is the right approach here?",
        "What exactly do you mean by that statement?",
        "Have we considered all the possible alternatives?",
        "Is there something important that I am missing here?",
        "Why would anyone make that kind of decision?",
        "How did this situation even get to this point?",
        "Do you think this is really the best solution available?",
        "Are we heading in the right direction with this plan?",
        "What are the long-term consequences of this choice?",
    ],
    "storytelling": [
        "Long ago in a distant land there lived a wise old king.",
        "It was a dark and stormy night when everything changed forever.",
        "The children gathered around as the elder began his tale.",
        "Once upon a time a young girl discovered a magical forest.",
        "And so it began, the adventure that would define their destiny.",
        "Nobody could have predicted what would happen next in the story.",
        "The hero stood at the crossroads, facing an impossible choice.",
        "Deep in the mountains there was a secret that nobody knew.",
        "As the sun set over the horizon, their journey was just beginning.",
        "And they lived not perfectly, but bravely, ever after.",
    ],
}


class VoiceRecorder:
    def __init__(
        self,
        output_dir: Path,
        metadata_file: Path,
        sample_rate: int = 22050,
        channels: int = 1,
        chunk: int = 1024,
    ):
        self.output_dir = Path(output_dir)
        self.metadata_file = Path(metadata_file)
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk = chunk
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)

        if not PYAUDIO_AVAILABLE:
            raise RuntimeError("PyAudio not installed. Run: pip install pyaudio")

        self.audio = pyaudio.PyAudio()

    def _get_next_filename(self, emotion: str) -> str:
        existing = list(self.output_dir.glob(f"{emotion}_*.wav"))
        idx = len(existing) + 1
        return f"{emotion}_{idx:04d}.wav"

    def record(self, duration_sec: float, filename: str) -> Path:
        out_path = self.output_dir / filename
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk,
        )

        frames = []
        num_chunks = int(self.sample_rate / self.chunk * duration_sec)

        print(f"  Recording for {duration_sec:.1f}s... ", end="", flush=True)
        for _ in range(num_chunks):
            data = stream.read(self.chunk, exception_on_overflow=False)
            frames.append(data)
        print("Done.")

        stream.stop_stream()
        stream.close()

        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b"".join(frames))

        return out_path

    def append_metadata(self, filename: str, text: str, emotion: str) -> None:
        with open(self.metadata_file, "a", encoding="utf-8") as f:
            f.write(f"{filename}|{text}|{emotion}\n")

    def run_session(
        self,
        emotion: str,
        sentences: Optional[List[str]] = None,
        duration_sec: float = 5.0,
    ) -> None:
        if sentences is None:
            sentences = SAMPLE_SENTENCES.get(emotion, SAMPLE_SENTENCES["neutral"])

        print(f"\n=== Recording Session: {emotion.upper()} ===")
        print(f"  Sample rate: {self.sample_rate} Hz")
        print(f"  Duration per recording: {duration_sec}s")
        print(f"  Total sentences: {len(sentences)}\n")

        for i, sentence in enumerate(sentences, 1):
            print(f"[{i}/{len(sentences)}] Say: \"{sentence}\"")
            input("  Press ENTER when ready...")
            filename = self._get_next_filename(emotion)
            out_path = self.record(duration_sec, filename)
            self.append_metadata(filename, sentence, emotion)
            print(f"  Saved: {out_path}")
            time.sleep(0.3)

        print(f"\n✓ Session complete. {len(sentences)} recordings saved.")

    def close(self) -> None:
        self.audio.terminate()


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice dataset recorder")
    parser.add_argument("--output_dir", default="data/raw")
    parser.add_argument("--metadata_file", default="data/metadata/recordings.txt")
    parser.add_argument("--emotion", default="neutral",
                        choices=list(SAMPLE_SENTENCES.keys()))
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--num_sentences", type=int, default=None)
    args = parser.parse_args()

    recorder = VoiceRecorder(
        output_dir=Path(args.output_dir),
        metadata_file=Path(args.metadata_file),
        sample_rate=args.sample_rate,
    )

    sentences = SAMPLE_SENTENCES.get(args.emotion, [])
    if args.num_sentences:
        sentences = sentences[: args.num_sentences]

    try:
        recorder.run_session(args.emotion, sentences, args.duration)
    finally:
        recorder.close()


if __name__ == "__main__":
    main()
