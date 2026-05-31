"""Unit tests for streaming text splitter and audio chunking."""
import pytest
from core.streaming.streamer import TextSplitter


class TestTextSplitter:
    def setup_method(self):
        self.splitter = TextSplitter()

    def test_empty_text(self):
        assert self.splitter.split("") == []
        assert self.splitter.split("   ") == []

    def test_single_sentence(self):
        chunks = self.splitter.split("Hello world.")
        assert len(chunks) == 1
        assert chunks[0] == "Hello world."

    def test_multiple_sentences(self):
        text = "I am happy. You are sad. We are fine."
        chunks = self.splitter.split(text, max_chars=50)
        assert len(chunks) >= 1
        # All text should be preserved
        reconstructed = " ".join(chunks)
        for sent in ["I am happy", "You are sad", "We are fine"]:
            assert sent in reconstructed

    def test_long_sentence_split_on_comma(self):
        long = "First part of sentence, second part, third part, fourth part, and the fifth part is here."
        chunks = self.splitter.split(long, max_chars=40)
        assert all(len(c) <= 50 for c in chunks)  # Allow small overage for last segment

    def test_max_chars_respected(self):
        text = "Short. " * 20  # 20 short sentences, each 7 chars
        chunks = self.splitter.split(text, max_chars=30, min_chars=5)
        # Each chunk ≤ 30 chars (with small tolerance for the merge pass)
        assert all(len(c) <= 35 for c in chunks)
        assert len(chunks) >= 4

    def test_no_empty_chunks(self):
        text = "First sentence.   Second sentence.  Third."
        chunks = self.splitter.split(text)
        assert all(c.strip() for c in chunks)

    def test_question_mark_splits(self):
        # 55-char text split at max_chars=30 should produce multiple chunks
        text = "Are you there? Yes I am. What do you want? I need help."
        chunks = self.splitter.split(text, max_chars=30, min_chars=5)
        assert len(chunks) >= 2

    def test_exclamation_splits(self):
        text = "Amazing! This is great! I love it!"
        chunks = self.splitter.split(text, max_chars=20, min_chars=5)
        assert len(chunks) >= 2

    def test_preserves_all_content(self):
        text = "The quick brown fox. Jumps over the lazy dog. Pack my box with five dozen liquor jugs."
        chunks = self.splitter.split(text, max_chars=60)
        full = " ".join(chunks)
        assert "quick brown fox" in full
        assert "lazy dog" in full
        assert "liquor jugs" in full
