"""Unit tests for emotion classifier — run without model download using mocks."""
import pytest
from unittest.mock import MagicMock, patch

from core.emotion.classifier import EmotionClassifier, EmotionResult, CANONICAL_EMOTIONS
from core.emotion.detector import EmotionDetector


class TestEmotionClassifierMocked:
    """Tests with mocked HuggingFace pipeline."""

    def _make_classifier(self, mock_pipeline):
        clf = EmotionClassifier.__new__(EmotionClassifier)
        clf.model_name = "mock"
        clf.fallback_model = "mock-fallback"
        clf.cache_dir = None
        clf.device = -1
        clf._pipeline = mock_pipeline
        clf._label_map = {"joy": "happy", "sadness": "sad", "anger": "angry",
                           "fear": "fear", "surprise": "surprise", "neutral": "neutral",
                           "disgust": "serious"}
        clf._loaded_model = "mock"
        return clf

    def test_classify_happy(self):
        mock_pipe = MagicMock(return_value=[[
            {"label": "joy", "score": 0.92},
            {"label": "neutral", "score": 0.05},
            {"label": "sadness", "score": 0.03},
        ]])
        clf = self._make_classifier(mock_pipe)
        result = clf.classify("I am so happy today!")
        assert result.emotion == "happy"
        assert result.intensity > 0.8

    def test_classify_sad(self):
        mock_pipe = MagicMock(return_value=[[
            {"label": "sadness", "score": 0.88},
            {"label": "neutral", "score": 0.08},
            {"label": "joy", "score": 0.04},
        ]])
        clf = self._make_classifier(mock_pipe)
        result = clf.classify("I miss my family so much.")
        assert result.emotion == "sad"

    def test_classify_empty_text(self):
        mock_pipe = MagicMock()
        clf = self._make_classifier(mock_pipe)
        result = clf.classify("")
        assert result.emotion == "neutral"
        mock_pipe.assert_not_called()

    def test_classify_whitespace_only(self):
        mock_pipe = MagicMock()
        clf = self._make_classifier(mock_pipe)
        result = clf.classify("   ")
        assert result.emotion == "neutral"

    def test_result_fields(self):
        mock_pipe = MagicMock(return_value=[[
            {"label": "joy", "score": 0.9},
            {"label": "neutral", "score": 0.1},
        ]])
        clf = self._make_classifier(mock_pipe)
        result = clf.classify("test")
        assert isinstance(result, EmotionResult)
        assert 0.0 <= result.intensity <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.scores, dict)


class TestEmotionDetector:
    def _make_detector(self, emotion="neutral", intensity=0.5):
        mock_result = EmotionResult(
            emotion=emotion, intensity=intensity, confidence=intensity,
            scores={emotion: intensity}, raw_label=emotion
        )
        mock_clf = MagicMock()
        mock_clf.classify.return_value = mock_result
        mock_clf.classify_multilingual.return_value = mock_result
        return EmotionDetector(classifier=mock_clf)

    def test_detect_basic(self):
        detector = self._make_detector("happy", 0.9)
        result = detector.detect("I won!")
        assert result.emotion == "happy"

    def test_punctuation_exclamation_boost(self):
        """Exclamation marks should boost 'excited' score."""
        mock_result = EmotionResult(
            emotion="neutral", intensity=0.5, confidence=0.5,
            scores={"neutral": 0.5, "excited": 0.1}, raw_label="neutral"
        )
        mock_clf = MagicMock()
        mock_clf.classify.return_value = mock_result
        detector = EmotionDetector(classifier=mock_clf)
        result = detector.detect("This is amazing!!!")
        # excited should be boosted by exclamation marks
        assert result.scores.get("excited", 0) > mock_result.scores.get("excited", 0)

    def test_caching(self):
        mock_clf = MagicMock()
        mock_clf.classify.return_value = EmotionResult(
            "neutral", 0.5, 0.5, {"neutral": 0.5}, "neutral"
        )
        detector = EmotionDetector(classifier=mock_clf)

        detector.detect("hello world")
        detector.detect("hello world")
        # Second call should use cache, classifier called only once
        assert mock_clf.classify.call_count == 1

    def test_detect_sentences(self):
        mock_clf = MagicMock()
        mock_clf.classify.return_value = EmotionResult(
            "neutral", 0.5, 0.5, {"neutral": 0.5}, "neutral"
        )
        detector = EmotionDetector(classifier=mock_clf)
        results = detector.detect_sentences("Hello. How are you? I am fine!")
        assert len(results) >= 2

    def test_override_emotion(self):
        detector = self._make_detector("neutral", 0.5)
        result = detector.detect_with_override("some text", emotion_override="angry", intensity_override=0.9)
        assert result.emotion == "angry"
        assert result.intensity == 0.9
