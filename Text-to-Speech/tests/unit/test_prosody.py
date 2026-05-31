"""Unit tests for prosody prediction."""
import pytest
from core.emotion.classifier import EmotionResult
from core.prosody.predictor import ProsodyPredictor, RULE_BASED_MAP
from core.prosody.prosody_model import ProsodyTarget


def _make_result(emotion: str, intensity: float = 0.8) -> EmotionResult:
    return EmotionResult(
        emotion=emotion, intensity=intensity, confidence=intensity,
        scores={emotion: intensity}, raw_label=emotion,
    )


class TestProsodyPredictor:
    def setup_method(self):
        self.predictor = ProsodyPredictor()  # No model path — rule-based

    def test_neutral_base(self):
        result = self.predictor.predict("Hello world.", _make_result("neutral", 1.0))
        assert abs(result.global_pitch - 1.0) < 0.01
        assert abs(result.global_energy - 1.0) < 0.01

    def test_excited_higher_pitch_energy(self):
        neutral = self.predictor.predict("ok", _make_result("neutral", 0.8))
        excited = self.predictor.predict("I won!", _make_result("excited", 0.8))
        assert excited.global_pitch > neutral.global_pitch
        assert excited.global_energy > neutral.global_energy
        assert excited.global_speed > neutral.global_speed

    def test_sad_lower_pitch_energy(self):
        neutral = self.predictor.predict("ok", _make_result("neutral", 0.8))
        sad = self.predictor.predict("I miss them.", _make_result("sad", 0.8))
        assert sad.global_pitch < neutral.global_pitch
        assert sad.global_energy < neutral.global_energy

    def test_intensity_zero_near_neutral(self):
        result = self.predictor.predict("excited text", _make_result("excited", 0.0))
        # At intensity 0, everything should be at 1.0 (neutral)
        assert abs(result.global_pitch - 1.0) < 0.1

    def test_all_emotions_in_range(self):
        for emotion in RULE_BASED_MAP:
            result = self.predictor.predict("test", _make_result(emotion, 0.8))
            assert 0.5 <= result.global_pitch <= 2.0, f"Pitch out of range for {emotion}"
            assert 0.4 <= result.global_energy <= 2.0, f"Energy out of range for {emotion}"
            assert 0.5 <= result.global_speed <= 2.0, f"Speed out of range for {emotion}"

    def test_language_adjustment_tamil(self):
        base = self.predictor.predict("hello", _make_result("neutral", 0.5))
        adjusted = self.predictor.adjust_for_language(base, "ta")
        assert adjusted.global_pitch > base.global_pitch
        assert adjusted.global_speed < base.global_speed

    def test_returns_prosody_target(self):
        result = self.predictor.predict("test", _make_result("happy"))
        assert isinstance(result, ProsodyTarget)
        assert isinstance(result.global_pitch, float)
        assert isinstance(result.global_energy, float)
        assert isinstance(result.to_dict(), dict)
