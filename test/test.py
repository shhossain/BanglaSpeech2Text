import unittest
import requests
import os
from pydub import AudioSegment
import io
from speech_recognition import AudioData

current_dir = os.path.dirname(os.path.realpath(__file__))
TEST_WAV = os.path.join(current_dir, "test.wav")
TEST_WAV_TEXT = "চলে যেতে বাদ্ধা আমে।"

TEST_WAV_2 = os.path.join(current_dir, "test2.wav")
TEST_WAV_TEXT_2 = "এইয়া দলাকিনের আন্দালি বিপর্ষা থেলে দলেখিন দেয়ে অনেক ব্যাংক্ষাটে পরেকেছে। এলে কারণ তাবসে কেল। আর করে আর নসে কেনা তেখাসন কর্তে ক্রান টেছে। একন আজি করিগণে তিনি তর্ন কর্তেছে আলোগ জমাজ্যাতা রড়তে হাই এই নেতের করেছেছে তিনিন।"

import sys

previous_path = os.path.abspath(os.path.dirname(current_dir))
sys.path.append(previous_path)

from banglaspeech2text import Speech2Text


def string_match_with_percentage(str1, str2, percentage):
    if not (0 <= percentage <= 100):
        raise ValueError("Percentage should be between 0 and 100.")

    str1 = str1.replace("\n", "").strip()
    str2 = str2.replace("\n", "").strip()

    min_length = min(len(str1), len(str2))
    matching_chars = sum(c1 == c2 for c1, c2 in zip(str1, str2))
    actual_percentage = (matching_chars / min_length) * 100

    return actual_percentage >= percentage


class TestBanglaSpeech2Text(unittest.TestCase):
    """Tests for `bangla_speech2text` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        self.speech2text = Speech2Text(model="tiny")

    def tearDown(self):
        """Tear down test fixtures, if any."""
        # shutil.rmtree(self.speech2text.cache_path)
        pass

    def test_is_model_available(self):
        urls = []
        models = self.speech2text.list_models().models
        for k in models.keys():
            for model in models[k]:
                urls.append(model["url"])

        for url in urls:
            r = requests.get(url)
            self.assertEqual(r.status_code, 200)

    def test_with_file(self):
        # test with file path
        text = self.speech2text(TEST_WAV)
        self.assertTrue(string_match_with_percentage(text, TEST_WAV_TEXT, 60))

    def test_with_audio_segment(self):
        # test with audio segment
        audio = AudioSegment.from_wav(TEST_WAV)
        text = self.speech2text(audio)
        self.assertTrue(string_match_with_percentage(text, TEST_WAV_TEXT, 60))

    def test_with_audio_data(self):
        # test with audio data
        with open(TEST_WAV, "rb") as f:
            audio = AudioData(f.read(), 16000, 2)
        text = self.speech2text(audio)
        self.assertTrue(string_match_with_percentage(text, TEST_WAV_TEXT, 60))

    def test_with_io(self):
        # test with io
        with open(TEST_WAV, "rb") as f:
            audio = io.BytesIO(f.read())
        text = self.speech2text(audio)
        self.assertTrue(string_match_with_percentage(text, TEST_WAV_TEXT, 60))

    def test_with_bytes(self):
        # test with bytes
        with open(TEST_WAV, "rb") as f:
            audio = f.read()
        text = self.speech2text(audio)
        self.assertTrue(string_match_with_percentage(text, TEST_WAV_TEXT, 60))

    def test_is_long_audio_working(self):
        """Test Bangla Speech2Text"""

        text = ""
        for r in self.speech2text.recognize(TEST_WAV_2):
            text += r

        self.assertTrue(string_match_with_percentage(text, TEST_WAV_TEXT_2, 10))


if __name__ == "__main__":
    unittest.main()
