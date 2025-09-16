import unittest
from EmotionDetection import emotion_detector

class TestEmotionDetector(unittest.TestCase):
    def test_emotions(self):
        self.assertEqual(emotion_detector("I am glad this happened")['dominant_emotion'], 'joy')
        self.assertEqual(emotion_detector("I am really mad about this")['dominant_emotion'], 'anger')
        self.assertEqual(emotion_detector("I feel disgusted just hearing about this")['dominant_emotion'], 'disgust')
        self.assertEqual(emotion_detector("I am so sad about this")['dominant_emotion'], 'sadness')
        self.assertEqual(emotion_detector("I am really afraid that this will happen")['dominant_emotion'], 'fear')

if __name__ == '__main__':
    unittest.main()
