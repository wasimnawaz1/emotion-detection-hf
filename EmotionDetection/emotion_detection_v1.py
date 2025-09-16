import requests
import json
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

HF_API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-emotion"
HF_HEADERS = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_TOKEN')}"}

def emotion_detector(text_to_analyze):
    """
    Uses Hugging Face Inference API for emotion detection.
    Maps Hugging Face labels into project schema: anger, disgust, fear, joy, sadness.
    """
    payload = {"inputs": text_to_analyze}

    response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload, timeout=30)

    if response.status_code != 200:
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None
        }

    result = response.json()[0]

    # Initialize with zeros
    emotions = {"anger": 0, "disgust": 0, "fear": 0, "joy": 0, "sadness": 0}

    for item in result:
        label = item["label"].lower()
        score = item["score"]

        if "anger" in label:
            emotions["anger"] = score
        elif "disgust" in label:
            emotions["disgust"] = score
        elif "fear" in label:
            emotions["fear"] = score
        elif "joy" in label:
            emotions["joy"] = score
        elif "sadness" in label:
            emotions["sadness"] = score
        elif "optimism" in label or "surprise" in label:
            # Treat optimism & surprise as joy
            emotions["joy"] = max(emotions["joy"], score)

    # Find dominant emotion
    dominant_emotion = max(emotions, key=emotions.get)
    emotions["dominant_emotion"] = dominant_emotion

    return emotions
