import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Set HUGGINGFACE_TOKEN in .env")

# Example model ids:
# "j-hartmann/emotion-english-distilroberta-base"   <- predicts Ekman emotions
# "cardiffnlp/twitter-roberta-base-emotion"        <- tweet model (may include optimism/surprise)
# "SamLowe/roberta-base-go_emotions"               <- GoEmotions variant
DEFAULT_MODEL = "j-hartmann/emotion-english-distilroberta-base"

def map_label_to_category(label: str):
    """Map diverse HF labels to one of: anger, disgust, fear, joy, sadness (or None)."""
    L = label.lower()
    # JOY-like
    if any(k in L for k in ("joy", "happy", "happiness", "optimism", "optimistic", "amusement", "excit", "relief", "love", "pride", "admiration", "approval", "gratitude")):
        return "joy"
    if any(k in L for k in ("anger", "angry", "annoy", "annoyance", "rage", "disapproval")):
        return "anger"
    if any(k in L for k in ("disgust", "disgusted", "dislik")):
        return "disgust"
    if any(k in L for k in ("fear", "afraid", "anxious", "anxiety", "nervous")):
        return "fear"
    if any(k in L for k in ("sad", "sadness", "grief", "sorrow", "remorse", "disappointment")):
        return "sadness"
    # surprise / neutral / other -> map to joy as optional, or ignore
    if "surprise" in L:
        return "joy"   # optional mapping; you can change to None if you prefer
    # fallback
    return None

def emotion_detector(text_to_analyze: str, model_id: str = DEFAULT_MODEL):
    HF_API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": text_to_analyze}

    try:
        resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
    except Exception as e:
        # network error or timeout
        return {'anger': None, 'disgust': None, 'fear': None, 'joy': None, 'sadness': None, 'dominant_emotion': None}

    if resp.status_code != 200:
        # API error (rate limit, model not available, etc.)
        return {'anger': None, 'disgust': None, 'fear': None, 'joy': None, 'sadness': None, 'dominant_emotion': None}

    data = resp.json()

    # Normalize different HF response shapes to a list of label-score dicts
    # Common shapes:
    #  - list of dicts: [ {'label':'joy','score':0.8}, ... ]
    #  - nested: [ [ {'label':..., 'score':...}, ... ] ]
    #  - dict: { 'joy': 0.8, 'sadness': 0.01, ... }
    preds = None
    if isinstance(data, list):
        # flatten one level if needed
        if data and isinstance(data[0], list):
            preds = data[0]
        else:
            preds = data
    elif isinstance(data, dict):
        # convert dict to list
        preds = [{"label": k, "score": float(v)} for k, v in data.items()]
    else:
        # unknown format
        return {'anger': None, 'disgust': None, 'fear': None, 'joy': None, 'sadness': None, 'dominant_emotion': None}

    # Aggregate scores into our 5 target emotions
    agg = {'anger': 0.0, 'disgust': 0.0, 'fear': 0.0, 'joy': 0.0, 'sadness': 0.0}
    for p in preds:
        if not isinstance(p, dict):
            continue
        lbl = str(p.get("label", "")).lower()
        score = float(p.get("score", 0.0))
        target = map_label_to_category(lbl)
        if target:
            # accumulate (use max() instead if you prefer picking the single highest contribution)
            agg[target] += score

    # If everything is zero (no mapping), fallback: map top prediction to category
    if all(v == 0.0 for v in agg.values()) and preds:
        top = max(preds, key=lambda x: float(x.get("score", 0.0)))
        fallback = map_label_to_category(str(top.get("label", "")).lower())
        if fallback:
            agg[fallback] = float(top.get("score", 0.0))

    # Normalize (optional) - keep raw aggregated scores or normalize to sum=1
    total = sum(agg.values())
    if total > 0:
        norm = {k: (v / total) for k, v in agg.items()}
    else:
        # keep as-is (zeros) so calling code can detect
        norm = agg

    dominant = max(norm, key=norm.get) if total > 0 else None
    result = {
        'anger': norm['anger'],
        'disgust': norm['disgust'],
        'fear': norm['fear'],
        'joy': norm['joy'],
        'sadness': norm['sadness'],
        'dominant_emotion': dominant
    }
    return result
