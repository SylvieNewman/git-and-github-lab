"""Step 1: Classify one review as POSITIVE or NEGATIVE.

This file sends a review title and text to the course's model. It does not
read the Amazon data or use a star rating.
"""

import json
from urllib import request
import gzip
from pathlib import Path


BASE_URL = "http://dobolyi.com:9000/v1"
API_KEY = "6418"
MODEL_NAME = "DeepSeek-V4-Flash-0731"


def build_sentiment_prompt(title, text):
    """Return instructions for an LLM to label one Amazon review.

    Parameters:
        title: The short title written by the reviewer.
        text: The main body of the review.
    """
    return f"""
You classify the sentiment of an Amazon review.

Use the review title and review text. Do not use or guess a star rating.

Choose POSITIVE when the reviewer is mainly satisfied, says the product is
sufficient, recommends it, or describes a good experience.
Choose NEGATIVE when the reviewer is mainly dissatisfied, says the product is
insufficient, complains about a problem, or describes a bad experience.

Read the title and text together. If they conflict, the overall meaning of the
text is more important than the title:
- A negative title followed by text saying that the product was sufficient or
  satisfactory is POSITIVE.
- A positive title followed by text that is mainly unsatisfactory is NEGATIVE.

For a very short review with little or no text, an optimistic title is usually
POSITIVE. However, short negative titles or short negative text, such as
"Terrible" or "Did not work", are NEGATIVE.

Also choose one primary emotion from this list:
anger, anticipation, disgust, fear, joy, sadness, surprise, trust.

Reply with JSON only, using this exact format:

{{"sentiment": "POSITIVE", "emotion": "joy"}}

Review title: {title}
Review text: {text}
""".strip()


def classify_review(title, text):
    """Ask the model to classify one review.

    The function returns only "POSITIVE" or "NEGATIVE".
    """
    request_body = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": build_sentiment_prompt(title, text)}
        ],
        "temperature": 0,
    }

    # Turn the Python dictionary into JSON before sending it to the model.
    request_data = json.dumps(request_body).encode("utf-8")
    model_request = request.Request(
        f"{BASE_URL}/chat/completions",
        data=request_data,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with request.urlopen(model_request) as response:
        response_data = json.loads(response.read().decode("utf-8"))

    answer = response_data["choices"][0]["message"]["content"].strip()

    classification = json.loads(answer)

    predicted_sentiment = classification["sentiment"].upper()
    llm_emotion = classification["emotion"].lower()

    valid_sentiments = ["POSITIVE", "NEGATIVE"]
    valid_emotions = [
        "anger",
        "anticipation",
        "disgust",
        "fear",
        "joy",
        "sadness",
        "surprise",
        "trust",
    ]

    if predicted_sentiment not in valid_sentiments:
        raise ValueError(f"Unexpected sentiment: {predicted_sentiment}")

    if llm_emotion not in valid_emotions:
        raise ValueError(f"Unexpected emotion: {llm_emotion}")

    return predicted_sentiment, llm_emotion

SCRIPT_FOLDER = Path(__file__).parent
DATA_FILE = SCRIPT_FOLDER / "Gift_Cards.jsonl.gz"
OUTPUT_FILE = SCRIPT_FOLDER / "first_100_predictions.jsonl"
NUMBER_OF_REVIEWS = 100


def classify_first_100_reviews():
    """Classify 100 reviews using only title and text."""

    with gzip.open(DATA_FILE, "rt", encoding="utf-8") as data_file:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as output_file:
            for review_number, line in enumerate(data_file, start=1):
                review = json.loads(line)

                # We do not give the model the star rating.
                title = review.get("title") or ""
                text = review.get("text") or ""

                predicted_sentiment, llm_emotion = classify_review(title, text)
                rating = review.get("rating")

                result = {
                    "review_number": review_number,
                    "title": title,
                    "text": text,
                    "rating": rating,
                    "llm_emotion": llm_emotion,
                    "predicted_sentiment": predicted_sentiment,
                }

                output_file.write(json.dumps(result) + "\n")

                print(
    f"Review {review_number}: "
    f"{predicted_sentiment} | Emotion = {llm_emotion} | "
    f"Star rating = {rating} | Title: {title}",
    flush=True,
)

                if review_number == NUMBER_OF_REVIEWS:
                    break

    print("\nFinished classifying 100 reviews.")
    print("Predictions were saved in first_100_predictions.jsonl.")


if __name__ == "__main__":
    classify_first_100_reviews()