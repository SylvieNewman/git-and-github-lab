import gzip
import json
import random
from pathlib import Path
from urllib import request


SCRIPT_FOLDER = Path(__file__).parent

DATA_FILE = SCRIPT_FOLDER / "Gift_Cards.jsonl.gz"
OUTPUT_FILE = SCRIPT_FOLDER / "balanced_150_predictions.jsonl"

BASE_URL = "http://dobolyi.com:9000/v1"
API_KEY = "6418"
MODEL_NAME = "DeepSeek-V4-Flash-0731"

SAMPLE_SIZE_PER_CLASS = 50
RANDOM_SEED = 6418

EMOTIONS = [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust",
]


def get_rating_class(rating):
    """Convert a star rating into the three assignment classes."""

    if rating >= 4:
        return "POSITIVE"

    if rating == 3:
        return "NEUTRAL"

    return "NEGATIVE"


def build_sentiment_prompt(title, text):
    """Create the three-class prompt for one review."""

    return f"""
You classify the sentiment and primary emotion of an Amazon review.

Use only the review title and review text. Do not use or guess a star rating.

Choose POSITIVE when the reviewer is mainly satisfied, says the product is
sufficient, recommends it, or describes a good experience.

Choose NEGATIVE when the reviewer is mainly dissatisfied, says the product is
insufficient, complains about a problem, or describes a bad experience.

Choose NEUTRAL when the review is genuinely mixed, ambivalent, or does not
clearly lean positive or negative.

If the title and text conflict, give more weight to the detailed review text.
For a very short review with little or no text, an optimistic title is usually
POSITIVE. However, a short negative title or short negative text is NEGATIVE.

Choose one primary emotion from:
anger, anticipation, disgust, fear, joy, sadness, surprise, trust.

Reply with JSON only, using this exact format:

{{"sentiment": "NEUTRAL", "emotion": "trust"}}

Review title: {title}
Review text: {text}
""".strip()


def classify_review(title, text):
    """Ask the model for a sentiment class and primary emotion."""

    request_body = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": build_sentiment_prompt(title, text),
            }
        ],
        "temperature": 0,
    }

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

    valid_sentiments = ["POSITIVE", "NEUTRAL", "NEGATIVE"]

    if predicted_sentiment not in valid_sentiments:
        raise ValueError(f"Unexpected sentiment: {predicted_sentiment}")

    if llm_emotion not in EMOTIONS:
        raise ValueError(f"Unexpected emotion: {llm_emotion}")

    return predicted_sentiment, llm_emotion


def select_balanced_sample():
    """Use reservoir sampling to select 50 reviews from each class."""

    random_generator = random.Random(RANDOM_SEED)

    samples = {
        "POSITIVE": [],
        "NEUTRAL": [],
        "NEGATIVE": [],
    }

    reviews_seen = {
        "POSITIVE": 0,
        "NEUTRAL": 0,
        "NEGATIVE": 0,
    }

    with gzip.open(DATA_FILE, "rt", encoding="utf-8") as data_file:
        for line in data_file:
            review = json.loads(line)

            rating = float(review["rating"])
            rating_class = get_rating_class(rating)

            reviews_seen[rating_class] += 1
            sample = samples[rating_class]

            if len(sample) < SAMPLE_SIZE_PER_CLASS:
                sample.append(review)
            else:
                replacement_position = random_generator.randrange(
                    reviews_seen[rating_class]
                )

                if replacement_position < SAMPLE_SIZE_PER_CLASS:
                    sample[replacement_position] = review

    for rating_class, sample in samples.items():
        if len(sample) < SAMPLE_SIZE_PER_CLASS:
            raise ValueError(
                f"Not enough {rating_class} reviews were found."
            )

    balanced_reviews = (
        samples["POSITIVE"]
        + samples["NEUTRAL"]
        + samples["NEGATIVE"]
    )

    random_generator.shuffle(balanced_reviews)

    return balanced_reviews


def run_balanced_classification():
    """Classify the 150 balanced reviews and save the results."""

    balanced_reviews = select_balanced_sample()
    correct_predictions = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as output_file:
        for review_number, review in enumerate(balanced_reviews, start=1):
            title = review.get("title") or ""
            text = review.get("text") or ""
            rating = float(review["rating"])
            rating_class = get_rating_class(rating)

            # The model receives only title and text.
            predicted_sentiment, llm_emotion = classify_review(title, text)

            is_correct = predicted_sentiment == rating_class

            if is_correct:
                correct_predictions += 1

            result = {
                "review_number": review_number,
                "title": title,
                "text": text,
                "rating": rating,
                "rating_class": rating_class,
                "predicted_sentiment": predicted_sentiment,
                "llm_emotion": llm_emotion,
                "is_correct": is_correct,
            }

            output_file.write(json.dumps(result) + "\n")

            print(
                f"Review {review_number}/150: "
                f"Expected = {rating_class} | "
                f"Model = {predicted_sentiment} | "
                f"Emotion = {llm_emotion}",
                flush=True,
            )

    accuracy = correct_predictions / len(balanced_reviews) * 100

    print(f"\nBalanced-run agreement: {accuracy:.1f}%")
    print(f"Saved results in: {OUTPUT_FILE.name}")


if __name__ == "__main__":
    run_balanced_classification()