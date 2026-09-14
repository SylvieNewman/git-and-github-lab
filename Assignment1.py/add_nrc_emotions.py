import json
import re
from collections import Counter
from pathlib import Path


SCRIPT_FOLDER = Path(__file__).parent

INPUT_FILE = SCRIPT_FOLDER / "first_100_predictions.jsonl"
LEXICON_FILE = SCRIPT_FOLDER / "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
OUTPUT_FILE = SCRIPT_FOLDER / "first_100_with_emotions.jsonl"

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


def load_nrc_lexicon():
    """Read the NRC word list and keep only the eight emotion labels."""

    lexicon = {}

    with open(LEXICON_FILE, "r", encoding="utf-8") as lexicon_file:
        for line in lexicon_file:
            word, emotion, association = line.strip().split("\t")

            if emotion in EMOTIONS and association == "1":
                if word not in lexicon:
                    lexicon[word] = []

                lexicon[word].append(emotion)

    return lexicon


def find_nrc_emotion(title, text, lexicon):
    """Score review words and return the highest-scoring NRC emotion."""

    words = re.findall(r"[a-z]+", f"{title} {text}".lower())
    scores = Counter()

    for word in words:
        for emotion in lexicon.get(word, []):
            scores[emotion] += 1

    highest_score = max(scores.values(), default=0)

    if highest_score == 0:
        return "none", dict(scores)

    # A fixed order makes tied results repeatable.
    for emotion in EMOTIONS:
        if scores[emotion] == highest_score:
            return emotion, dict(scores)


def add_nrc_emotions():
    """Add NRC emotions to the LLM prediction results."""

    lexicon = load_nrc_lexicon()
    total_reviews = 0
    agreements = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as input_file:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as output_file:
            for line in input_file:
                review = json.loads(line)

                nrc_emotion, nrc_scores = find_nrc_emotion(
                    review["title"],
                    review["text"],
                    lexicon,
                )

                review["nrc_emotion"] = nrc_emotion
                review["nrc_scores"] = nrc_scores
                review["emotion_agreement"] = (
                    review["llm_emotion"] == nrc_emotion
                )

                total_reviews += 1

                if review["emotion_agreement"]:
                    agreements += 1

                output_file.write(json.dumps(review) + "\n")

                print(
                    f"Review {review['review_number']}: "
                    f"LLM = {review['llm_emotion']} | "
                    f"NRC = {nrc_emotion}"
                )

    agreement_rate = agreements / total_reviews * 100

    print(f"\nEmotion agreement: {agreements}/{total_reviews} "
          f"({agreement_rate:.1f}%)")
    print(f"Saved results in: {OUTPUT_FILE.name}")


if __name__ == "__main__":
    add_nrc_emotions()