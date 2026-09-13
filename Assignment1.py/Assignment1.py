"""Step 1: Classify one review as POSITIVE or NEGATIVE.

This file sends a review title and text to the course's model. It does not
read the Amazon data or use a star rating.
"""

import json
from urllib import request


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

Reply with exactly one word: POSITIVE or NEGATIVE.

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

    answer = response_data["choices"][0]["message"]["content"].strip().upper()

    if answer not in ["POSITIVE", "NEGATIVE"]:
        raise ValueError(f"The model returned an unexpected answer: {answer}")

    return answer


if __name__ == "__main__":
    # These two examples are a Step 1 spot-check, not part of the data set.
    positive_sentiment = classify_review(
        "Perfect gift",
        "The card arrived quickly and was easy to use.",
    )
    print(f"Positive example: {positive_sentiment}")

    negative_sentiment = classify_review(
        "Very disappointing",
        "The code did not work and I could not use the card.",
    )
    print(f"Negative example: {negative_sentiment}")
