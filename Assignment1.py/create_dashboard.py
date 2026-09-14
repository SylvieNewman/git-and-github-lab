import html
import json
from collections import Counter
from pathlib import Path


SCRIPT_FOLDER = Path(__file__).parent

ORIGINAL_FILE = SCRIPT_FOLDER / "first_100_with_emotions.jsonl"
BALANCED_FILE = SCRIPT_FOLDER / "balanced_150_predictions.jsonl"
OUTPUT_FILE = SCRIPT_FOLDER / "dashboard.html"

COLORS = {
    "POSITIVE": "#2a9d8f",
    "NEUTRAL": "#e9c46a",
    "NEGATIVE": "#e76f51",
}


def load_reviews(file_path):
    reviews = []

    with open(file_path, "r", encoding="utf-8") as input_file:
        for line in input_file:
            reviews.append(json.loads(line))

    return reviews


def prepare_reviews(reviews, three_classes):
    for review in reviews:
        rating = float(review["rating"])

        if three_classes:
            if rating >= 4:
                expected_label = "POSITIVE"
            elif rating == 3:
                expected_label = "NEUTRAL"
            else:
                expected_label = "NEGATIVE"
        else:
            expected_label = "POSITIVE" if rating >= 4 else "NEGATIVE"

        review["expected_label"] = expected_label
        review["is_correct"] = (
            review["predicted_sentiment"] == expected_label
        )


def make_distribution_bar(label, count, total):
    percentage = count / total * 100 if total else 0
    width = max(percentage, 1) if count else 0

    return f"""
    <div class="bar-row">
        <div class="bar-label">{label.title()}</div>
        <div class="bar-track">
            <div class="bar-fill"
                 style="width: {width}%; background: {COLORS[label]};">
            </div>
        </div>
        <div class="bar-value">{count} ({percentage:.1f}%)</div>
    </div>
    """


def make_accuracy_bar(label, correct, total):
    percentage = correct / total * 100 if total else 0

    return f"""
    <div class="bar-row">
        <div class="bar-label">{label.title()}</div>
        <div class="bar-track">
            <div class="bar-fill"
                 style="width: {percentage}%; background: {COLORS[label]};">
            </div>
        </div>
        <div class="bar-value">{percentage:.1f}%</div>
    </div>
    """


def build_dashboard_section(
    section_id,
    title,
    subtitle,
    reviews,
    labels,
    emotion_mode,
):
    total_reviews = len(reviews)
    correct_reviews = sum(review["is_correct"] for review in reviews)
    incorrect_reviews = total_reviews - correct_reviews
    agreement_rate = correct_reviews / total_reviews * 100

    expected_counts = Counter(
        review["expected_label"] for review in reviews
    )

    correct_by_class = Counter(
        review["expected_label"]
        for review in reviews
        if review["is_correct"]
    )

    confusion_matrix = {
        (expected, predicted): 0
        for expected in labels
        for predicted in labels
    }

    for review in reviews:
        confusion_matrix[
            (review["expected_label"], review["predicted_sentiment"])
        ] += 1

    distribution_bars = ""

    for label in labels:
        distribution_bars += make_distribution_bar(
            label,
            expected_counts[label],
            total_reviews,
        )

    accuracy_bars = ""

    for label in labels:
        accuracy_bars += make_accuracy_bar(
            label,
            correct_by_class[label],
            expected_counts[label],
        )

    matrix_header = ""

    for label in labels:
        matrix_header += f"<th>Model {label.lower()}</th>"

    matrix_rows = ""

    for expected in labels:
        matrix_cells = ""

        for predicted in labels:
            matrix_cells += (
                f"<td>{confusion_matrix[(expected, predicted)]}</td>"
            )

        matrix_rows += f"""
        <tr>
            <th>{expected}</th>
            {matrix_cells}
        </tr>
        """

    if emotion_mode == "both":
        emotion_headers = """
        <th>LLM emotion</th>
        <th>NRC emotion</th>
        """
    else:
        emotion_headers = """
        <th>NRC emotion</th>
        """

    review_rows = ""

    for review in reviews:
        result = "Match" if review["is_correct"] else "Mismatch"
        result_class = "match" if review["is_correct"] else "mismatch"

        llm_emotion = review.get("llm_emotion", "Not recorded")
        nrc_emotion = review.get("nrc_emotion", "Not recorded")

        if emotion_mode == "both":
            emotion_cells = f"""
            <td>{html.escape(str(llm_emotion))}</td>
            <td>{html.escape(str(nrc_emotion))}</td>
            """
        else:
            # The balanced file currently stores its NRC label here.
            emotion_cells = f"""
            <td>{html.escape(str(llm_emotion))}</td>
            """

        review_rows += f"""
        <tr data-result="{result}">
            <td>{review["review_number"]}</td>
            <td>{html.escape(str(review["title"]))}</td>
            <td>{review["rating"]}</td>
            <td>{review["expected_label"]}</td>
            <td>{review["predicted_sentiment"]}</td>
            {emotion_cells}
            <td class="{result_class}">{result}</td>
            <td>{html.escape(str(review["text"]))}</td>
        </tr>
        """

    return f"""
    <section class="run-section">
        <h2>{title}</h2>
        <p class="subtitle">{subtitle}</p>

        <div class="metrics">
            <div class="metric">
                <div class="metric-label">Reviews scored</div>
                <div class="metric-value">{total_reviews}</div>
            </div>

            <div class="metric">
                <div class="metric-label">Agreement with rating</div>
                <div class="metric-value">{agreement_rate:.1f}%</div>
            </div>

            <div class="metric">
                <div class="metric-label">Matches</div>
                <div class="metric-value">{correct_reviews}</div>
            </div>

            <div class="metric">
                <div class="metric-label">Mismatches</div>
                <div class="metric-value">{incorrect_reviews}</div>
            </div>
        </div>

        <h3>Rating-label distribution</h3>
        {distribution_bars}

        <h3>Accuracy by rating-label class</h3>
        {accuracy_bars}

        <h3>Rating label compared with model prediction</h3>
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Rating label</th>
                        {matrix_header}
                    </tr>
                </thead>
                <tbody>
                    {matrix_rows}
                </tbody>
            </table>
        </div>

        <h3>Review details</h3>

        <label for="{section_id}-filter">Show:</label>
        <select id="{section_id}-filter">
            <option value="all">All reviews</option>
            <option value="Match">Matches only</option>
            <option value="Mismatch">Mismatches only</option>
        </select>

        <p id="{section_id}-count"></p>

        <div class="table-container">
            <table id="{section_id}-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Title</th>
                        <th>Stars</th>
                        <th>Rating label</th>
                        <th>Model prediction</th>
                        {emotion_headers}
                        <th>Result</th>
                        <th>Review text</th>
                    </tr>
                </thead>
                <tbody>
                    {review_rows}
                </tbody>
            </table>
        </div>
    </section>
    """


def create_dashboard():
    original_reviews = load_reviews(ORIGINAL_FILE)
    balanced_reviews = load_reviews(BALANCED_FILE)

    prepare_reviews(original_reviews, three_classes=False)
    prepare_reviews(balanced_reviews, three_classes=True)

    original_section = build_dashboard_section(
        "original",
        "Original first-100 review run",
        "Binary labels: 4-5 stars = positive; 1-3 stars = negative.",
        original_reviews,
        ["POSITIVE", "NEGATIVE"],
        "both",
    )

    balanced_section = build_dashboard_section(
        "balanced",
        "Balanced 150-review run",
        "Three-class labels: 50 positive, 50 neutral, and 50 negative reviews.",
        balanced_reviews,
        ["POSITIVE", "NEUTRAL", "NEGATIVE"],
        "nrc",
    )

    html_page = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Amazon Gift Card Review Sentiment Dashboard</title>
    <style>
        body {{
            margin: 0;
            font-family: Arial, sans-serif;
            background: #f5f7fb;
            color: #1f2937;
        }}

        header {{
            background: #1d3557;
            color: white;
            padding: 36px 8%;
        }}

        h1 {{
            margin: 0;
        }}

        header p {{
            color: #dbeafe;
            margin-bottom: 0;
        }}

        main {{
            max-width: 1200px;
            margin: auto;
            padding: 30px;
        }}

        .run-section {{
            background: white;
            padding: 24px;
            margin-bottom: 28px;
            border-radius: 10px;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.08);
        }}

        .subtitle, .metric-label {{
            color: #64748b;
        }}

        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin: 22px 0;
        }}

        .metric {{
            background: #f8fafc;
            padding: 18px;
            border-radius: 8px;
        }}

        .metric-value {{
            font-size: 30px;
            font-weight: bold;
            margin-top: 8px;
        }}

        .bar-row {{
            display: grid;
            grid-template-columns: 170px 1fr 100px;
            gap: 12px;
            align-items: center;
            margin: 14px 0;
        }}

        .bar-track {{
            height: 20px;
            background: #e5e7eb;
            border-radius: 10px;
            overflow: hidden;
        }}

        .bar-fill {{
            height: 100%;
            border-radius: 10px;
        }}

        .bar-value {{
            text-align: right;
            font-weight: bold;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}

        th {{
            background: #1d3557;
            color: white;
            text-align: left;
        }}

        th, td {{
            padding: 10px;
            border-bottom: 1px solid #e5e7eb;
            vertical-align: top;
        }}

        .match {{
            color: #198754;
            font-weight: bold;
        }}

        .mismatch {{
            color: #c1121f;
            font-weight: bold;
        }}

        .table-container {{
            overflow-x: auto;
            margin-bottom: 20px;
        }}

        select {{
            padding: 8px;
            margin-left: 8px;
        }}
    </style>
</head>
<body>
    <header>
        <h1>Amazon Gift Card Review Sentiment</h1>
        <p>Model predictions compared with rating-based labels</p>
    </header>

    <main>
        {original_section}
        {balanced_section}
    </main>

    <script>
        function filterReviews(sectionId) {{
            const selectedFilter =
                document.getElementById(sectionId + "-filter").value;

            const rows =
                document.querySelectorAll(
                    "#" + sectionId + "-table tbody tr"
                );

            let visibleCount = 0;

            for (const row of rows) {{
                const shouldShow =
                    selectedFilter === "all" ||
                    row.dataset.result === selectedFilter;

                row.style.display = shouldShow ? "" : "none";

                if (shouldShow) {{
                    visibleCount += 1;
                }}
            }}

            document.getElementById(sectionId + "-count").textContent =
                visibleCount + " review(s) visible";
        }}

        for (const sectionId of ["original", "balanced"]) {{
            document.getElementById(sectionId + "-filter").addEventListener(
                "change",
                function () {{
                    filterReviews(sectionId);
                }}
            );

            filterReviews(sectionId);
        }}
    </script>
</body>
</html>
"""

    with open(OUTPUT_FILE, "w", encoding="utf-8") as output_file:
        output_file.write(html_page)

    print(f"Dashboard created: {OUTPUT_FILE.name}")


if __name__ == "__main__":
    create_dashboard()