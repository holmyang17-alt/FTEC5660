#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Create and return your LangChain chain once."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_deepseek import ChatDeepSeek

    model = ChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",
        temperature=0,
    )

    system_text = (
        "You are an expert at reading supermarket receipts. "
        "From the receipt image, find two numbers and return them as a single JSON object "
        "with keys pre_discount_total and final_payment. "
        "pre_discount_total is the sum of ALL item prices BEFORE any discount, promotion, "
        "or coupon is applied. Equivalently, it is the SUBTOTAL line amount plus every "
        "discount/promotion/coupon amount added back as a positive number. "
        "Do NOT include the ROUNDING line. "
        "final_payment is the final amount actually paid after ROUNDING, "
        "the payment line such as OCTOPUS, VISA, CASH, etc. "
        "Return ONLY the JSON object. No explanation, no currency symbols, no extra text. "
        "Example of the shape: pre_discount_total is 107.70, final_payment is 102.30."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_text),
        ("human", [
            {"type": "text", "text": "Here is the receipt image. Return the JSON object."},
            {"type": "image_url", "image_url": {"url": "{image_url}"}},
        ]),
    ])

    return prompt | model


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string."""
    from collections import Counter

    def _amount(value: Any) -> Decimal:
        s = str(value).replace(",", "").replace("HK$", "").replace("$", "").strip()
        return Decimal(s).quantize(Decimal("0.01"))

    n_votes = 3
    inputs = []
    for path in images:
        url = image_data_url(path)
        for _ in range(n_votes):
            inputs.append({"image_url": url})

    results = chain.batch(inputs)

    per_image = []
    for i in range(len(images)):
        votes = []
        for j in range(n_votes):
            text = response_text(results[i * n_votes + j])
            data = None
            try:
                data = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                m = re.search(r"\{.*\}", text, flags=re.S)
                if m:
                    try:
                        data = json.loads(m.group(0))
                    except (json.JSONDecodeError, TypeError):
                        data = None
            if isinstance(data, dict) and "pre_discount_total" in data and "final_payment" in data:
                try:
                    votes.append((
                        _amount(data["final_payment"]),
                        _amount(data["pre_discount_total"]),
                    ))
                except (InvalidOperation, TypeError):
                    pass
        per_image.append(votes)

    total_q1 = Decimal("0")
    total_q2 = Decimal("0")
    for votes in per_image:
        if not votes:
            continue
        best, _ = Counter(votes).most_common(1)[0]
        total_q1 += best[0]
        total_q2 += best[1]

    return {
        QUERY_1: f"HK${total_q1:.2f}",
        QUERY_2: f"HK${total_q2:.2f}",
    }


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
