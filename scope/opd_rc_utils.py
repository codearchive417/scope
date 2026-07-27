import re

from math_verify import parse, verify

from scope.prompts import FALLBACK_NO_GOLD, fallback_answer_mismatch


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks and trim surrounding whitespace."""
    stripped = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    return stripped.strip()


def extract_boxed_answer(text: str) -> str | None:
    """Extract the answer from \\boxed{} format."""
    think_end = text.rfind("</think>")
    search_text = text[think_end + len("</think>") :] if think_end != -1 else text

    idx = search_text.find(r"\boxed{")
    if idx == -1:
        return None
    start = idx + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(search_text) and depth > 0:
        if search_text[i] == "{":
            depth += 1
        elif search_text[i] == "}":
            depth -= 1
        i += 1
    if depth == 0:
        return search_text[start : i - 1].strip()
    return None


def _preprocess_for_parse(answer: str | None) -> str | None:
    if answer is None:
        return None
    ratio_match = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*:\s*(-?\d+(?:\.\d+)?)\s*", answer)
    if ratio_match:
        return rf"\frac{{{ratio_match.group(1)}}}{{{ratio_match.group(2)}}}"
    return answer


def verify_opd_rc_answer(rewrite_text: str, gold_solution: str) -> bool:
    gold_answer = extract_boxed_answer(gold_solution)
    if gold_answer is None:
        # GSM8K answers may be plain numbers after ####
        if "####" in gold_solution:
            gold_answer = gold_solution.split("####")[-1].strip()
        else:
            gold_answer = gold_solution.strip()

    pred_answer = extract_boxed_answer(rewrite_text)
    if pred_answer is None:
        return False

    gold_parsed = parse(gold_answer)
    pred_parsed = parse(_preprocess_for_parse(pred_answer))
    if gold_parsed is not None and pred_parsed is not None:
        try:
            return bool(verify(gold_parsed, pred_parsed))
        except Exception:
            pass

    pred_norm = re.sub(r"\s+", "", pred_answer or "").lower()
    gt_norm = re.sub(r"\s+", "", gold_answer or "").lower()
    return pred_norm == gt_norm and pred_norm != ""


def get_fallback_text(gold_solution: str) -> str:
    gold_answer = extract_boxed_answer(gold_solution)
    if gold_answer is None and "####" in gold_solution:
        gold_answer = gold_solution.split("####")[-1].strip()
    if gold_answer is None:
        return FALLBACK_NO_GOLD
    return fallback_answer_mismatch(gold_answer)


def verify_opd_rc_length_ratio(
    rewrite_text: str, reference_text: str, min_ratio: float
) -> bool:
    """Return True if OPD+RC text is at least min_ratio of reference length (char-based)."""
    ref_len = len(reference_text.strip())
    if ref_len == 0:
        return True
    rewrite_len = len(rewrite_text.strip())
    return rewrite_len >= min_ratio * ref_len


def get_len_ratio_fallback_text(min_ratio: float) -> str:
    pct = int(min_ratio * 100)
    return (
        f"The rewritten solution is too short compared to the reference reasoning. "
        f"Please preserve more necessary steps and details (at least about {pct}% of the reference length). "
        "Do not omit intermediate calculations or units needed to reach the answer."
    )
