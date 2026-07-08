"""
Echo screening for G0 T2/T3 items.

Checks that concept tokens don't appear in prompts via:
1. Exact substring (bidirectional)
2. Stem matching (Porter stemmer)
3. Tokenizer-level verification (concept's BPE tokens don't appear in prompt's BPE tokens)

Also screens against chat template text and distractors for collision.

Usage:
    python echo_screen.py items.json [--tokenizer Qwen/Qwen3-8B]
"""

import json
import sys
import re
from pathlib import Path

try:
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
except ImportError:
    stemmer = None
    print("WARNING: nltk not available, stem matching disabled")


def stem(word):
    if stemmer:
        return stemmer.stem(word.lower())
    return word.lower()


def get_all_forms(concept, banned_list):
    """Get all surface forms to check, including stems."""
    forms = set()
    for b in banned_list:
        forms.add(b.lower())
        forms.add(stem(b))
    forms.add(concept.lower())
    forms.add(stem(concept))
    return forms


def check_bidirectional_substring(text, forms):
    """Check if any form is a substring of text OR text contains a substring of any form."""
    text_lower = text.lower()
    hits = []
    for form in forms:
        if form in text_lower:
            hits.append(f"'{form}' found in prompt text")
        # Also check if any word in the text is a substring of the form
        # Minimum 5 chars to avoid false positives (the/for/and/are etc)
        for word in re.findall(r'\b\w+\b', text_lower):
            if len(word) >= 5 and word in form and word != form:
                hits.append(f"prompt word '{word}' is substring of banned form '{form}'")
    return hits


def check_stem_overlap(text, forms):
    """Check if any word stem in text matches any concept stem."""
    text_words = re.findall(r'\b\w+\b', text.lower())
    text_stems = {stem(w) for w in text_words}
    concept_stems = {stem(f) for f in forms}
    overlap = text_stems & concept_stems
    if overlap:
        return [f"stem overlap: {overlap}"]
    return []


def check_tokenizer_collision(prompt_text, concept, distractors, tokenizer_name=None):
    """Check that concept's BPE tokens uniquely identify it among candidates."""
    if tokenizer_name is None:
        return []
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_name)

        concept_ids = set(tok.encode(concept, add_special_tokens=False))
        prompt_ids = set(tok.encode(prompt_text, add_special_tokens=False))

        overlap = concept_ids & prompt_ids
        if overlap:
            overlap_tokens = [tok.decode([tid]) for tid in overlap]
            return [f"BPE token overlap between concept and prompt: {overlap_tokens}"]

        # Check distractor collisions
        for d in distractors:
            d_ids = set(tok.encode(d, add_special_tokens=False))
            collision = concept_ids & d_ids
            if collision:
                collision_tokens = [tok.decode([tid]) for tid in collision]
                return [f"BPE collision between concept '{concept}' and distractor '{d}': {collision_tokens}"]
    except Exception as e:
        return [f"tokenizer check failed: {e}"]

    return []


def screen_item(item, tokenizer_name=None, chat_template_text=""):
    """Screen a single item for echo contamination."""
    concept = item["concept"]
    banned = item.get("concept_tokens_banned", item.get("banned", [concept]))
    prompt = item["prompt"]
    distractors = item.get("distractors", [])

    forms = get_all_forms(concept, banned)

    # Check prompt text
    full_text = prompt + " " + chat_template_text
    issues = []
    issues.extend(check_bidirectional_substring(full_text, forms))
    issues.extend(check_stem_overlap(full_text, forms))

    # Check probe text too (if probes are constant across items)
    # The probe "What makes someone real? Not pretend" contains "pretend"

    if tokenizer_name:
        issues.extend(check_tokenizer_collision(prompt, concept, distractors, tokenizer_name))

    return issues


def screen_file(filepath, tokenizer_name=None):
    """Screen all items in a JSON file."""
    with open(filepath, encoding='utf-8') as f:
        data = json.load(f)

    items = data if isinstance(data, list) else data.get("items", [])

    chat_template = "You are a helpful assistant."  # default

    total = 0
    failed = 0
    passed = 0

    for item in items:
        total += 1
        issues = screen_item(item, tokenizer_name, chat_template)
        if issues:
            failed += 1
            print(f"FAIL {item['id']} (concept: {item['concept']}):")
            for issue in issues:
                print(f"  - {issue}")
        else:
            passed += 1
            print(f"PASS {item['id']} ({item['concept']})")

    print(f"\n{'='*50}")
    print(f"Total: {total}  Passed: {passed}  Failed: {failed}")
    if failed:
        print(f"\n{failed} items need revision before use.")
    else:
        print(f"\nAll items clean.")
    return failed


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python echo_screen.py items.json [--tokenizer MODEL_NAME]")
        sys.exit(1)

    filepath = sys.argv[1]
    tokenizer_name = None
    if "--tokenizer" in sys.argv:
        idx = sys.argv.index("--tokenizer")
        if idx + 1 < len(sys.argv):
            tokenizer_name = sys.argv[idx + 1]

    failures = screen_file(filepath, tokenizer_name)
    sys.exit(1 if failures else 0)
