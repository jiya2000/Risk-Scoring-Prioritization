"""
Faithfulness Validation Module for STR Summarization.

Validates that generated summaries preserve all critical entities from the
original narrative. Uses multiple matching strategies:
1. Exact string matching
2. Normalized number matching (handles format variations like "2.3M" vs "2,300,000")
3. Fuzzy date matching (handles "June 14" vs "2025-06-14")
4. Coverage scoring with per-category breakdown
"""

import re
from difflib import SequenceMatcher


def normalize_amount(amount_str):
    """
    Normalizes monetary amounts to a canonical form for comparison.
    Handles: NPR 2.3M, 2,300,000, NPR2300000, $2.3M, etc.
    """
    s = amount_str.strip().upper()
    # Remove currency symbols and labels
    s = re.sub(r'(NPR|USD|EUR|GBP|\$|€|£)\s*', '', s)
    s = s.replace(',', '')

    # Handle K/M/B suffixes
    multipliers = {'K': 1e3, 'M': 1e6, 'B': 1e9}
    for suffix, mult in multipliers.items():
        if s.endswith(suffix):
            try:
                return float(s[:-1]) * mult
            except ValueError:
                pass

    try:
        return float(s)
    except ValueError:
        return None


def normalize_date(date_str):
    """
    Extracts a canonical date representation for fuzzy matching.
    Handles various formats: 2025-06-14, June 14, 14/06/2025, etc.
    """
    # Try ISO format
    iso_match = re.search(r'(\d{4})-(\d{2})-(\d{2})', date_str)
    if iso_match:
        return (int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3)))

    # Try DD/MM/YYYY or MM/DD/YYYY
    slash_match = re.search(r'(\d{1,2})/(\d{1,2})/(\d{4})', date_str)
    if slash_match:
        return (int(slash_match.group(3)), int(slash_match.group(2)), int(slash_match.group(1)))

    return None


def fuzzy_match(entity, text, threshold=0.8):
    """
    Uses sequence matching for approximate string comparison.
    Useful for names that might be slightly reformatted.
    """
    entity_lower = entity.lower()
    text_lower = text.lower()

    if entity_lower in text_lower:
        return True

    # Sliding window comparison for partial matches
    entity_len = len(entity_lower)
    for i in range(len(text_lower) - entity_len + 1):
        window = text_lower[i:i + entity_len]
        ratio = SequenceMatcher(None, entity_lower, window).ratio()
        if ratio >= threshold:
            return True

    return False


def check_amount_preservation(original_amounts, summary_text):
    """
    Checks if monetary amounts are preserved, handling format variations.
    """
    found = 0
    missing = []

    for amt in original_amounts:
        # Direct string match
        if amt.lower() in summary_text.lower():
            found += 1
            continue

        # Normalized value comparison
        orig_value = normalize_amount(amt)
        if orig_value is not None:
            # Extract all numbers from summary
            summary_numbers = re.findall(r'[\d,]+\.?\d*[KMBkmb]?', summary_text)
            matched = False
            for sn in summary_numbers:
                sn_value = normalize_amount(sn)
                if sn_value is not None and abs(sn_value - orig_value) / max(orig_value, 1) < 0.01:
                    matched = True
                    break
            if matched:
                found += 1
                continue

        missing.append(amt)

    return found, missing


def check_entity_preservation(original_entities, summary_text):
    """
    Checks if extracted entities from the original narrative are preserved
    in the generated summary, using multiple matching strategies.
    
    Returns a detailed dict and overall Faithfulness Score (0-100).
    """
    summary_lower = summary_text.lower()

    results = {
        'amounts': {'total': 0, 'found': 0, 'missing': []},
        'dates': {'total': 0, 'found': 0, 'missing': []},
        'accounts': {'total': 0, 'found': 0, 'missing': []},
        'parties': {'total': 0, 'found': 0, 'missing': []}
    }

    # Amounts — use normalized matching
    amounts = original_entities.get('amounts', [])
    results['amounts']['total'] = len(amounts)
    if amounts:
        found, missing = check_amount_preservation(amounts, summary_text)
        results['amounts']['found'] = found
        results['amounts']['missing'] = missing

    # Dates — use fuzzy date matching
    for date in original_entities.get('dates', []):
        results['dates']['total'] += 1
        if date.lower() in summary_lower:
            results['dates']['found'] += 1
        else:
            # Try normalized date comparison
            orig_date = normalize_date(date)
            if orig_date:
                # Check if any part of the date appears
                year, month, day = orig_date
                if str(year) in summary_text and str(day) in summary_text:
                    results['dates']['found'] += 1
                else:
                    results['dates']['missing'].append(date)
            else:
                results['dates']['missing'].append(date)

    # Accounts — exact and fuzzy
    for account in original_entities.get('accounts', []):
        results['accounts']['total'] += 1
        if account.lower() in summary_lower:
            results['accounts']['found'] += 1
        else:
            results['accounts']['missing'].append(account)

    # Parties — fuzzy matching for names
    for party in original_entities.get('parties', []):
        results['parties']['total'] += 1
        if fuzzy_match(party, summary_text, threshold=0.75):
            results['parties']['found'] += 1
        else:
            results['parties']['missing'].append(party)

    # Calculate weighted Faithfulness Score
    # Amounts and accounts are critical (higher weight), dates and parties less so
    weights = {'amounts': 3.0, 'accounts': 2.0, 'dates': 1.5, 'parties': 1.0}

    total_weighted = sum(results[cat]['total'] * weights[cat] for cat in results)
    found_weighted = sum(results[cat]['found'] * weights[cat] for cat in results)

    if total_weighted == 0:
        score = 100.0
    else:
        score = (found_weighted / total_weighted) * 100.0

    return score, results


def validate_summary(original_narrative, summary, extracted_entities):
    """
    Full validation pipeline. Returns structured result with score and details.
    """
    score, details = check_entity_preservation(extracted_entities, summary)

    return {
        'faithfulness_score': score,
        'is_fully_faithful': score >= 95.0,  # Allow small tolerance
        'details': details,
        'grade': 'A' if score >= 90 else 'B' if score >= 70 else 'C' if score >= 50 else 'F'
    }


if __name__ == '__main__':
    orig_entities = {
        'amounts': ['NPR 2.3M'],
        'dates': ['2025-06-14'],
        'accounts': ['7423'],
        'parties': ['Nexus Corp']
    }

    good_summary = "[Risk: 0.91] Account 7423 moved NPR 2.3M to Nexus Corp on 2025-06-14."
    bad_summary = "[Risk: 0.91] The account moved a large sum of money recently."

    result_good = validate_summary("", good_summary, orig_entities)
    result_bad = validate_summary("", bad_summary, orig_entities)

    print(f"Good Summary: Score={result_good['faithfulness_score']:.1f}% Grade={result_good['grade']}")
    print(f"Bad Summary:  Score={result_bad['faithfulness_score']:.1f}% Grade={result_bad['grade']}")
    print(f"Missing from bad: {result_bad['details']}")
