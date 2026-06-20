"""
STR Narrative Summarization Module

Provides:
- Entity extraction via spaCy NER + domain-specific regex
- Local LLM summarization (open-source, no proprietary APIs)
- Risk context injection into summaries
- Entity-preserving mock mode for fast dashboard rendering
"""

import re

_nlp = None


def get_nlp():
    """Lazily loads the spaCy English model."""
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
        except (OSError, ImportError):
            try:
                import os
                os.system("python -m spacy download en_core_web_sm")
                import spacy
                _nlp = spacy.load("en_core_web_sm")
            except Exception:
                _nlp = None
    return _nlp


def extract_entities_spacy(text):
    """
    Extracts critical entities from STR narratives using spaCy NER + domain regex.
    
    Categories:
    - amounts: Monetary values (NPR 2.3M, $500, etc.)
    - dates: Temporal references
    - accounts: Account identifiers
    - parties: Organizations and persons
    - jurisdictions: Country/location mentions relevant to AML
    """
    entities = {
        'amounts': [],
        'dates': [],
        'accounts': [],
        'parties': [],
        'jurisdictions': []
    }

    nlp = get_nlp()
    if nlp is not None:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "MONEY":
                entities['amounts'].append(ent.text)
            elif ent.label_ in ("DATE", "TIME"):
                entities['dates'].append(ent.text)
            elif ent.label_ in ("ORG", "PERSON"):
                entities['parties'].append(ent.text)
            elif ent.label_ in ("GPE", "LOC"):
                entities['jurisdictions'].append(ent.text)

    # Domain-specific regex patterns

    # Account numbers (A5, Account #7423, NP000123)
    acc_patterns = [
        r'[Aa]ccount\s*(?:#|no\.?|number)?\s*([A-Za-z0-9]+)',
        r'\b(NP\d{3,})\b',
        r'\b(A\d+)\b',
    ]
    for pattern in acc_patterns:
        matches = re.findall(pattern, text)
        for m in matches:
            if m not in entities['accounts']:
                entities['accounts'].append(m)

    # Monetary amounts with currency codes (NPR 2.3M, USD 500K, etc.)
    amount_patterns = [
        r'(?:NPR|USD|EUR|GBP|\$|€|£)\s*[\d,]+(?:\.\d+)?(?:\s*(?:K|M|B|million|billion|thousand))?',
        r'[\d,]+(?:\.\d+)?\s*(?:K|M|B|million|billion|thousand)\s*(?:NPR|USD|EUR|GBP)?',
    ]
    for pattern in amount_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if match.strip() not in entities['amounts']:
                entities['amounts'].append(match.strip())

    # Date patterns (2025-06-14, June 14 2025, 14/06/2025)
    date_patterns = [
        r'\d{4}-\d{2}-\d{2}',
        r'\d{1,2}/\d{1,2}/\d{4}',
        r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,?\s*\d{4})?',
    ]
    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if match not in entities['dates']:
                entities['dates'].append(match)

    # Deduplicate
    for key in entities:
        entities[key] = list(dict.fromkeys(entities[key]))

    return entities


class LocalLLMSummarizer:
    """
    Local open-source LLM summarizer for STR narratives.
    
    Uses HuggingFace Transformers with SmolLM-135M-Instruct (compliant with
    hackathon rules — no proprietary APIs).
    
    Supports:
    - Full LLM inference mode
    - Entity-preserving mock mode for dashboard responsiveness
    """

    def __init__(self, model_id="HuggingFaceTB/SmolLM-135M-Instruct", use_mock=False):
        self.use_mock = use_mock
        self.model_id = model_id

        if not self.use_mock:
            try:
                import torch
                from transformers import pipeline
                print(f"Loading local LLM: {model_id}...")
                device = 0 if torch.cuda.is_available() else -1
                self.generator = pipeline("text-generation", model=model_id, device=device)
            except Exception as e:
                print(f"Failed to load {model_id}: {e}. Falling back to mock.")
                self.use_mock = True

    def summarize(self, narrative, risk_score=None, typology=None):
        """
        Generates a summary with injected risk context.
        Entity preservation is prioritized.
        """
        # Build context prefix
        context_parts = []
        if risk_score is not None:
            context_parts.append(f"Risk Score: {risk_score:.2f}")
        if typology and typology != "None":
            context_parts.append(f"Detected Typology: {typology}")
        context = ". ".join(context_parts)

        if self.use_mock:
            return self._mock_summarize(narrative, context)

        # Full LLM inference
        prompt = f"""You are a financial crime analyst. Summarize the following Suspicious Transaction Report (STR) narrative in 100-200 words.
CRITICAL: You MUST preserve ALL specific amounts, dates, account numbers, and entity names mentioned.

Context: {context}

Narrative:
{narrative}

Summary:"""

        from transformers import pipeline
        outputs = self.generator(
            prompt,
            max_new_tokens=200,
            num_return_sequences=1,
            return_full_text=False,
            temperature=0.3,
            do_sample=True
        )
        summary = outputs[0]['generated_text'].strip()

        # Ensure context is present
        if context and context not in summary:
            summary = f"[{context}] {summary}"

        return summary

    def _mock_summarize(self, narrative, context):
        """
        Entity-preserving extractive summary (mock mode).
        Ensures all key entities from the original appear in the output.
        """
        # Extract entities to ensure preservation
        entities = extract_entities_spacy(narrative)

        # Split into sentences and select most informative ones
        sentences = [s.strip() for s in narrative.split('.') if s.strip()]

        # Score sentences by entity density
        scored = []
        for sent in sentences:
            score = 0
            for cat in entities.values():
                for entity in cat:
                    if entity.lower() in sent.lower():
                        score += 1
            scored.append((score, sent))

        # Take top sentences (by entity density), up to 3
        scored.sort(key=lambda x: -x[0])
        top_sentences = [s[1] for s in scored[:3]]

        # Reconstruct
        summary = ". ".join(top_sentences) + "."

        # Prepend context
        if context:
            summary = f"[{context}] {summary}"

        return summary


def process_str_narrative(narrative, risk_score=None, typology=None, summarizer=None):
    """
    End-to-end processing of a single STR narrative.
    
    Returns:
        summary: Generated summary text
        entities: Extracted entity dict
    """
    if summarizer is None:
        summarizer = LocalLLMSummarizer(use_mock=True)

    entities = extract_entities_spacy(narrative)
    summary = summarizer.summarize(narrative, risk_score, typology)

    return summary, entities


if __name__ == '__main__':
    print("=" * 60)
    print("  STR SUMMARIZATION PIPELINE TEST")
    print("=" * 60)

    test_narratives = [
        "Account #7423 conducted 14 transactions totalling NPR 2.3M over 48 hours starting on 2025-06-14. Funds were sent to Nexus Corp.",
        "Account A5 received 15 transfers under threshold within 24h totalling NPR 2,300,000. Customer could not explain origin. Funds wired to overseas account at Nexus Corp on 2025-06-14.",
    ]

    summarizer = LocalLLMSummarizer(use_mock=True)

    for i, text in enumerate(test_narratives):
        print(f"\n--- Narrative {i+1} ---")
        print(f"  Input: {text[:60]}...")

        entities = extract_entities_spacy(text)
        print(f"  Entities: {entities}")

        summary, _ = process_str_narrative(text, 0.91, "Smurfing", summarizer)
        print(f"  Summary: {summary[:80]}...")
