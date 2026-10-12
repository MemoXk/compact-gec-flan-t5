"""Retokenizers for scoring system outputs against tokenized GEC benchmarks.

CoNLL-2014 and BEA-2019 sources are distributed pre-tokenized ("changes ." / "do n't"), and both
the M2 scorer and ERRANT (tokenise=False) compare whitespace tokens.  A system that writes ordinary
text ("changes.") therefore pays a spurious edit for every re-attached punctuation mark.  The fair
protocol is to retokenize EVERY system's output with a tokenizer that reproduces the benchmark's own
convention; this module provides the candidates and a round-trip test that picks one per benchmark.
"""
import re
import spacy
from spacy.util import compile_infix_regex

_NLP = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer", "tagger", "attribute_ruler"])
_NLP_NOHYPH = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer", "tagger", "attribute_ruler"])
_inf = [p for p in _NLP_NOHYPH.Defaults.infixes if "-" not in p and "–" not in p]
_NLP_NOHYPH.tokenizer.infix_finditer = compile_infix_regex(_inf).finditer


def spacy_tok(s):
    return " ".join(t.text for t in _NLP.make_doc(s) if not t.is_space)


def spacy_nohyph_tok(s):
    return " ".join(t.text for t in _NLP_NOHYPH.make_doc(s) if not t.is_space)


_CLITIC = re.compile(r"(?i)(\w)(n't|'s|'re|'ve|'ll|'d|'m)\b")
_PUNCT_AFTER = re.compile(r"(\S)([.,!?;:%)\]\"](?:\s|$))")
_PUNCT_BEFORE = re.compile(r"(^|\s)([(\[\"])(\S)")


def regex_tok(s):
    s = _CLITIC.sub(r"\1 \2", s)
    for _ in range(3):
        s = _PUNCT_AFTER.sub(r"\1 \2", s)
    s = _PUNCT_BEFORE.sub(r"\1\2 \3", s)
    return re.sub(r"\s+", " ", s).strip()


def detok(s):
    """Approximate natural-text rendering of a tokenized sentence (for the round-trip test)."""
    s = re.sub(r" (n't|'s|'re|'ve|'ll|'d|'m)\b", r"\1", s)
    s = re.sub(r" ([.,!?;:%)\]])", r"\1", s)
    s = re.sub(r"([(\[]) ", r"\1", s)
    s = re.sub(r"(\w) - (\w)", r"\1-\2", s)
    return s


TOKENIZERS = {"spacy": spacy_tok, "spacy_nohyphen": spacy_nohyph_tok, "regex": regex_tok}
