"""The word/line mapping — the pure part of the alignment core, no ML needed.

These are the functions that turn WhisperX's flat word list back into the input's
line structure and carry through the honest `null` timing signal.
"""

from __future__ import annotations

from app.align import (
    Word,
    _collect_words,
    _map_words_to_lines,
    _norm,
)


def test_norm_strips_punctuation_and_case():
    assert _norm("Hello,") == "hello"
    assert _norm("don't") == "dont"
    assert _norm("...") == ""


def test_collect_words_keeps_unresolved_as_none():
    result = {
        "segments": [
            {
                "words": [
                    {"word": "hello", "start": 1.0, "end": 1.5},
                    {"word": "world"},  # aligner couldn't place it
                    {"word": "  "},  # empty after strip -> dropped
                ]
            }
        ]
    }
    words = _collect_words(result)
    assert [w.text for w in words] == ["hello", "world"]
    assert words[0].start == 1.0 and words[0].end == 1.5
    assert words[1].start is None and words[1].end is None


def test_map_words_to_lines_preserves_line_structure():
    lines_text = ["Hello darkness my old friend", "I've come to talk"]
    aligned = [
        Word("Hello", 0.0, 0.5),
        Word("darkness", 0.6, 1.2),
        Word("my", 1.3, 1.4),
        Word("old", 1.5, 1.8),
        Word("friend", 1.9, 2.5),
        Word("Ive", 3.0, 3.2),
        Word("come", 3.3, 3.5),
        Word("to", 3.6, 3.7),
        Word("talk", 3.8, 4.2),
    ]
    lines = _map_words_to_lines(lines_text, aligned)
    assert len(lines) == 2
    assert lines[0].text == "Hello darkness my old friend"
    assert lines[0].start == 0.0
    assert lines[0].end == 2.5
    assert [w.text for w in lines[0].words] == lines_text[0].split()
    assert lines[1].start == 3.0 and lines[1].end == 4.2


def test_line_all_unresolved_gets_null_bounds():
    lines_text = ["one two"]
    aligned = [Word("one", None, None), Word("two", None, None)]
    lines = _map_words_to_lines(lines_text, aligned)
    assert lines[0].start is None
    assert lines[0].end is None
    assert all(w.start is None for w in lines[0].words)


def test_line_bounds_ignore_unresolved_words():
    # A held final word the model dropped shouldn't wipe out the line's timing.
    lines_text = ["sing it loud"]
    aligned = [Word("sing", 1.0, 1.5), Word("it", 1.6, 1.7), Word("loud", None, None)]
    lines = _map_words_to_lines(lines_text, aligned)
    assert lines[0].start == 1.0
    assert lines[0].end == 1.7
