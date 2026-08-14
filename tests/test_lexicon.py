"""Tests for the offline act lexicon."""

from video_nsfw_tagger.lexicon import find_matches, find_tags


def test_find_matches_returns_matched_patterns():
    lexicon = {
        "kissing": ["kiss", "making out"],
        "nudity": ["nude", "naked"],
    }
    assert find_matches("Two people making out, fully naked", lexicon) == {
        "kissing": ["making out"],
        "nudity": ["naked"],
    }
    assert find_matches("Nothing here", lexicon) == {}


def test_find_tags():
    lexicon = {
        "kissing": ["kiss", "kissing"],
        "nudity": ["nude", "naked"],
    }
    assert find_tags("Two people kissing on a couch", lexicon) == ["kissing"]
    assert find_tags("A naked person", lexicon) == ["nudity"]
    assert find_tags("Nothing here", lexicon) == []


def test_find_tags_uses_word_boundaries():
    lexicon = {
        "genital_focus": ["ass", "cock"],
        "kissing": ["kiss"],
    }
    # Substrings inside larger words must not match.
    assert find_tags("People wearing glasses in class", lexicon) == []
    assert find_tags("A peacock strutting around", lexicon) == []
    # Whole-word and phrase matches still work, case-insensitively.
    assert find_tags("He grabbed her Ass", lexicon) == ["genital_focus"]
    assert find_tags("a Kiss!", lexicon) == ["kissing"]


def test_find_matches_skips_negated_patterns():
    """Negation cues preceding a match suppress it."""
    lexicon = {
        "kissing": ["kissing", "kiss"],
        "nudity": ["naked", "exposed"],
        "bdsm_kink": ["bondage"],
        "vaginal_sex": ["sex"],
    }
    # Direct negation: "No kissing:", "not exposed", "No bondage"
    assert find_matches("No kissing: their faces are apart", lexicon) == {}
    assert find_matches("She is not exposed in the image", lexicon) == {}
    assert find_matches("No bondage: no ropes or belts", lexicon) == {}
    # "no sex" negates the match
    assert find_matches("There are no sex toys present", lexicon) == {}


def test_find_matches_keeps_non_negated_patterns():
    """Non-negated matches are unaffected by the negation filter."""
    lexicon = {
        "kissing": ["kissing"],
        "nudity": ["naked"],
        "vaginal_sex": ["sex"],
    }
    assert find_matches("They are kissing passionately", lexicon) == {
        "kissing": ["kissing"]
    }
    assert find_matches("A naked person stands there", lexicon) == {"nudity": ["naked"]}
    assert find_matches("They are having sex", lexicon) == {"vaginal_sex": ["sex"]}


def test_find_matches_negation_with_positive_in_same_text():
    """Negation suppresses only the negated occurrence, not later ones."""
    lexicon = {
        "kissing": ["kissing"],
        "vaginal_sex": ["sex"],
    }
    text = "No kissing is happening, but they are having sex"
    assert find_matches(text, lexicon) == {"vaginal_sex": ["sex"]}


def test_find_matches_negation_with_colon_dash():
    """Negation cues followed by ':' or '-' still suppress the match."""
    lexicon = {
        "kissing": ["kissing"],
        "nudity": ["naked"],
    }
    assert find_matches("No kissing - their faces are apart", lexicon) == {}
    assert find_matches("Without naked: fully clothed scene", lexicon) == {}
