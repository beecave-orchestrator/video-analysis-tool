"""Tests for the offline act lexicon."""

from video_nsfw_tagger.lexicon import find_tags


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
