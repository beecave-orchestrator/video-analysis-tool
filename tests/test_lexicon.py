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
