import pytest

from perspectival.util import truncate_at_stopping_strings


TEST_TEXT = "This is a text! How you doin? Good. Wow!"


@pytest.mark.parametrize(
    "text, stopping_strings, look_forward, result",
    [
        (TEST_TEXT, ["text! How", "!", "."], False, "This is a text!"),
        (TEST_TEXT, ["text! How", "!", "."], True, "This is a "),
        (TEST_TEXT, ["7", "\n", "texti"], False, TEST_TEXT),
        (TEST_TEXT, ["7", "\n", "texti"], True, TEST_TEXT),
        (" Hi!", [" "], False, " "),
        (" Hi!", [" "], True, ""),
    ],
)
def test_truncate_at_stopping_strings(
    text,
    stopping_strings,
    look_forward,
    result,
):
    truncated_text = truncate_at_stopping_strings(text, stopping_strings, look_forward)
    assert truncated_text == result
