from typing import List


def truncate_at_stopping_strings(
    text: str, stopping_strings: List[str], look_forward: bool = False
):
    """Return the part of `text` until any of the substrings in `stopping_strings`
    occurred.

    Arguments:
    :text: Text to truncate
    :stopping_strings: List of strings. The text is truncated at the earliest
        occurrence of any of these strings
    :look_forward: If True, the stopping string is not included in the truncated text"""
    string_positions = [text.find(s) for s in stopping_strings]
    stopping_positions = [
        text.find(stopping_string) + (len(stopping_string) if not look_forward else 0)
        for pos, stopping_string in zip(string_positions, stopping_strings)
        if pos >= 0
    ]
    if len(stopping_positions) > 0:
        stopping_position = min(stopping_positions)
    else:
        stopping_position = None
    return text[:stopping_position]
