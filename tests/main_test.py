from alexandria.segmentation import segment_instructions


def test_segments_each_non_empty_line() -> None:
    prompt = "First instruction.\nSecond instruction."
    assert segment_instructions(prompt) == ["First instruction.", "Second instruction."]


def test_strips_list_markers_and_blank_lines() -> None:
    prompt = "- Answer in English.\n\n1. Be concise.\n"
    assert segment_instructions(prompt) == ["Answer in English.", "Be concise."]


def test_returns_empty_for_blank_prompt() -> None:
    assert segment_instructions("   \n\n") == []
