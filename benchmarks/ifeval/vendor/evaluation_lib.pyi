from collections.abc import Mapping

type _KwargValue = str | int | float | list[str]

class InputExample:
    key: int
    instruction_id_list: list[str]
    prompt: str
    kwargs: list[dict[str, _KwargValue]]
    def __init__(
        self,
        key: int,
        instruction_id_list: list[str],
        prompt: str,
        kwargs: list[dict[str, _KwargValue]],
    ) -> None: ...

class OutputExample:
    instruction_id_list: list[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: list[bool]

def test_instruction_following_strict(inp: InputExample, prompt_to_response: Mapping[str, str]) -> OutputExample: ...
def test_instruction_following_loose(inp: InputExample, prompt_to_response: Mapping[str, str]) -> OutputExample: ...
