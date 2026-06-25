from alexandria.segmentation import segment_instructions

_EXAMPLE_PROMPT = """
You are a helpful assistant.
- Always answer in English.
- Keep responses concise.
- Always answer in English.
""".strip()


def main() -> None:
    for instruction in segment_instructions(_EXAMPLE_PROMPT):
        print(instruction)


if __name__ == "__main__":
    main()
