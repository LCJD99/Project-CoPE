from src.cope.stage1.tokenizer_expand import extract_virtual_tokens


def test_extract_virtual_tokens_sorted_and_wrapped():
    registry = {"tokens": {"B_TOOL": {}, "A_TOOL": {}}}
    tokens = extract_virtual_tokens(registry)
    assert tokens == ["<A_TOOL>", "<B_TOOL>"]
