"""Token counting for the context budget. Falls back to chars/4 if tiktoken can't load."""

_enc = None
_tried = False


def count_tokens(text: str) -> int:
    global _enc, _tried
    if not _tried:
        _tried = True
        try:
            import tiktoken

            _enc = tiktoken.get_encoding("cl100k_base")
        except Exception:  # noqa: BLE001 - offline / no cached BPE file
            _enc = None
    if _enc is not None:
        return len(_enc.encode(text))
    return max(1, len(text) // 4)
