import json
from typing import Dict, List, Optional, Tuple


def load_tokenizer(tokenizer_file_path: str) -> List[str]:
    """
    Loads the list of tokens from a tokenizer file.
    """
    tokens: List[str] = []
    with open(tokenizer_file_path, "r", encoding="utf-8") as f:
        header_line = f.readline()
        assert header_line == "# FBAM_TOK v1.0\n", "Invalid header line"
        lines = f.readlines()
        for line in lines:
            token = json.loads(line)
            tokens.append(token)
    tokens_sorted = sorted(tokens, key=len, reverse=True)
    return tokens_sorted


class _TrieNode:
    __slots__ = ("children", "token_id")

    def __init__(self):
        self.children: Dict[str, "_TrieNode"] = {}
        self.token_id: Optional[int] = None


class TokenizerHandle:
    def __init__(self, tokenizer: "Tokenizer"):
        self.tokenizer = tokenizer
        self.current_node: Optional[_TrieNode] = tokenizer.root

    def add_char(self, char: int):
        """
        Consume one more character (given as its Unicode code point).
        If the path exists in the trie, we advance; otherwise we go to the dead state.
        """
        if self.current_node is None:
            return

        c = chr(char)
        nxt = self.current_node.children.get(c)
        if nxt is None:
            self.current_node = None
        else:
            self.current_node = nxt

    def get_current_token(self) -> Optional[Tuple[str, int]]:
        if self.current_node is None:
            return None
        tid = self.current_node.token_id
        if tid is None:
            return None
        return self.tokenizer.tokens[tid], tid

    def has_continuation(self) -> bool:
        return self.current_node is not None and bool(self.current_node.children)

    def is_dead(self) -> bool:
        return self.current_node is None

    def reset(self):
        self.current_node = self.tokenizer.root


class Tokenizer:
    def __init__(self, tokens: List[str]):
        self.tokens = tokens
        self.root = _TrieNode()
        for idx, tok in enumerate(tokens):
            node = self.root
            for ch in tok:
                node = node.children.setdefault(ch, _TrieNode())
            node.token_id = idx

    def new_tokenization_handle(self) -> TokenizerHandle:
        return TokenizerHandle(self)
