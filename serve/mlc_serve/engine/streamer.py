from typing import List, Deque
from collections import deque

kReplacementCharacter = b"\xef\xbf\xbd".decode("utf8")


class TextStreamer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.prefix_tokens: List[int] = []
        self.pending_tokens: Deque[int] = deque([])

    def put(self, delta_tokens: List[int]) -> str:
        if len(delta_tokens) == 0:
            return ""

        ret = ""
        for delta_token in delta_tokens:
            self.pending_tokens.append(delta_token)
            all_tokens = self.prefix_tokens + list(self.pending_tokens)

            prefix_str = (
                self.tokenizer.decode(self.prefix_tokens)
                if len(self.prefix_tokens) > 0
                else ""
            )
            full_str = self.tokenizer.decode(all_tokens)
            prefix_len = len(prefix_str)

            new_pending_tokens: Deque[int] = deque([])
            if full_str[:prefix_len] == prefix_str:
                # Case 1. prefix_str is a prefix of `full_str`.
                # We cannot naively do `validated_str = self.tokenizer.decode(validated_tokens)`
                # since it will lose the contextual information, such as ' '.
                validated_str = full_str[prefix_len:]
                while (
                    len(self.pending_tokens) > 0
                    and len(new_pending_tokens) < 3
                    and len(validated_str) >= 1
                    and validated_str[len(validated_str) - 1 :] == kReplacementCharacter
                ):
                    new_pending_tokens.appendleft(self.pending_tokens.pop())
                    validated_str = validated_str[: len(validated_str) - 1]
            else:
                # Case 2. prefix_str is not a prefix of `full_str`.
                # Pop pending tokens from the back.
                # - Pop until prefix_str is indeed a prefix of full_str.
                # - A valid UTF-8 has 4 chars at most.
                #   So there will be at most 3 tokens popped.
                # - If there are no more than 3 pending tokens, skip popping.
                #   This is because it is impossible to make full_str contain
                #   prefix_str without popping all the pending tokens.
                if len(self.pending_tokens) < 3:
                    continue
                get_valid_full_str = False
                while len(self.pending_tokens) > 0 and len(new_pending_tokens) < 3:
                    new_pending_tokens.appendleft(self.pending_tokens.pop())
                    all_tokens.pop()
                    full_str = self.tokenizer.decode(all_tokens)
                    if full_str[:prefix_len] == prefix_str:
                        get_valid_full_str = True
                        break
                if get_valid_full_str:
                    # We find a full_str which starts from prefix_str
                    # So we return the sliced full string without the prefix.
                    validated_str = full_str[prefix_len:]
                else:
                    # We cannot find a full_str which starts from prefix_str by
                    # popping 3 tokens.
                    # In this case, the remaining pending tokens are invalid UTF-8
                    # characters already, so we return the decoded pending tokens.
                    validated_str = self.tokenizer.decode(self.pending_tokens)

            if len(self.pending_tokens) > 0:
                # set the new prefix
                self.prefix_tokens = list(self.pending_tokens)
            self.pending_tokens = new_pending_tokens

            ret += validated_str
        return ret

    def finish(self) -> str:
        all_tokens = self.prefix_tokens + list(self.pending_tokens)
        prefix_str = (
            self.tokenizer.decode(self.prefix_tokens)
            if len(self.prefix_tokens) > 0
            else ""
        )
        full_str = self.tokenizer.decode(all_tokens) if len(all_tokens) > 0 else ""
        prefix_len = len(prefix_str)

        if full_str[:prefix_len] == prefix_str:
            return full_str[prefix_len:]
        else:
            return self.tokenizer.decode(self.pending_tokens)
