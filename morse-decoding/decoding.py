import random
import itertools
from typing import Collection, Tuple

from trie import Trie


class Decoder:
    def __init__(self, dictionary: Collection[str]) -> None:
        """Construct a decoder from a dictionary of allowed words.

        Each word is a string of letters ("a" through "z").
        """
        self.sep_letter = " "
        self.sep_word = "|"
        # Trie for morse code tree maybe an overkill, but this is scalable to any similar encoding
        self.morse_trie = Trie(words=REV_MORSE_CODE)
        # Useful as english dictionary size can go upto 100K words
        self.eng_trie = Trie(words=dictionary)
        self.MAX_MORSE_LEN = 4  # as only 'a-z' chars, max code for a letter = 4.

    def _encode(self, message: str, adversarial: bool = False) -> str:
        """
        Encodes a given message into morse code, optionally obfuscating it
        by randomly replacing the letter and word breaks.
        Assumes the message only contains 'a-z' with single spaces between words.
        """
        morse = ""
        for char in message:
            if char != " ":
                morse += MORSE_CODE[char]
                if adversarial and random.randint(0, 1) == 1:
                    continue
                else:
                    morse += " "
            else:
                if adversarial and random.randint(0, 1) == 1:
                    continue
                else:
                    morse = (morse[:-1] + "|") if morse[-1] == " " else (morse + "|")
        return morse

    def decode(self, morse: str) -> Collection[str]:
        """Return all possible decodings of the input message 'morse'.

        Each returned string is a space-separated sequence of words
        from the dictionary.
        """
        # Split and later join along already given word breaks
        results_segments = []
        for seg in morse.split(self.sep_word):
            cache = {} # a bit naive application of a cache
            results_segments.append(self._helper(seg, ("", "", ""), cache))
        result = [" ".join(merged) for merged in itertools.product(*results_segments)]
        return result

    def _helper(
        self, morse: str, state: Tuple[str, str, str], cache: dict
    ) -> Collection[str]:
        if (morse, *state) in cache:
            return cache[result]
        result = []
        pre_code, pre_word, pre_sent = state
        for i in range(self.MAX_MORSE_LEN + 1):
            # Exit Condition
            if not morse[i:]:
                if not pre_code and not pre_word:
                    result.append(pre_sent)
                cache[(morse, *state)] = result
                return result

            # We encounter a definite (given) letter break
            # Dont try to extend pre_code/pre_word any further
            if morse[i] == self.sep_letter:
                if pre_code in REV_MORSE_CODE and self.eng_trie.startsWith(
                    pre_word + REV_MORSE_CODE[pre_code]
                ):
                    current_word = pre_word + REV_MORSE_CODE[pre_code]
                    partial_state = ("", current_word, pre_sent)
                    partial_result = self._helper(morse[i + 1 :], partial_state, cache)
                    for w in partial_result:
                        result.append(w)

                    # Also check if current_word itself can is valid for a word break
                    if self.eng_trie.search(pre_word + REV_MORSE_CODE[pre_code]):
                        updated_sent = (
                            pre_sent + " " + pre_word + REV_MORSE_CODE[pre_code]
                            if pre_sent
                            else pre_word + REV_MORSE_CODE[pre_code]
                        )
                        partial_state = ("", "", updated_sent)
                        partial_result = self._helper(
                            morse[i + 1 :], partial_state, cache
                        )
                        for w in partial_result:
                            result.append(w)

            pre_code += morse[i]
            # If we see an invalid code for a letter -> Early return
            # Also takes care of return on encountring a letter separator
            if not self.morse_trie.startsWith(pre_code):
                cache[(morse, *state)] = result
                return result

            # Try expanding the current code for a letter and hoping for a valid word
            if pre_code not in REV_MORSE_CODE or not self.eng_trie.startsWith(
                pre_word + REV_MORSE_CODE[pre_code]
            ):
                continue

            # We encounter a possible letter break
            partial_state = ("", pre_word + REV_MORSE_CODE[pre_code], pre_sent)
            partial_result = self._helper(morse[i + 1 :], partial_state, cache)
            for w in partial_result:
                result.append(w)

            # We encounter a possible word break
            if self.eng_trie.search(pre_word + REV_MORSE_CODE[pre_code]):
                updated_sent = (
                    pre_sent + " " + pre_word + REV_MORSE_CODE[pre_code]
                    if pre_sent
                    else pre_word + REV_MORSE_CODE[pre_code]
                )
                partial_word_state = ("", "", updated_sent)
                partial_word_result = self._helper(
                    morse[i + 1 :], partial_word_state, cache
                )
                for w in partial_word_result:
                    result.append(w)


MORSE_CODE = {
    "a": ".-",
    "b": "-...",
    "c": "-.-.",
    "d": "-..",
    "e": ".",
    "f": "..-.",
    "g": "--.",
    "h": "....",
    "i": "..",
    "j": ".---",
    "k": "-.-",
    "l": ".-..",
    "m": "--",
    "n": "-.",
    "o": "---",
    "p": ".--.",
    "q": "--.-",
    "r": ".-.",
    "s": "...",
    "t": "-",
    "u": "..-",
    "v": "...-",
    "w": ".--",
    "x": "-..-",
    "y": "-.--",
    "z": "--..",
}

REV_MORSE_CODE = {value: key for key, value in MORSE_CODE.items()}

if __name__ == "__main__":
    print(REV_MORSE_CODE)
    print(max(len(k) for k in REV_MORSE_CODE))
