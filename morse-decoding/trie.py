from typing import Collection


class Node:
    def __init__(self, val="", end=False, children=None):
        self.char = val
        self.isEnd = end
        self.children = children if children else {}  # DICT of Node refs


class Trie:
    def __init__(self, words: Collection[str] = None):
        """
        If words is a non-null collection, populate the trie
        """
        self.root = Node()
        if words:
            self._populate(words)

    def search(self, word: str) -> bool:
        curr = self.root
        for char in word:
            if char not in curr.children:
                return False
            curr = curr.children[char]
        # if query word is part of prefix but was not
        # inserted, we need to return false
        return curr.isEnd

    def startsWith(self, prefix: str) -> bool:
        curr = self.root
        for char in prefix:
            if char not in curr.children:
                return False
            curr = curr.children[char]
        return True

    def _insert(self, word: str) -> None:
        curr = self.root
        for i, char in enumerate(word):
            if char not in curr.children:
                curr.children[char] = Node(char)
            curr = curr.children[char]
            if i == len(word) - 1:
                curr.isEnd = True

    def _populate(self, words: Collection[str]) -> None:
        """
        Iterates through a collection of words and inserts them into the trie
        """
        for word in words:
            self._insert(word)


if __name__ == "__main__":
    # Tests

    my_words = {"car", "cat", "dog", "doggo", "caterpillar", "hello", "help"}
    test_trie = Trie(words=my_words)

    assert test_trie.search("car")
    assert test_trie.startsWith("cater")
    assert test_trie.startsWith("hell")
    assert test_trie.search("doggo")
    assert not test_trie.search("doogo")
