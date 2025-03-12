# Morse Code Decoder

*Ninad Khargonkar*

Programming Language: Python

## Usage

- `main.py`: Tests for the main more code decoding tasks. Also includes some custom test cases
- `decoding.py`: `Decoder` class is implemented here.
  - `Decoder` also includes an `_encode()` method which can encode english messages to morse with optional, random removal of separators 
- `trie.py`: A `Trie` class is implemented here. `Trie` used for efficient search for prefix and complete words from the given dictionary


## Implementation Notes

- The decoding is implemented as a recursive algorithm with a "state" being `(morse code prefix, english word prefix, currently built sentence)` tuple.
  - We try to expand the `morse code prefix` by looking to add upto first 4 chars from given morse string
  - If such an expansion is legal, we also check for exapnding the `english word prefix` by checking in trie of english words
  - Then recursive calls with the expanded english word are made with an additional call if the word is valid (wrt provided dictionary) 
- As the dictionary size could include greater than 10000 words, I used a Trie method which is commonly used for efficient string prefix searches.
- Code is auto formatted using `black` (in case some formatting looks too opinionated).
