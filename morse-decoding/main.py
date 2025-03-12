from decoding import Decoder
from trie import Trie

if __name__ == "__main__":
    # Test Cases

    # Case-1
    decoder = Decoder(dictionary={"a", "awe", "eye", "we"})
    assert set(decoder.decode(".-|.--.")) == {"a we"}
    assert set(decoder.decode(". -.--.")) == {"eye"}
    assert set(decoder.decode(".- .--.")) == {"awe", "a we"}
    assert set(decoder.decode(".-.--.")) == {"awe", "eye", "a we"}

    # Case-2
    decoder = Decoder(dictionary={"brains", "mesne", "robot", "rodeo", "tease", "the"})
    morse = "- .... .|.-. --- -... --- -|-... .-. .- .. -. ..."
    assert set(decoder.decode(morse)) == {"the robot brains"}
    morse = "- .... .|.-. --- -... --- --... .-. .- .. -. ..."
    assert set(decoder.decode(morse)) == {"the robot brains"}
    morse = "-.....|.-. --- -...-----....-..-..-...."
    assert set(decoder.decode(morse)) == {"the robot brains", "the rodeo mesne tease"}
    morse = "- .... .|.-. --- -... --- -|-... .-. .- .. -. ..."
    assert (set(decoder.decode(morse))) == {"the robot brains"}
    morse = "- ......-. --- -... -----....-. .- .. -...."
    assert "the robot brains" in (set(decoder.decode(morse)))

    # # Case-3
    with open("dictionary.txt", "r") as f:
        dictionary = f.read().split("\n")  # last element is an empty string for EOF
        dictionary = set(dictionary[:-1])
    decoder = Decoder(dictionary)

    for i, result in enumerate(
        [
            (),
            (),
            (),
            (),
            (),
            ("tom",),
            ("mom", "too"),
            (),
            ("ooo",),
            (),
            (),
            ("tom tom",),
            ("mom tom", "too tom", "tom mom", "tom too"),
        ]
    ):
        morse = "-" * (i + 1)
        assert set(decoder.decode(morse)) == set(result)

    # Custom Cases
    print("Testing custom cases")
    decoder = Decoder(dictionary)
    for message in [
        "von ate pcs",
        "information they might",
        "international development university",
        "black market prices today",
    ]:
        print(message)
        morse = decoder._encode(message, adversarial=True)
        result = set(decoder.decode(morse))
        assert message in result
