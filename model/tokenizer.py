"""
tokenizer.py
Converts numbers ↔ token indices
"""

# Special tokens
PAD_TOKEN = 0   # padding (to make sequences equal length)
SOS_TOKEN = 1   # start of sequence
EOS_TOKEN = 2   # end of sequence

# Number range we support (0 to 999)
MIN_NUM = 0
MAX_NUM = 999

# Offset — first 3 indices are special tokens
# so number 0 maps to index 3, number 1 to index 4, etc.
OFFSET = 3

# Total vocabulary size
# 1000 numbers (0-999) + 3 special tokens
VOCAB_SIZE = MAX_NUM - MIN_NUM + 1 + OFFSET  # = 1003


def encode_number(n):
    """
    Convert a number to its token index.
    Example: 8 → 11  (8 + offset of 3)
    """
    if n < MIN_NUM or n > MAX_NUM:
        raise ValueError(f"Number {n} out of supported range {MIN_NUM}-{MAX_NUM}")
    return int(n) + OFFSET


def decode_number(token):
    """
    Convert a token index back to a number.
    Example: 11 → 8  (11 - offset of 3)
    """
    if token < OFFSET:
        raise ValueError(f"Token {token} is a special token, not a number")
    return token - OFFSET


def encode_sequence(seq):
    """
    Convert a list of numbers into token indices.
    Wraps with SOS and EOS.
    Example: [2, 4, 8, 16] → [1, 5, 7, 11, 19, 2]
    """
    tokens = [SOS_TOKEN]
    tokens += [encode_number(n) for n in seq]
    tokens += [EOS_TOKEN]
    return tokens


def decode_sequence(tokens):
    """
    Convert token indices back to numbers.
    Ignores special tokens.
    Example: [1, 5, 7, 11, 19, 2] → [2, 4, 8, 16]
    """
    numbers = []
    for t in tokens:
        if t in (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN):
            continue
        numbers.append(decode_number(t))
    return numbers


def pad_sequence(tokens, max_len):
    """
    Pad a token sequence to max_len with PAD_TOKEN.
    Example: [1, 5, 7, 2] padded to 8 → [1, 5, 7, 2, 0, 0, 0, 0]
    """
    if len(tokens) >= max_len:
        return tokens[:max_len]
    return tokens + [PAD_TOKEN] * (max_len - len(tokens))


if __name__ == "__main__":
    print(f"Vocabulary size: {VOCAB_SIZE}")
    print()

    # test a sequence
    seq = [2, 4, 8, 16]
    encoded = encode_sequence(seq)
    decoded = decode_sequence(encoded)

    print(f"Original sequence : {seq}")
    print(f"Encoded tokens    : {encoded}")
    print(f"Decoded back      : {decoded}")
    print()

    # test padding
    padded = pad_sequence(encoded, max_len=10)
    print(f"Padded to length 10: {padded}")
    print()

    # test a target number
    target = 32
    target_token = encode_number(target)
    decoded_target = decode_number(target_token)
    print(f"Target number   : {target}")
    print(f"Target token    : {target_token}")
    print(f"Decoded target  : {decoded_target}")