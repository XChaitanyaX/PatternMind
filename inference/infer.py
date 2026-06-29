import torch
import sys
import os

# add parent folder to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.tokenizer import encode_sequence, decode_number, VOCAB_SIZE
from model.model import PatternMind

EMBED_DIM = 128
NUM_HEADS = 8
NUM_BLOCKS = 4
FF_DIM = 256
MAX_SEQ_LEN = 7
DROPOUT = 0.1
MODEL_PATH = "training/patternmind_final.pt"


def load_model():
    model = PatternMind(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        ff_dim=FF_DIM,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT,
    )

    # load trained weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))

    # set to eval mode — turns off dropout
    model.eval()

    return model


def predict(model, sequence):
    # encode sequence to tokens
    tokens = encode_sequence(sequence)

    # convert to tensor — shape (1, seq_len)
    input_tensor = torch.tensor([tokens], dtype=torch.long)

    # forward pass — no gradient needed during inference
    with torch.no_grad():
        output = model(input_tensor)

    # get highest scoring token
    predicted_token = output.argmax(dim=-1).item()

    # decode token back to number
    predicted_number = decode_number(predicted_token)

    return predicted_number


if __name__ == "__main__":
    print("Loading PatternMind...")
    model = load_model()
    print("PatternMind ready!\n")
    print("Type a sequence of 5 numbers separated by spaces.")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("Enter sequence: ").strip()

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        # parse input
        try:
            sequence = [int(x) for x in user_input.split()]
        except ValueError:
            print("Please enter valid whole numbers.\n")
            continue

        # validate length
        if len(sequence) != 5:
            print("Please enter exactly 5 numbers.\n")
            continue

        # validate range
        if any(n < 0 or n > 999 for n in sequence):
            print("Numbers must be between 0 and 999.\n")
            continue

        # predict
        next_number = predict(model, sequence)

        # show result
        print(f"\n  Sequence : {sequence}")
        print(f"  Next     : {next_number}")
