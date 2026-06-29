import torch
import torch.nn as nn
import json
import random
import sys
import os

# add parent folder to path so we can import our own files
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.tokenizer import encode_sequence, encode_number, VOCAB_SIZE
from model.model import PatternMind


BATCH_SIZE = 32  # sequences per batch
EPOCHS = 20  # how many times we pass through full dataset
LEARNING_RATE = 0.0001  # how big each weight update is
EMBED_DIM = 128  # vector size for each token
NUM_HEADS = 8  # attention heads
NUM_BLOCKS = 4  # transformer blocks
FF_DIM = 256  # feed forward layer size
MAX_SEQ_LEN = 7  # SOS + 5 numbers + EOS
DROPOUT = 0.1  # randomly turn off 10% neurons during training
CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint.pt")  # where to save progress


def load_dataset(path="data/dataset.json"):
    with open(path, "r") as f:
        data = json.load(f)
    random.shuffle(data)  # shuffle so model doesn't learn order
    return data


# ─────────────────────────────────────────
# PREPARE BATCH
# converts raw sequences into tensors
# the model only understands tensors
# ─────────────────────────────────────────
def prepare_batch(batch):
    inputs = []
    targets = []

    for sample in batch:
        # encode input sequence → token indices
        encoded = encode_sequence(sample["input"])
        inputs.append(encoded)

        # encode target number → token index
        target_token = encode_number(sample["target"])
        targets.append(target_token)

    # convert lists to tensors
    # tensors are like numpy arrays but pytorch can train on them
    inputs_tensor = torch.tensor(inputs, dtype=torch.long)
    targets_tensor = torch.tensor(targets, dtype=torch.long)

    return inputs_tensor, targets_tensor


def train():
    print("Loading dataset...")
    dataset = load_dataset()
    print(f"Loaded {len(dataset)} samples")

    # ── create model ──
    model = PatternMind(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        ff_dim=FF_DIM,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT,
    )

    # count and print total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"PatternMind parameters: {total_params:,}")

    # ── loss function ──
    # CrossEntropyLoss — standard for classification
    # ignore_index=0 means ignore PAD tokens in loss calculation
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # ── optimiser ──
    # Adam — adjusts learning rate automatically, works great out of the box
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ── load checkpoint if exists ──
    # this lets us resume training if colab session dies
    start_epoch = 0
    if os.path.exists(CHECKPOINT_PATH):
        print("Resuming from checkpoint...")
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint["model"])
        optimiser.load_state_dict(checkpoint["optimiser"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")

    # ── training loop ──
    for epoch in range(start_epoch, EPOCHS):
        model.train()  # set model to training mode (enables dropout)

        total_loss = 0
        total_correct = 0
        total_samples = 0

        # split dataset into batches
        random.shuffle(dataset)
        batches = [
            dataset[i : i + BATCH_SIZE]
            for i in range(0, len(dataset), BATCH_SIZE)
        ]

        for batch_idx, batch in enumerate(batches):
            # ── Step 1: prepare batch ──
            inputs, targets = prepare_batch(batch)

            # ── Step 2: forward pass ──
            # send inputs through model, get predictions
            outputs = model(inputs)
            # outputs shape: (batch_size, vocab_size)

            # ── Step 3: calculate loss ──
            loss = criterion(outputs, targets)

            # ── Step 4: backward pass ──
            # zero out old gradients first — important!
            optimiser.zero_grad()
            # calculate new gradients
            loss.backward()

            # ── Step 5: update weights ──
            optimiser.step()

            # ── track progress ──
            total_loss += loss.item()

            # check how many predictions were correct
            predicted = outputs.argmax(dim=-1)  # pick highest scoring token
            total_correct += (predicted == targets).sum().item()
            total_samples += len(batch)

            # print progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                accuracy = total_correct / total_samples * 100
                print(
                    f"  Epoch {epoch + 1}/{EPOCHS} | "
                    f"Batch {batch_idx + 1}/{len(batches)} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Accuracy: {accuracy:.1f}%"
                )

        # ── end of epoch summary ──
        avg_loss = total_loss / len(batches)
        accuracy = total_correct / total_samples * 100
        print(
            f"\nEpoch {epoch + 1}/{EPOCHS} complete — "
            f"Loss: {avg_loss:.4f} | "
            f"Accuracy: {accuracy:.1f}%\n"
        )

        # ── save checkpoint after every epoch ──
        # if colab dies, we resume from here
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimiser": optimiser.state_dict(),
                "loss": avg_loss,
            },
            CHECKPOINT_PATH,
        )
        print(f"Checkpoint saved → {CHECKPOINT_PATH}")

    print("\nTraining complete!")

    # save final model separately
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "patternmind_final.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Final model saved → {save_path}")


if __name__ == "__main__":
    train()
