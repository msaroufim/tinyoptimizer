from transformers import GPT2LMHeadModel, GPT2Tokenizer, Adafactor, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.optim import AdamW  # Using PyTorch's AdamW
from tqdm import tqdm

# torch.set_manual_seed(0)

from datetime import datetime, timedelta
import logging
import socket

logging.basicConfig(
   format="%(levelname)s:%(asctime)s %(message)s",
   level=logging.INFO,
   datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"

# Keep a max of 100,000 alloc/free events in the recorded history
# leading up to the snapshot.
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000

def start_record_memory_history() -> None:
   if not torch.cuda.is_available():
       logger.info("CUDA unavailable. Not recording memory history")
       return

   logger.info("Starting snapshot record_memory_history")
   torch.cuda.memory._record_memory_history(
       max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
   )

def stop_record_memory_history() -> None:
   if not torch.cuda.is_available():
       logger.info("CUDA unavailable. Not recording memory history")
       return

   logger.info("Stopping snapshot record_memory_history")
   torch.cuda.memory._record_memory_history(enabled=None)

def export_memory_snapshot() -> None:
   if not torch.cuda.is_available():
       logger.info("CUDA unavailable. Not exporting memory snapshot")
       return

   # Prefix for file names.
   host_name = socket.gethostname()
   timestamp = datetime.now().strftime(TIME_FORMAT_STR)
   file_prefix = f"{host_name}_{timestamp}"

   try:
       logger.info(f"Saving snapshot to local file: {file_prefix}.pickle")
       torch.cuda.memory._dump_snapshot(f"{file_prefix}.pickle")
   except Exception as e:
       logger.error(f"Failed to capture memory snapshot {e}")
       return

# Load pre-trained model and tokenizer
model_name = "distilgpt2"
model = GPT2LMHeadModel.from_pretrained(model_name).cuda()
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set the padding token
tokenizer.pad_token = tokenizer.eos_token

# Load WikiText-2 dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
dataset, val_dataset = dataset.train_test_split(test_size=0.99).values()

# Tokenize the texts with padding, truncation, and return PyTorch tensors
def encode(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512, return_tensors="pt")

tokenized_dataset = dataset.map(encode, batched=True)

# Convert the dataset into a format suitable for the DataLoader
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define the training loop
def train_model(optimizer_class, optimizer_params, epochs=1):
    optimizer = optimizer_class(model.parameters(), **optimizer_params)
    model.train()

    loss_values = []
    dataloader = DataLoader(tokenized_dataset, batch_size=4, shuffle=True, collate_fn=data_collator)
    for epoch in range(epochs):
        for batch in tqdm(dataloader):
            inputs, labels = (batch["input_ids"].to(model.device), batch["input_ids"].to(model.device))
            model.zero_grad()
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            loss_values.append(loss.item())
            print(loss.item())
    return loss_values

# Training parameters
adam_params = {'lr': 1e-3}

start_record_memory_history()
# adafactor_params = {'lr': 1e-3}  # Adafactor can typically use a higher learning rate

# Train with Adam optimizer
adam_loss = train_model(AdamW, adam_params)

# Reset the model to its initial state
# model = GPT2LMHeadModel.from_pretrained(model_name).cuda()

# Train with Adafactor optimizer
# adafactor_loss = train_model(Adafactor, {})

export_memory_snapshot()
stop_record_memory_history()

# Plot the loss curves
def plot_loss_curves(adam_loss, adafactor_loss):
    # plt.plot(adam_loss, label='Adam')
    plt.plot(adafactor_loss, label='Adafactor')
    plt.title('Loss Curves')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('hf.png')
    plt.show()

plot_loss_curves(adam_loss, adafactor_loss)