from transformers import GPT2LMHeadModel, GPT2Tokenizer, Adafactor, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.optim import AdamW  # Using PyTorch's AdamW
from tqdm import tqdm

# torch.set_manual_seed(0)


# Load pre-trained model and tokenizer
model_name = "distilgpt2"
model = GPT2LMHeadModel.from_pretrained(model_name).cuda()
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set the padding token
tokenizer.pad_token = tokenizer.eos_token

# Load WikiText-2 dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
dataset, val_dataset = dataset.train_test_split(test_size=0.9).values()

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
# adafactor_params = {'lr': 1e-3}  # Adafactor can typically use a higher learning rate

# Train with Adam optimizer
adam_loss = train_model(AdamW, adam_params)

# Reset the model to its initial state
model = GPT2LMHeadModel.from_pretrained(model_name).cuda()

# Train with Adafactor optimizer
adafactor_loss = train_model(Adafactor, {})

# Plot the loss curves
def plot_loss_curves(adam_loss, adafactor_loss):
    plt.plot(adam_loss, label='Adam')
    plt.plot(adafactor_loss, label='Adafactor')
    plt.title('Loss Curves')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('hf.png')
    plt.show()

plot_loss_curves(adam_loss, adafactor_loss)