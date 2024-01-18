from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, Adafactor
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

# Load pre-trained model and tokenizer
model_name = "distilgpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Prepare a tiny dataset
texts = [
    "Hello, how are you?",
    "I am a language model.",
    "This is a test sentence.",
    "Learning to train models is fun."
]
encoded_texts = [tokenizer.encode(text, add_special_tokens=True) for text in texts]

# Creating a simple dataset
class SimpleDataset(Dataset):
    def __len__(self):
        return len(encoded_texts)

    def __getitem__(self, idx):
        return torch.tensor(encoded_texts[idx], dtype=torch.long)

dataset = SimpleDataset()

# Define a simple training loop
def train_model(optimizer_class, optimizer_params, epochs=2):
    optimizer = optimizer_class(model.parameters(), **optimizer_params)
    model.train()

    loss_values = []
    for epoch in range(epochs):
        for batch in dataset:
            inputs, labels = (batch, batch)
            outputs = model(inputs, labels=labels)
            loss = outputs[0]
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_values.append(loss.item())
    
    return loss_values

# Function to plot loss curves
def plot_loss_curves(adam_loss, adafactor_loss):
    plt.plot(adam_loss, label='Adam')
    plt.plot(adafactor_loss, label='Adafactor')
    plt.title('Loss Curves')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Training parameters
adam_params = {'lr': 5e-5}
adafactor_params = {}

# Running the training loop with Adam optimizer
adam_loss = train_model(AdamW, adam_params)

# Reset the model to its initial state
model = GPT2LMHeadModel.from_pretrained(model_name)

# Running the training loop with Adafactor optimizer
adafactor_loss = train_model(Adafactor, adafactor_params)

# Plot the loss curves
plot_loss_curves(adam_loss, adafactor_loss)
