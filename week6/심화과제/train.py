import json
import random
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch
import wandb

# Load the corpus
with open('/kaggle/input/corpus/corpus.json', 'r', encoding='utf-8') as f:
    corpus = json.load(f)

# Split the data into training and validation
train_data, valid_data = train_test_split(corpus, test_size=0.2, random_state=42)

# Preprocess data for training
def preprocess_data(data):
    """Converts list of dictionaries into instruction-response format."""
    return [
        {
            'input_text': sample['instruction'],
            'target_text': sample['response']
        } for sample in data
    ]

train_dataset = preprocess_data(train_data)
valid_dataset = preprocess_data(valid_data)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Tokenize data
# def tokenize_function(data):
#     input_text = data['input_text']
#     target_text = data['target_text']
#     return tokenizer(f"Instruction: {input_text} Response: {target_text}", padding='max_length', truncation=True)

def tokenize_function(data):
    input_text = data['input_text']
    target_text = data['target_text']
    # Tokenize both instruction and response together
    encoding = tokenizer(f"Instruction: {input_text} Response: {target_text}", padding='max_length', truncation=True)
    # Add labels - same as input_ids shifted to the right
    encoding['labels'] = encoding['input_ids'].copy()  # Add labels for loss calculation
    return encoding


tokenizer.pad_token = tokenizer.eos_token 

train_encodings = [tokenize_function(sample) for sample in train_dataset]
valid_encodings = [tokenize_function(sample) for sample in valid_dataset]


# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    logging_dir="./logs",
    logging_steps=10,
    report_to="wandb"
)

# Trainer setup
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val) for key, val in self.encodings[idx].items()}  
        return item

    def __len__(self):
        return len(self.encodings)

train_dataset = Dataset(train_encodings)
valid_dataset = Dataset(valid_encodings)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset
)

trainer.train()

wandb.init(project='Chapter4-1-advanced')
wandb.run.name = 'gpt-finetuning'
wandb.finish()
