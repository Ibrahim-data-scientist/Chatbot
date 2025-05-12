import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

# Load GPT-2 tokenizer and model
model_name = "gpt2"  # You can use "gpt2-medium" for a larger model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Tokenize the dataset
def load_dataset(file_path, tokenizer):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128  # Block size can be adjusted based on your needs
    )

def data_collator():
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Set to False because GPT-2 is not trained with masked language modeling (mlm)
    )

# Load the dataset and collator
train_dataset = load_dataset('formatted_conversations.txt', tokenizer)
collator = data_collator()

# Set up the training arguments
training_args = TrainingArguments(
    output_dir='./gpt2-finetuned-chatbot',
    overwrite_output_dir=True,
    num_train_epochs=10,  # You can increase this depending on your dataset size
    per_device_train_batch_size=4,  # Adjust based on available GPU memory
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collator,
    train_dataset=train_dataset
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./gpt2-finetuned-chatbot")
tokenizer.save_pretrained("./gpt2-finetuned-chatbot")

