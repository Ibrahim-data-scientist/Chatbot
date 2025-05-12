from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained('./gpt2-finetuned')
tokenizer = GPT2Tokenizer.from_pretrained('./gpt2-finetuned')

# Function to generate response
def generate_response(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output = model.generate(input_ids, max_length=150, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Test the chatbot
while True:
  user_input = input("Enter your promt ",)
  response = generate_response(f"User: {user_input} <|endoftext|> Response:")
  print(response)