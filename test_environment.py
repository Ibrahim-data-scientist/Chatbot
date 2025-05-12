import streamlit as st
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

# Streamlit app layout
st.title("Bulipe Tech Personal Chatbot ")
st.write("Enter a message and get a response from the chatbot.")

user_input = st.text_input("You:", "")

if user_input:
    prompt = f"User: {user_input} <|endoftext|> Response:"
    response = generate_response(prompt)
    st.text_area("Chatbot:", value=response, height=200)
