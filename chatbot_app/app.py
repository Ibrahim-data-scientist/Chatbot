from flask import Flask, request, jsonify, render_template
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(__name__)

# Load model and tokenizer
model_path = './model/gpt2-finetuned-chatbot'
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Generation function
def generate_response(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    with torch.no_grad():
        output = model.generate(input_ids, max_length=150, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0], skip_special_tokens=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get("message", "")
    formatted_input = f"User: {user_input} <|endoftext|> Response:"
    bot_response = generate_response(formatted_input)
    return jsonify({"response": bot_response})

if __name__ == '__main__':
    app.run(debug=True)
