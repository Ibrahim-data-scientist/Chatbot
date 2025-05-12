from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

model_name = "google/flan-t5-base"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

print(f"Model {model_name} loaded with 4-bit quantization.")