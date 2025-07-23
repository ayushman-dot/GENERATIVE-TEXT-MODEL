!pip install transformers
!pip install torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

def generate_text(prompt, max_length=200):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs, 
        max_length=max_length, 
        do_sample=True, 
        top_k=50, 
        top_p=0.95, 
        temperature=0.8
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

prompt = "The future of artificial intelligence in education is"
generated_text = generate_text(prompt)
print(generated_text)
