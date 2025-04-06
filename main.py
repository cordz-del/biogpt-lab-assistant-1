import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "microsoft/BioGPT-Large"  # or another BioGPT variant
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

def generate_bio_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    st.title("BioGPT Lab Assistant")
    user_question = st.text_area("Ask about genes, proteins, or upload data")
    if st.button("Generate Answer"):
        answer = generate_bio_answer(user_question)
        st.write(answer)

if __name__ == "__main__":
    main()
