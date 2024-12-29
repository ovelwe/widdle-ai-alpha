from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "IlyaGusev/saiga_llama3_8b"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

def chat(prompt, history=None, max_length=512):
    if history is None:
        history = []

    history.append(f"ты: {prompt}")
    context = "\n".join(history) + "\nвидл:"

    inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    new_reply = response.split("widdle-alpha:")[-1].strip()
    history.append(f"widdle-alpha: {new_reply}")
    return new_reply, history


if __name__ == "__main__":
    print("работает")
    dialog_history = []

    while True:
        user_input = input("Я: ")

        reply, dialog_history = chat(user_input, dialog_history)
        print(f"widdle-alpha: {reply}")
