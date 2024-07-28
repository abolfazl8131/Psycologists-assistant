from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")


   
model = AutoModelForCausalLM.from_pretrained("./psyco_assistant")
tokenizer = AutoTokenizer.from_pretrained("./psyco_assistant")
inputs = tokenizer(
    [
      "<HUMAN>: what is mental illness in psycology?"
    ],
    return_tensors="pt",

).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=128)
print(outputs)
print(tokenizer.batch_decode(outputs))
