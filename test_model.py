from llama_cpp import Llama
import os
from dotenv import load_dotenv

from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()

# Authenticate using token from .env
from huggingface_hub import login
login(token=os.getenv("HF_TOKEN"))

model_name = os.getenv("MODEL")

model_path = os.path.join("models", model_name)

#llm = Llama(model_path=model_path, n_ctx=4096, verbose=False)


# llm = Llama.from_pretrained(
# 	repo_id="QuantFactory/Llama-3.2-3B-GGUF",
# 	filename="Llama-3.2-3B.Q2_K.gguf",
# )

# response = llm("Q: What is the capital of Italy?\nA:", max_tokens=50)
# print(response["choices"][0]["text"].strip())


### Primo test con Mistral .gguf (models folder)

# Prompt di test
#prompt = "Ciao! Come stai oggi?"

# Generazione
# output = llm.create_completion(
#     prompt=prompt,
#     max_tokens=100,
#     temperature=0.7,
#     top_p=0.9,
#     stop=["Q:", "\n\n"]
# )
#print(output["choices"][0]["text"].strip())

### Secondo test usando funzione diversa ´create_chat_completion´

# output = llm.create_chat_completion(
#     messages=[
#         {"role": "system", "content": "Sei un assistente molto utile che risponde in italiano corretto."},
#         {"role": "user", "content": "Ciao! Come stai oggi?"}
#     ],
#     max_tokens=200,
#     temperature=0.7,
#     top_p=0.95
# )
# print(output["choices"][0]["message"]["content"].strip())

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")

# Create instruction prompt with proper formatting
instruction = """[INST] 
Write a creative short story about a robot learning to appreciate human art in a post-apocalyptic world.
Keep the story under 300 words and include a twist ending.
[/INST]"""

# Tokenize input
inputs = tokenizer(instruction, return_tensors="pt")

# Generate response with adjusted parameters
outputs = model.generate(
    inputs.input_ids,
    max_new_tokens=500,
    temperature=0.8,
    top_p=0.95,
    repetition_penalty=1.1,
    do_sample=True
)

# Decode and print output
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated Story:\n", response.split("[/INST]")[-1].strip())