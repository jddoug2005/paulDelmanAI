from transformers import pipeline
import torch

# This downloads a small, fast model (1.1 billion parameters)
# It requires about 2GB of disk space and runs on most modern CPUs/GPUs
print("Loading AI (this may take a minute on the first run)...")
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Set up the chat pipeline
pipe = pipeline(
    "text-generation", 
    model=model_id, 
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, 
    device_map="auto"
)

def chat():
    print("\n--- AI Started! Type 'quit' to exit ---\n")
    messages = []
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            break
            
        messages.append({"role": "user", "content": user_input})
        
        # Format the prompt for the model
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Generate response
        outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        response = outputs[0]["generated_text"].split("<|assistant|>")[-1].strip()
        
        print(f"AI: {response}\n")
        messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    chat()