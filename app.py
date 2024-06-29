from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import json
import os

os.environ['FLASK_ENV'] = 'production'

app = Flask(__name__)

# Load the model and tokenizer during application startup
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import bitsandbytes as bnb

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained("nvidia/Llama3-ChatQA-1.5-8B", cache_dir='./model')
model = AutoModelForCausalLM.from_pretrained(
    "nvidia/Llama3-ChatQA-1.5-8B",
    cache_dir='./model',
    quantization_config=bnb_config,
    device_map='auto'
)
model.config.use_cache = True  # Enable cache in inference mode

# Load retriever tokenizer and model
retriever_tokenizer = AutoTokenizer.from_pretrained('nvidia/dragon-multiturn-query-encoder')
query_encoder = AutoModel.from_pretrained('nvidia/dragon-multiturn-query-encoder')
context_encoder = AutoModel.from_pretrained('nvidia/dragon-multiturn-context-encoder')


# memory = json.load(open("memory.json"))["context"]
# Load memory from JSON file with error handling
try:
    with open("memory.json", "r") as f:
        memory = json.load(f)["context"]
except json.JSONDecodeError as e:
    print(f"Error loading memory.json: {e}")
    memory = []

def get_retrieved_context(messages, memory):
    if not memory:
        return ""
    
    # Format query for the retriever
    formatted_query_for_retriever = '\n'.join([
        f"{turn['role']}: {turn['content'] if isinstance(turn['content'], str) else json.dumps(turn['content'])}" 
        for turn in messages
    ]).strip()
    
    query_input = retriever_tokenizer(formatted_query_for_retriever, return_tensors='pt')
    ctx_input = retriever_tokenizer(memory, padding=True, truncation=True, max_length=512, return_tensors='pt')
    
    # Generate embeddings for the query and the context
    query_emb = query_encoder(**query_input).last_hidden_state[:, 0, :]
    ctx_emb = context_encoder(**ctx_input).last_hidden_state[:, 0, :]
    
    # Compute similarity scores using dot product and rank by similarity
    similarities = query_emb.matmul(ctx_emb.transpose(0, 1))
    ranked_results = torch.argsort(similarities, dim=-1, descending=True)
    
    # Retrieve all ranked results
    retrieved_chunks = [memory[idx] for idx in ranked_results.tolist()[0]]
    context = "\n\n".join(retrieved_chunks)
    
    return context

def update_memory(messages):
    def format_message(message):
        if isinstance(message, dict):
            return message.get('content', '')
        elif isinstance(message, str):
            return message
        else:
            return str(message)

    # Get the last user message and AI response
    user_message = next((format_message(msg) for msg in reversed(messages) if msg['role'] == 'user'), '')
    ai_response = next((format_message(msg) for msg in reversed(messages) if msg['role'] == 'assistant'), '')

    if user_message.strip():  # Only add non-empty user messages
        memory.append("User: " + user_message.strip())
    if ai_response.strip():   # Only add non-empty AI responses
        memory.append("Luna: " + ai_response.strip())
    
    with open("memory.json", "w") as f:
        json.dump({"context": memory}, f, indent=4)

def get_formatted_input(messages, context):
    system = "System: You are Luna, a funny and helpful AI assistant with her own personality. Give short, context-aware answers and engage in pleasant conversation. Always respond to the user's latest message."
    #instruction = "Please give a full and complete answer for the question."
    formatted_contexts = context.split("\n\n")
    formatted_input = ""

    for ctx in formatted_contexts:
        conversation = []
        for item in messages:
            role = "User" if item['role'] == 'user' else "Assistant"
            content = item['content'] if isinstance(item['content'], str) else json.dumps(item['content'])
            conversation.append(f"{role}: {content}")
        
        formatted_input += f"{system}\n{ctx}\n" + "\n".join(conversation) + "\nAssistant:\n\n"

    return formatted_input.strip()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    print("Received data:", data)
    usr_input = data.get("message", "")
    user_message = {"role": "user", "content": usr_input}
    messages = data.get("messages", [])  # Get the entire message history
    messages.append(user_message)  # Add the new user message
    
    context = get_retrieved_context(messages, memory)
    formatted_input = get_formatted_input(messages, context)
    
    tokenized_prompt = tokenizer(tokenizer.bos_token + formatted_input, return_tensors="pt").to(model.device)
    
    # Check if tokenized_prompt is not empty
    if tokenized_prompt.input_ids.size(1) == 0:
        return jsonify({"error": "Tokenization resulted in an empty prompt. Please check the input."})
    
    outputs = model.generate(
        input_ids=tokenized_prompt.input_ids, 
        attention_mask=tokenized_prompt.attention_mask, 
        max_new_tokens=256, 
        do_sample=True,  # Enable sampling
        temperature=0.7,  # Adjust temperature for more varied outputs
        top_p=0.95,  # Use top-p sampling
        no_repeat_ngram_size=3  # Avoid repeating 3-grams
    )
    
    response = outputs[0][tokenized_prompt.input_ids.shape[-1]:]
    answer = tokenizer.decode(response, skip_special_tokens=True)

    # Add the AI response to the messages
    messages.append({"role": "assistant", "content": answer})

    # Update memory with the new conversation turn
    update_memory(messages)
    
    print("Sending response:", answer)
    return jsonify({"answer": answer})
if __name__ == "__main__":
    app.run(debug=False)

