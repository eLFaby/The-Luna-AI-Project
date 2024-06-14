from transformers import AutoModelForCausalLM, AutoTokenizer,TextStreamer
from IPython.display import Markdown
import textwrap
import streamlit as st
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path='./Gemma-1.1-2b-instruct-dollyfinetuned/'
tokenizer = AutoTokenizer.from_pretrained(model_path, device_map='auto')
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
class ChatState:
    __START_TURN_USER__ = "<start_of_turn>user\n"
    __START_TURN_MODEL__ = "<start_of_turn>model\n"
    __END_TURN__ = "<end_of_turn>\n"

    def __init__(self, model, tokenizer, system=""):
        self.model = model
        self.tokenizer = tokenizer
        self.system = system
        self.history = []

    def add_to_history_as_user(self, message):
        self.history.append(self.__START_TURN_USER__ + message + self.__END_TURN__)

    def add_to_history_as_model(self, message):
        self.history.append(self.__START_TURN_MODEL__ + message + self.__END_TURN__)

    def get_history(self):
        return "".join(self.history)

    def get_full_prompt(self):
        prompt = self.get_history() + self.__START_TURN_MODEL__
        if len(self.system) > 0:
            prompt = self.system + "\n" + prompt
        return prompt

    def send_message(self, message):
        self.add_to_history_as_user(message)
        prompt = self.get_full_prompt()
        inputs = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        outputs = self.model.generate(inputs, max_length=inputs.shape[1] + 50, num_return_sequences=1)
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = result.replace(prompt, "").strip()
        self.add_to_history_as_model(result)
        return result

def display_chat(prompt, text):
    text = text.replace('<start_of_turn>user\n', '').replace('<start_of_turn>model\n', '').replace('<end_of_turn>', '')

    formatted_prompt = f"<font size='+1' color='brown'>🙋‍♂️<blockquote>{prompt}</blockquote></font>"
    formatted_text = f"<font size='+1' color='teal'>🤖\n\n{text}\n</font>"

    return formatted_prompt + formatted_text

# Initialize the chatbot state
chat = ChatState(model, tokenizer)

# Streamlit app setup
st.title("Chatbot")
st.write("Type your message and press Enter to chat with the bot.")

# User input
user_input = st.text_input("Your message:")

if user_input:
    response = chat.send_message(user_input)
    st.write(display_chat(user_input, response))
    st.experimental_rerun()

# Display chat history
for message in chat.history:
    if message.startswith(chat.__START_TURN_USER__):
        user_message = message.replace(chat.__START_TURN_USER__, "").replace(chat.__END_TURN__, "")
        st.write(f"**You:** {user_message}")
    else:
        bot_message = message.replace(chat.__START_TURN_MODEL__, "").replace(chat.__END_TURN__, "")
        st.write(f"**Bot:** {bot_message}")