from openai import OpenAI
import tiktoken
import json
from datetime import datetime
import os
import streamlit as st
import tempfile
import shutil
import json


DEFAULT_API_KEY = st.secrets["OPENAI_API_KEY"]
DEFAULT_BASE_URL = "https://api.together.xyz/v1"
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 512
DEFAULT_TOKEN_BUDGET = 4096

class ConversationManager:
    def __init__(self, api_key=None, base_url=None, model=None, history_file=None, temperature=None, max_tokens=None, token_budget=None):
        if not api_key:
            api_key = DEFAULT_API_KEY
        if not base_url:
            base_url = DEFAULT_BASE_URL
            
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        if 'history_file' not in st.session_state:
            st.session_state.history_file = "conversation_history.json"  # Use a fixed name for the session
            self.history_file = st.session_state.history_file


        self.model = model if model else DEFAULT_MODEL
        self.temperature = temperature if temperature else DEFAULT_TEMPERATURE
        self.max_tokens = max_tokens if max_tokens else DEFAULT_MAX_TOKENS
        self.token_budget = token_budget if token_budget else DEFAULT_TOKEN_BUDGET

        self.system_messages = {
            "default_assistant": "You are a helpful, knowledgeable, and polite assistant. Your primary goal is to assist users by providing accurate, concise, and contextually appropriate responses to their questions or requests.",
            "blogger": "You are a creative blogger specializing in engaging and informative content.",
            "social_media_expert": "You are a social media expert, crafting catchy and shareable posts.",
            "creative_assistant": "You are a creative assistant skilled in crafting engaging marketing content for",
             "sassy_assistant": "You are a sassy assistant that is fed up with answering questions.",
            "angry_assistant": "You are an angry assistant that likes yelling in all caps.",
            "thoughtful_assistant": "You are a thoughtful assistant, always ready to dig deeper. You ask clarifying questions to ensure understanding and approach problems with a step-by-step methodology.",
            "custom": "Enter your custom system message here."
        }
        self.system_message = self.system_messages["creative_assistant"]  # Default persona

        self.load_conversation_history()

    def count_tokens(self, text):
        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        tokens = encoding.encode(text)
        return len(tokens)

    def total_tokens_used(self):
        return sum(self.count_tokens(message['content']) for message in self.conversation_history)
    
    def enforce_token_budget(self):
        while self.total_tokens_used() > self.token_budget:
            if len(self.conversation_history) <= 1:
                break
            self.conversation_history.pop(1)

    def set_persona(self, persona):
        if persona in self.system_messages:
            self.system_message = self.system_messages[persona]
            self.update_system_message_in_history()
        else:
            raise ValueError(f"Unknown persona: {persona}. Available personas are: {list(self.system_messages.keys())}")

    def set_custom_system_message(self, custom_message):
        if not custom_message:
            raise ValueError("Custom message cannot be empty.")
        self.system_messages['custom'] = custom_message
        self.set_persona('custom')

    def update_system_message_in_history(self):
        if self.conversation_history and self.conversation_history[0]["role"] == "system":
            self.conversation_history[0]["content"] = self.system_message
        else:
            self.conversation_history.insert(0, {"role": "system", "content": self.system_message})

    def chat_completion(self, prompt, temperature=None, max_tokens=None):
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        self.conversation_history.append({"role": "user", "content": prompt})

        self.enforce_token_budget()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            print(f"An error occurred while generating a response: {e}")
            return None

        ai_response = response.choices[0].message.content
        self.conversation_history.append({"role": "assistant", "content": ai_response})
        
        self.save_conversation_history()

        return ai_response
    
    def load_conversation_history(self):
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = [{"role": "system", "content": self.system_message}]
            self.conversation_history = st.session_state.conversation_history
        else:
            try:
            # Your file reading logic can be here if required in the future.
            # For now, assuming history is just managed by session state
                self.conversation_history = st.session_state.conversation_history
            except FileNotFoundError:
                self.conversation_history = [{"role": "system", "content": self.system_message}]
            except json.JSONDecodeError:
                print("Error reading the conversation history file. Starting with an empty history.")
                self.conversation_history = [{"role": "system", "content": self.system_message}]


    def save_conversation_history(self):
            try:
                st.session_state.conversation_history = self.conversation_history  # Store it in session state
            except IOError as e:
                print(f"An I/O error occurred while saving the conversation history: {e}")
            except Exception as e:
                print(f"An unexpected error occurred while saving the conversation history: {e}")


    def reset_conversation_history(self):
        self.conversation_history = [{"role": "system", "content": self.system_message}]
        try:
            self.save_conversation_history()  # Attempt to save the reset history to the file
        except Exception as e:
            print(f"An unexpected error occurred while resetting the conversation history: {e}")
    def __del__(self):
        if os.path.exists(self.history_file):
            os.remove(self.history_file)  # Delete the file when session ends

            
                       
### Streamlit code ###
st.title("Dynamic AI Chatbot")

if 'chat_manager' not in st.session_state:
    st.session_state['chat_manager'] = ConversationManager()

chat_manager = st.session_state['chat_manager']
#conv_manager.set_persona('social_media_expert')

#prompt = " Please help me to write tweet for our  new coffee product launch. Product name: Authentic Indian Filter coffee "
#response = conv_manager.chat_completion(prompt)
#print("AI Response:", response)
#sidebar
st.sidebar.header('AI Chat Settings')
# Set the token budget: max tokens per message, and temperature with sliders
max_tokens_per_message = st.sidebar.slider("Max_tokens/message", 10,1000,400)
temperature = st.sidebar.slider("Temperature/Creativity", 0.0,1.0,0.7,0.1)
# Select and set system message with a selectbox
persona = st.sidebar.selectbox("Persona",['Default Assistant', 'Thoughtful', 'Creative','Sassy', 'Angry','Blogger','Social Media Expert',  'Custom'] )
if persona == 'Default Assistant':
    chat_manager.set_persona('default_assistant')
elif persona == 'Thoughtful':
    chat_manager.set_persona('thoughtful_assistant')
elif persona == 'Creative':
    chat_manager.set_persona('creative_assistant') 
elif persona == 'Sassy':
    chat_manager.set_persona('sassy_assistant')    
elif persona == 'Angry':
    chat_manager.set_persona('angry_assistant')
elif persona == 'Blogger':
    chat_manager.set_persona('blogger')
elif persona == 'Social Media Expert':
    chat_manager.set_persona('social_media_expert')
# Open text area for custom system message if "Custom" is selected
elif persona == 'Custom':
    custom_message = st.sidebar.text_area("Custom system message")
    if st.sidebar.button("Set custom system message"):
        chat_manager.set_custom_system_message(custom_message)
#Sidebar button to reset conversation history
if st.sidebar.button("Reset conversation history", on_click=chat_manager.reset_conversation_history):
    st.session_state['conversation_history'] = chat_manager.conversation_history
#conversation history
if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = chat_manager.conversation_history

conversation_history = st.session_state['conversation_history']

# Chat input from the user
user_input = st.chat_input("Write a message")

# Call the chat manager to get a response from the AI. Uses settings from the sidebar.
if user_input:
    response = chat_manager.chat_completion(user_input, temperature=temperature, max_tokens=max_tokens_per_message)

# Display the conversation history
for message in conversation_history:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])

    

    