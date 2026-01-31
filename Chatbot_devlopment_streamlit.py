from openai import OpenAI
import tiktoken
import json
from datetime import datetime
import os
import streamlit as st

# ===============================
# ENV & SECURITY
# ===============================

# OLD (INSECURE – DO NOT USE)
# DEFAULT_API_KEY = '05381c772b2d42d6d6c4650e20d4681cb36ee257063a61f94b564088d0e4739a'

# NEW (SECURE – ENV VARIABLE)
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY")

if not DEFAULT_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in environment variables")

# ===============================
# MODEL & API CONFIG
# ===============================

# OLD (TOGETHER AI)
# DEFAULT_BASE_URL = "https://api.together.xyz/v1"
# DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

# NEW (OPENAI)
DEFAULT_MODEL = "gpt-4o-mini"

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 512
DEFAULT_TOKEN_BUDGET = 4096

# ===============================
# CONVERSATION MANAGER
# ===============================

class ConversationManager:
    def __init__(
        self,
        api_key=None,
        model=None,
        temperature=None,
        max_tokens=None,
        token_budget=None
    ):

        # OLD (TOGETHER)
        # self.client = OpenAI(
        #     api_key=api_key,
        #     base_url=DEFAULT_BASE_URL
        # )

        # NEW (OPENAI – base_url auto-handled)
        self.client = OpenAI(api_key=DEFAULT_API_KEY)

        self.model = model if model else DEFAULT_MODEL
        self.temperature = temperature if temperature else DEFAULT_TEMPERATURE
        self.max_tokens = max_tokens if max_tokens else DEFAULT_MAX_TOKENS
        self.token_budget = token_budget if token_budget else DEFAULT_TOKEN_BUDGET

        self.system_messages = {
            "default_assistant": "You are a helpful, knowledgeable, and polite assistant.",
            "blogger": "You are a creative blogger specializing in engaging content.",
            "social_media_expert": "You are a social media expert crafting viral posts.",
            "creative_assistant": "You are a creative assistant for marketing content.",
            "sassy_assistant": "You are a sassy assistant.",
            "angry_assistant": "You are an angry assistant that shouts in all caps.",
            "thoughtful_assistant": "You are a thoughtful assistant who reasons step by step.",
            "custom": ""
        }

        self.system_message = self.system_messages["creative_assistant"]

        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = [
                {"role": "system", "content": self.system_message}
            ]

        self.conversation_history = st.session_state.conversation_history

    # ===============================
    # TOKEN HANDLING
    # ===============================

    def count_tokens(self, text):
        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    def total_tokens_used(self):
        return sum(self.count_tokens(m["content"]) for m in self.conversation_history)

    def enforce_token_budget(self):
        while self.total_tokens_used() > self.token_budget:
            if len(self.conversation_history) > 2:
                self.conversation_history.pop(1)
                self.conversation_history.pop(1)
            else:
                break

    # ===============================
    # PERSONA MANAGEMENT
    # ===============================

    def set_persona(self, persona):
        self.system_message = self.system_messages[persona]
        self.conversation_history = [
            {"role": "system", "content": self.system_message}
        ]
        st.session_state.conversation_history = self.conversation_history

    def set_custom_system_message(self, message):
        self.system_messages["custom"] = message
        self.set_persona("custom")

    # ===============================
    # CHAT COMPLETION
    # ===============================

    def chat_completion(self, prompt, temperature=None, max_tokens=None):
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        self.conversation_history.append({"role": "user", "content": prompt})
        self.enforce_token_budget()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation_history,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        ai_response = response.choices[0].message.content
        self.conversation_history.append(
            {"role": "assistant", "content": ai_response}
        )

        st.session_state.conversation_history = self.conversation_history
        return ai_response


# ===============================
# STREAMLIT UI
# ===============================

st.title("Dynamic AI Chatbot (Secure OpenAI Version)")

if "chat_manager" not in st.session_state:
    st.session_state.chat_manager = ConversationManager()

chat_manager = st.session_state.chat_manager

# Sidebar
st.sidebar.header("AI Chat Settings")

max_tokens = st.sidebar.slider("Max tokens", 50, 1500, 900)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)

persona = st.sidebar.selectbox(
    "Persona",
    [
        "default_assistant",
        "thoughtful_assistant",
        "creative_assistant",
        "sassy_assistant",
        "angry_assistant",
        "blogger",
        "social_media_expert",
        "custom",
    ],
)

if persona == "custom":
    custom_msg = st.sidebar.text_area("Custom system message")
    if st.sidebar.button("Apply Custom Persona"):
        chat_manager.set_custom_system_message(custom_msg)
else:
    chat_manager.set_persona(persona)

if st.sidebar.button("Reset Conversation"):
    chat_manager.set_persona(persona)

# Chat input
user_input = st.chat_input("Type your message")

if user_input:
    chat_manager.chat_completion(
        user_input,
        temperature=temperature,
        max_tokens=max_tokens
    )

# Render chat
for msg in chat_manager.conversation_history:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
