import openai
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_google_genai import ChatGoogleGenerativeAI

# Streamlit UI setup
st.set_page_config(page_title="Gemini Chatbot", page_icon="🤖")
st.title("🤖 Gemini Chatbot (LangChain + OpenAI Summary)")
st.info("Provide your API keys below. All data stays in your session. Click 'End Chat' to get summary & sentiment.")

# API key input
gemini_key = st.text_input("🔐 Enter Gemini API Key", type="password")
openai_key = st.text_input("🔐 Enter OpenAI API Key (v0.28.1)", type="password")

if gemini_key and openai_key:
    openai.api_key = openai_key  # Set OpenAI key

    # Initialize memory/chatbot once
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(return_messages=True)
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = ConversationChain(
            llm=ChatGoogleGenerativeAI(
                model="gemini-pro", google_api_key=gemini_key
            ),
            memory=st.session_state.memory,
            verbose=False
        )

    # Input prompt
    user_input = st.text_input("You:", key="user_input")
    if user_input:
        response = st.session_state.chatbot.run(user_input)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))
        st.text_area("Bot:", value=response, height=100)

    # Display chat history
    if st.session_state.chat_history:
        with st.expander("🗂️ Chat History"):
            for speaker, message in st.session_state.chat_history:
                st.markdown(f"**{speaker}:** {message}")

    # End Chat
    if st.button("🔚 End Chat"):
        chat_text = "\n".join([f"{s}: {m}" for s, m in st.session_state.chat_history])

        try:
            # OpenAI v0.28.1 summarization
            summary_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Summarize the conversation in 150 words."},
                    {"role": "user", "content": chat_text}
                ]
            )
            sentiment_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Determine the sentiment of this conversation. Just answer: Positive, Neutral, or Negative."},
                    {"role": "user", "content": chat_text}
                ]
            )

            summary = summary_response.choices[0].message.content.strip()
            sentiment = sentiment_response.choices[0].message.content.strip()

            st.subheader("📝 Conversation Summary")
            st.write(summary)

            st.subheader("📊 Sentiment")
            st.write(sentiment)

        except Exception as e:
            st.error(f"Something went wrong: {e}")

        # Reset session
        st.session_state.chat_history = []
        st.session_state.memory.clear()
        del st.session_state.chatbot
