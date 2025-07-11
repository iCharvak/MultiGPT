import streamlit as st
import google.generativeai as genai
from langchain_openai import OpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
import os
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="LangChain Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        background-color: #f0f2f6;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .bot-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    
    .summary-box {
        background-color: #fff3e0;
        border: 2px solid #ff9800;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .sentiment-positive {
        color: #4caf50;
        font-weight: bold;
    }
    
    .sentiment-negative {
        color: #f44336;
        font-weight: bold;
    }
    
    .sentiment-neutral {
        color: #ff9800;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'conversation_started' not in st.session_state:
        st.session_state.conversation_started = False
    if 'gemini_api_key' not in st.session_state:
        st.session_state.gemini_api_key = ""
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = ""
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory()
    if 'conversation_ended' not in st.session_state:
        st.session_state.conversation_ended = False

def configure_gemini(api_key):
    """Configure Gemini API"""
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-pro')
    except Exception as e:
        st.error(f"Error configuring Gemini: {str(e)}")
        return None

def get_gemini_response(model, message, conversation_history):
    """Get response from Gemini API"""
    try:
        # Create context from conversation history
        context = "Previous conversation:\n"
        for msg in conversation_history[-5:]:  # Last 5 messages for context
            role = "User" if msg["role"] == "user" else "Assistant"
            context += f"{role}: {msg['content']}\n"
        
        # Add current message
        full_prompt = f"{context}\nUser: {message}\nAssistant:"
        
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        st.error(f"Error getting Gemini response: {str(e)}")
        return "I apologize, but I encountered an error processing your request."

def summarize_conversation(openai_api_key, messages):
    """Summarize conversation using OpenAI API with LangChain"""
    try:
        # Initialize OpenAI with LangChain
        llm = OpenAI(
            api_key=openai_api_key,
            model_name="gpt-3.5-turbo-instruct",
            temperature=0.3,
            max_tokens=200
        )
        
        # Create conversation text
        conversation_text = ""
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation_text += f"{role}: {msg['content']}\n"
        
        # Create prompt template for summarization
        summary_template = """
        Please summarize the following conversation in exactly 150 words. Focus on the main topics discussed and key points exchanged.

        Conversation:
        {conversation}

        Summary (150 words):
        """
        
        prompt = PromptTemplate(
            input_variables=["conversation"],
            template=summary_template
        )
        
        # Create chain and get summary
        formatted_prompt = prompt.format(conversation=conversation_text)
        summary = llm.invoke(formatted_prompt)
        
        return summary.strip()
        
    except Exception as e:
        st.error(f"Error summarizing conversation: {str(e)}")
        return "Unable to generate summary due to an error."

def analyze_sentiment(openai_api_key, messages):
    """Analyze sentiment of the conversation using OpenAI API"""
    try:
        llm = OpenAI(
            api_key=openai_api_key,
            model_name="gpt-3.5-turbo-instruct",
            temperature=0.1,
            max_tokens=50
        )
        
        # Create conversation text
        conversation_text = ""
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation_text += f"{role}: {msg['content']}\n"
        
        # Create prompt for sentiment analysis
        sentiment_prompt = f"""
        Analyze the overall sentiment of this conversation. Respond with only one word: "Positive", "Negative", or "Neutral".

        Conversation:
        {conversation_text}

        Sentiment:
        """
        
        sentiment = llm.invoke(sentiment_prompt)
        return sentiment.strip()
        
    except Exception as e:
        st.error(f"Error analyzing sentiment: {str(e)}")
        return "Neutral"

def display_message(message, is_user=True):
    """Display a chat message with proper styling"""
    if is_user:
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>You:</strong> {message}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message bot-message">
            <strong>Assistant:</strong> {message}
        </div>
        """, unsafe_allow_html=True)

def main():
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 LangChain Chatbot</h1>
        <p>Powered by Gemini AI with OpenAI Summarization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for API keys
    with st.sidebar:
        st.header("🔑 API Configuration")
        
        # Gemini API Key
        gemini_key = st.text_input(
            "Enter Gemini API Key:",
            type="password",
            value=st.session_state.gemini_api_key,
            help="Get your API key from Google AI Studio"
        )
        
        # OpenAI API Key
        openai_key = st.text_input(
            "Enter OpenAI API Key:",
            type="password",
            value=st.session_state.openai_api_key,
            help="Required for conversation summarization"
        )
        
        # Start conversation button
        if st.button("🚀 Start Conversation", type="primary"):
            if gemini_key and openai_key:
                st.session_state.gemini_api_key = gemini_key
                st.session_state.openai_api_key = openai_key
                st.session_state.conversation_started = True
                st.session_state.conversation_ended = False
                st.success("Conversation started!")
                st.rerun()
            else:
                st.error("Please provide both API keys to start the conversation.")
        
        # End conversation button
        if st.session_state.conversation_started and not st.session_state.conversation_ended:
            if st.button("🛑 End Conversation", type="secondary"):
                st.session_state.conversation_ended = True
                st.rerun()
        
        # Reset button
        if st.button("🔄 Reset Chat"):
            st.session_state.messages = []
            st.session_state.conversation_started = False
            st.session_state.conversation_ended = False
            st.session_state.memory = ConversationBufferMemory()
            st.rerun()
    
    # Main chat area
    if not st.session_state.conversation_started:
        st.info("👈 Please enter your API keys in the sidebar and click 'Start Conversation' to begin.")
        
        # Instructions
        st.markdown("""
        ### How to get API keys:
        
        **Gemini API Key:**
        1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Sign in with your Google account
        3. Create a new API key
        4. Copy the key and paste it in the sidebar
        
        **OpenAI API Key:**
        1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
        2. Sign in or create an account
        3. Create a new API key
        4. Copy the key and paste it in the sidebar
        """)
    
    elif st.session_state.conversation_ended:
        st.success("Conversation ended! Generating summary...")
        
        if st.session_state.messages:
            # Generate summary and sentiment
            with st.spinner("Generating summary and analyzing sentiment..."):
                summary = summarize_conversation(st.session_state.openai_api_key, st.session_state.messages)
                sentiment = analyze_sentiment(st.session_state.openai_api_key, st.session_state.messages)
            
            # Display summary
            st.markdown("""
            <div class="summary-box">
                <h3>📝 Conversation Summary</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.write(summary)
            
            # Display sentiment with color coding
            sentiment_class = f"sentiment-{sentiment.lower()}"
            st.markdown(f"""
            <div class="summary-box">
                <h3>😊 Conversation Sentiment</h3>
                <p class="{sentiment_class}">Overall Sentiment: {sentiment}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display conversation history
            st.markdown("### 💬 Conversation History")
            for message in st.session_state.messages:
                display_message(message["content"], message["role"] == "user")
        else:
            st.info("No messages to summarize.")
    
    else:
        # Configure Gemini
        gemini_model = configure_gemini(st.session_state.gemini_api_key)
        
        if gemini_model:
            # Display conversation history
            for message in st.session_state.messages:
                display_message(message["content"], message["role"] == "user")
            
            # Chat input
            user_input = st.chat_input("Type your message here...")
            
            if user_input:
                # Add user message
                st.session_state.messages.append({"role": "user", "content": user_input})
                display_message(user_input, True)
                
                # Get bot response
                with st.spinner("Thinking..."):
                    bot_response = get_gemini_response(gemini_model, user_input, st.session_state.messages)
                
                # Add bot response
                st.session_state.messages.append({"role": "assistant", "content": bot_response})
                display_message(bot_response, False)
                
                # Store in memory for context
                st.session_state.memory.save_context(
                    {"input": user_input},
                    {"output": bot_response}
                )
                
                st.rerun()
        else:
            st.error("Failed to configure Gemini API. Please check your API key.")

if __name__ == "__main__":
    main()