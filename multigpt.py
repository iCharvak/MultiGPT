import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import google.generativeai as genai
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E2E2E;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        max-width: 80%;
    }
    .user-message {
        background-color: #E3F2FD;
        margin-left: auto;
        border-left: 4px solid #2196F3;
    }
    .assistant-message {
        background-color: #F3E5F5;
        margin-right: auto;
        border-left: 4px solid #9C27B0;
    }
    .chat-container {
        height: 400px;
        overflow-y: auto;
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .summary-box {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    .sentiment-positive {
        color: #4CAF50;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #F44336;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #FF9800;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'conversation_active' not in st.session_state:
        st.session_state.conversation_active = False
    if 'gemini_llm' not in st.session_state:
        st.session_state.gemini_llm = None
    if 'conversation_summary' not in st.session_state:
        st.session_state.conversation_summary = None
    if 'conversation_sentiment' not in st.session_state:
        st.session_state.conversation_sentiment = None

def setup_gemini_llm(api_key):
    """Setup Gemini LLM with API key"""
    try:
        genai.configure(api_key=api_key)
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=api_key,
            temperature=0.7
        )
        return llm, None
    except Exception as e:
        return None, str(e)

def get_conversation_text():
    """Convert conversation history to text format"""
    conversation_text = ""
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            conversation_text += f"User: {msg['content']}\n"
        else:
            conversation_text += f"Assistant: {msg['content']}\n"
    return conversation_text

def summarize_conversation(openai_api_key, conversation_text):
    """Summarize conversation using OpenAI and get sentiment"""
    try:
        # Setup OpenAI LLM
        llm = OpenAI(
            openai_api_key=openai_api_key,
            temperature=0.3,
            max_tokens=200
        )
        
        # Summary prompt
        summary_prompt = PromptTemplate(
            input_variables=["conversation"],
            template="""
            Please summarize the following conversation in exactly 150 words or less. 
            Focus on the main topics discussed and key points:
            
            Conversation:
            {conversation}
            
            Summary:
            """
        )
        
        # Sentiment prompt
        sentiment_prompt = PromptTemplate(
            input_variables=["conversation"],
            template="""
            Analyze the overall sentiment of the following conversation and classify it as one of:
            - Positive
            - Negative  
            - Neutral
            
            Provide only the sentiment word (Positive/Negative/Neutral) and a brief 10-word explanation.
            
            Conversation:
            {conversation}
            
            Sentiment:
            """
        )
        
        # Create chains
        summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
        sentiment_chain = LLMChain(llm=llm, prompt=sentiment_prompt)
        
        # Get summary and sentiment
        summary = summary_chain.run(conversation=conversation_text)
        sentiment = sentiment_chain.run(conversation=conversation_text)
        
        return summary.strip(), sentiment.strip(), None
        
    except Exception as e:
        return None, None, str(e)

def display_chat_message(role, content):
    """Display a chat message with proper styling"""
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>You:</strong> {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>Assistant:</strong> {content}
        </div>
        """, unsafe_allow_html=True)

def main():
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Chat Assistant</h1>', unsafe_allow_html=True)
    
    # Sidebar for API keys
    with st.sidebar:
        st.header("🔑 API Configuration")
        
        # Gemini API Key
        gemini_api_key = st.text_input(
            "Gemini API Key:",
            type="password",
            help="Get your API key from Google AI Studio"
        )
        
        # OpenAI API Key
        openai_api_key = st.text_input(
            "OpenAI API Key:",
            type="password",
            help="Required for conversation summary"
        )
        
        if gemini_api_key and openai_api_key:
            st.success("✅ Both API keys provided!")
        elif gemini_api_key:
            st.info("⚠️ OpenAI API key needed for summary")
        
        st.markdown("---")
        
        # Chat controls
        st.header("💬 Chat Controls")
        
        # Start conversation button
        if not st.session_state.conversation_active:
            if st.button("🚀 Start Conversation", use_container_width=True):
                if gemini_api_key:
                    llm, error = setup_gemini_llm(gemini_api_key)
                    if error:
                        st.error(f"❌ Error setting up Gemini: {error}")
                    else:
                        st.session_state.gemini_llm = llm
                        st.session_state.conversation_active = True
                        st.session_state.messages = []
                        st.rerun()
                else:
                    st.error("❌ Please enter Gemini API key first!")
        
        # End conversation button
        if st.session_state.conversation_active:
            if st.button("🛑 End Conversation", use_container_width=True):
                if openai_api_key and st.session_state.messages:
                    with st.spinner("Generating summary..."):
                        conversation_text = get_conversation_text()
                        summary, sentiment, error = summarize_conversation(
                            openai_api_key, conversation_text
                        )
                        
                        if error:
                            st.error(f"❌ Error generating summary: {error}")
                        else:
                            st.session_state.conversation_summary = summary
                            st.session_state.conversation_sentiment = sentiment
                
                st.session_state.conversation_active = False
                st.rerun()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_summary = None
            st.session_state.conversation_sentiment = None
            st.rerun()
    
    # Main chat interface
    if st.session_state.conversation_active:
        st.markdown("### 💬 Chat Interface")
        
        # Display conversation history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                display_chat_message(message["role"], message["content"])
        
        # User input
        user_input = st.chat_input("Type your message here...")
        
        if user_input:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Get AI response
            try:
                # Create context from conversation history
                context = ""
                for msg in st.session_state.messages[-10:]:  # Last 10 messages for context
                    if msg["role"] == "user":
                        context += f"Human: {msg['content']}\n"
                    else:
                        context += f"Assistant: {msg['content']}\n"
                
                # Generate response
                response = st.session_state.gemini_llm.invoke([
                    HumanMessage(content=f"Context: {context}\n\nCurrent message: {user_input}")
                ])
                
                # Add AI response
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response.content
                })
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")
    
    else:
        # Welcome screen
        st.markdown("### 👋 Welcome to AI Chat Assistant")
        st.markdown("""
        This chatbot uses Google's Gemini AI for conversations and OpenAI for summarization.
        
        **Features:**
        - 🤖 Context-aware conversations using Gemini Pro
        - 📝 Automatic conversation summarization
        - 😊 Sentiment analysis of your chat
        - 💾 Session-based memory
        
        **How to use:**
        1. Enter your Gemini API key in the sidebar
        2. Enter your OpenAI API key (for summary feature)
        3. Click "Start Conversation" to begin
        4. Chat naturally with the AI assistant
        5. Click "End Conversation" to get a summary
        """)
        
        # Display summary if available
        if st.session_state.conversation_summary:
            st.markdown("---")
            st.markdown("### 📋 Conversation Summary")
            st.markdown(f"""
            <div class="summary-box">
                {st.session_state.conversation_summary}
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.conversation_sentiment:
                sentiment_lower = st.session_state.conversation_sentiment.lower()
                if "positive" in sentiment_lower:
                    sentiment_class = "sentiment-positive"
                elif "negative" in sentiment_lower:
                    sentiment_class = "sentiment-negative"
                else:
                    sentiment_class = "sentiment-neutral"
                
                st.markdown(f"""
                **Conversation Sentiment:** 
                <span class="{sentiment_class}">{st.session_state.conversation_sentiment}</span>
                """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        Made with ❤️ using Streamlit, LangChain, Gemini AI, and OpenAI<br>
        Context-aware conversations with intelligent summarization
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()