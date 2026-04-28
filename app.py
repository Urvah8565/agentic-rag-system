import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

# --- 1. Page Configuration ---
st.set_page_config(page_title="Agentic RAG Analytics", page_icon="🤖")
st.title("📈 Agentic RAG System")
st.markdown("An autonomous agent that extracts data from documents and runs mathematical trends.")

# --- 2. Dynamic File Upload ---
# Allows the user to upload their own file instead of relying on hardcoded data
uploaded_file = st.file_uploader("Upload a text file (optional, uses default if empty)", type=["txt"])

if uploaded_file is not None:
    # Overwrite the default data file with the user's new file
    with open("company_data.txt", "wb") as file:
        file.write(uploaded_file.getbuffer())
    
    st.success("File uploaded successfully! Updating the agent's memory...")
    st.cache_resource.clear() # Clear the old memory so FAISS re-indexes the new file

# --- 3. Authentication ---
# Putting the API key in the sidebar keeps the main UI clean
api_key = st.sidebar.text_input("Enter Google Gemini API Key:", type="password")

# --- 4. Core Logic & Agent Setup ---
@st.cache_resource
def initialize_agent(_api_key):
    os.environ["GOOGLE_API_KEY"] = _api_key
    
    # Set up the RAG pipeline
    loader = TextLoader("company_data.txt")  
    docs = loader.load()
    
    # Using HuggingFace for local embeddings to save on API costs
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(docs, embeddings)
    retriever = vector_db.as_retriever()

    # Tool 1: Document Search
    @tool
    def search_company_data(query: str) -> str:
        """Searches the provided text file for sales numbers, Q1/Q2 data, or general info."""
        results = retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in results])

    # Tool 2: Math Calculator
    @tool
    def calculate_trend_slope(x_values: list[float], y_values: list[float]) -> float:
        """Calculates the linear regression slope for X and Y value lists."""
        n = len(x_values)
        if n == 0 or n != len(y_values): 
            return 0.0
        
        # Standard mathematical slope formula
        sum_x, sum_y = sum(x_values), sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_xx = sum(x * x for x in x_values)
        
        denominator = (n * sum_xx - sum_x**2)
        if denominator == 0: 
            return 0.0
            
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return round(slope, 4)

    # Assemble the LangGraph Agent with the Brain (LLM) and its Tools
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    tools_list = [search_company_data, calculate_trend_slope]
    
    return create_react_agent(llm, tools_list)

# --- 5. Main App Execution ---
if api_key:
    with st.spinner("Waking up the agent and building the vector database..."):
        agent = initialize_agent(api_key)
        st.success("System is ready!")
    
    # Default prompt to guide the user on how to talk to the agent
    default_prompt = """Step 1: Use the search tool to extract sales values for January, February, and March.
Step 2: Return ONLY the numbers clearly (example: 10, 20, 30).
Step 3: Use the math tool with X=[1,2,3] and Y=[10,20,30].
Step 4: Calculate the regression slope.
Step 5: Give final answer clearly."""
    
    user_query = st.text_area("What would you like the agent to do?", value=default_prompt)
    
    if st.button("Run Analysis"):
        with st.spinner("Agent is thinking, retrieving data, and calculating..."):
            # Triggering the LangGraph cyclical reasoning loop
            response = agent.invoke({"messages": [HumanMessage(content=user_query)]})
            
            st.markdown("### 📊 Final Output")
            
            # Cleaning up the raw JSON/signature output from the LLM for a clean UI
            raw_data = response["messages"][-1].content
            if isinstance(raw_data, list):
                clean_text = raw_data[0].get('text', str(raw_data))
                st.info(clean_text)
            else:
                st.info(raw_data)
else:
    st.warning("Please enter your API Key in the sidebar to begin.")
    