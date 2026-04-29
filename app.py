import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

#1.Page Configuration
st.set_page_config(page_title="Agentic RAG Analytics", page_icon="🤖")
st.title("📈 Agentic RAG System")
st.markdown("Upload data → AI extracts insights → Automatically calculates trends using an intelligent agent.")

st.markdown("""
###  How to use:
1. Enter your Gemini API key (left sidebar)
2. Upload a `.txt` file OR use default data
3. Click **Run Analysis**
4. Get AI-powered trend insights 📊
""")

#2.Dynamic File Upload
uploaded_file = st.file_uploader("Upload a text file (optional, uses default if empty)", type=["txt"])

if uploaded_file is not None:
    with open("company_data.txt", "wb") as file:
        file.write(uploaded_file.getbuffer())
    
    st.success("File uploaded successfully! Updating the agent's memory...")
    st.cache_resource.clear()

#3.API Key
api_key = st.sidebar.text_input("Enter Google Gemini API Key:", type="password")

#4.Agent Setup
@st.cache_resource
def initialize_agent(_api_key):
    os.environ["GOOGLE_API_KEY"] = _api_key

    loader = TextLoader("company_data.txt")  
    docs = loader.load()

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(docs, embeddings)
    retriever = vector_db.as_retriever()

    #Tool 1:Search
    @tool
    def search_company_data(query: str) -> str:
        """Search for numerical sales data or relevant company information from the document."""
        results = retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in results])

    #Tool 2:Regression
    @tool
    def calculate_trend_slope(x_values: list[float], y_values: list[float]) -> float:
        """Calculate linear regression slope to determine trend."""
        n = len(x_values)
        if n == 0 or n != len(y_values): 
            return 0.0

        sum_x, sum_y = sum(x_values), sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_xx = sum(x * x for x in x_values)

        denominator = (n * sum_xx - sum_x**2)
        if denominator == 0:
            return 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return round(slope, 4)

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    tools_list = [search_company_data, calculate_trend_slope]

    return create_react_agent(llm, tools_list)

#5.Run App
if api_key:
    with st.spinner("Initializing agent and building knowledge base..."):
        agent = initialize_agent(api_key)
        st.success("System is ready!")

    #AGENTIC PROMPT 
    default_prompt = """
Analyze the provided document and identify any numerical sales data.
Use that data to determine the overall trend by calculating the linear regression slope.
Explain whether the trend is increasing, decreasing, or stable.
"""

    user_query = st.text_area("What would you like the agent to do?", value=default_prompt)

    st.caption(" The agent will automatically extract data and calculate trends.")

    if st.button("Run Analysis"):
        with st.spinner("Agent is thinking, retrieving data, and calculating..."):
            response = agent.invoke({"messages": [HumanMessage(content=user_query)]})

            st.markdown("### 📊 Final Output")

            raw_data = response["messages"][-1].content

            if isinstance(raw_data, list):
                clean_text = raw_data[0].get('text', str(raw_data))
                st.info(clean_text)
            else:
                st.info(raw_data)

else:
    st.info(" Enter your Gemini API key in the sidebar to activate the AI agent.")
    st.markdown(" This app uses **BYOK (Bring Your Own Key)** for security.")
