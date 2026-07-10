import os
import pdfplumber
import streamlit as st
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage



#1. Page Configuration
st.set_page_config(page_title="Agentic RAG Analytics", page_icon="🤖")
st.title("📈 Agentic RAG System")
st.markdown("Upload data → AI extracts insights → Automatically calculates trends using an intelligent agent.")

st.markdown("""
###  How to use:
1. Enter your Gemini API key (left sidebar)
2. Upload a `.txt` or `.pdf` file OR use default data
3. Click **Run Analysis**
4. Get AI-powered trend insights 📊
""")

#2. Dynamic File Upload
uploaded_file = st.file_uploader("Upload a file (optional, uses default if empty)", type=["txt", "pdf"])

if uploaded_file is not None:
    file_ext = uploaded_file.name.split(".")[-1].lower()
    save_path = f"company_data.{file_ext}"
    with open(save_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    st.session_state["uploaded_file_path"] = save_path
    st.success("File uploaded successfully! Updating the agent's memory...")
    st.cache_resource.clear()

# --- 3. API Key ---
api_key = st.sidebar.text_input("Enter Google Gemini API Key:", type="password")

# --- 4. Agent Setup ---
@st.cache_resource
def initialize_agent(_api_key):
    os.environ["GOOGLE_API_KEY"] = _api_key

    file_path = st.session_state.get("uploaded_file_path", "company_data.txt")

    if file_path.endswith(".pdf"):
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        docs = [Document(page_content=text)]
    else:
        loader = TextLoader(file_path)
        docs = loader.load()

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(docs, embeddings)
    retriever = vector_db.as_retriever()

    #Tool 1: Search
    @tool
    def search_company_data(query: str) -> str:
        """Search for numerical sales data or relevant company information from the document."""
        results = retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in results])

    #Tool 2: Regression
    @tool
    def calculate_trend_regression(x_values: list[float], y_values: list[float]) -> dict:
        """
        Calculate linear regression slope and intercept.
        Returns the regression equation and trend.
        """

        n = len(x_values)

        if n == 0 or n != len(y_values):
            return {"error": "Invalid data"}

        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x*y for x, y in zip(x_values, y_values))
        sum_xx = sum(x*x for x in x_values)

        denominator = n * sum_xx - sum_x**2

        if denominator == 0:
            return {"error": "Cannot compute regression"}

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n
        equation = f"Sales = {slope:.2f} × Month + {intercept:.2f}"
        trend = (
            "Increasing"
            if slope > 0
            else "Decreasing"
            if slope < 0
            else "Stable"
        )

        return {
            "slope": round(slope, 2),
            "intercept": round(intercept, 2),
            "equation": equation,
            "trend": trend
        }

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    tools_list = [search_company_data, calculate_trend_regression]

    return create_react_agent(llm, tools_list)

#5. Run App
if api_key:
    with st.spinner("Initializing agent and building knowledge base..."):
        agent = initialize_agent(api_key)
        st.success("System is ready!")

    #prompt
    default_prompt = """
Analyze the uploaded business document.

Extract all numerical values.

Generate a professional Business Analysis Report.

Include:

1. Data Summary
Total records
Total Sales
Average Sales
Highest Sales
Lowest Sales

2. Trend Analysis
Perform Linear Regression.
Display the Regression Equation.
Display the Regression Slope.
Display the Regression Intercept.
Explain the trend in simple business language.

3. Prediction
Predict next month's sales.

4. Recommendations
- Provide 5 business recommendations.
"""

    user_query = st.text_area("What would you like the agent to do?", value=default_prompt)

    st.caption("The agent will automatically extract data and calculate trends.")

    if st.button("Run Analysis"):
        with st.spinner("Agent is thinking, retrieving data, and calculating..."):
            try:
                response = agent.invoke({"messages": [HumanMessage(content=user_query)]})

                st.markdown("### Final Output")

                raw_data = response["messages"][-1].content

                if isinstance(raw_data, list):
                    clean_text = raw_data[0].get('text', str(raw_data))
                    st.info(clean_text)
                else:
                    st.info(raw_data)

            except Exception as e:
                error_msg = str(e).lower()
                if "quota" in error_msg or "rate" in error_msg or "exhausted" in error_msg or "429" in error_msg:
                    st.error("API quota exceeded. Your Gemini API key has reached its limit. Please wait or use a different API key.")
                elif "invalid" in error_msg or "api key" in error_msg or "401" in error_msg or "403" in error_msg:
                    st.error("Invalid API key. Please check your Gemini API key and try again.")
                elif "network" in error_msg or "connection" in error_msg or "timeout" in error_msg:
                    st.error("Network error. Please check your internet connection and try again.")
                else:
                    st.error(f"Something went wrong: {e}")

else:
    st.info("Enter your Gemini API key in the sidebar to activate the AI agent.")
    st.markdown("This app uses **BYOK (Bring Your Own Key)** for security.")
