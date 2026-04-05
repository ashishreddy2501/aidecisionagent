import streamlit as st
import os
from typing import TypedDict, List

from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI

# ==============================
# SET API KEY
# ==============================
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# ==============================
# STATE & AGENTS
# ==============================
class AgentState(TypedDict):
    query: str
    plan: str
    research: str
    analysis: str
    critique: str
    final: str
    history: List[str]

# Optimization: Cache the LLM instance
@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

llm = get_llm()

# ==============================
# AGENTS
# ==============================

@st.cache_resource
def planner_agent(state):
    prompt = f"""
    Break down the user query into a clear step-by-step plan.

    Query: {state['query']}
    """
    response = llm.invoke(prompt)
    return {**state, "plan": response.content}

@st.cache_resource
def researcher_agent(state):
    prompt = f"""
    Conduct research based on the plan.

    Plan:
    {state['plan']}

    Provide useful factual insights.
    """
    response = llm.invoke(prompt)
    return {**state, "research": response.content}

@st.cache_resource
def analyst_agent(state):
    prompt = f"""
    Analyze the research and extract insights.

    Research:
    {state['research']}

    Provide:
    - Key insights
    - Pros
    - Cons
    """
    response = llm.invoke(prompt)
    return {**state, "analysis": response.content}

@st.cache_resource
def critic_agent(state):
    prompt = f"""
    Critically evaluate the analysis.

    Analysis:
    {state['analysis']}
    """
    response = llm.invoke(prompt)
    return {**state, "critique": response.content}

@st.cache_resource
def decision_agent(state):
    prompt = f"""
    Based on the analysis, provide a final decision.

    Include:
    - Final recommendation
    - Reasoning
    - Confidence score (1-10)

    Analysis:
    {state['analysis']}
    """
    response = llm.invoke(prompt)
    return {**state, "final": response.content}

# ==============================
# OPTIMIZED LANGGRAPH WORKFLOW
# ==============================

@st.cache_resource
def get_graph():
    builder = StateGraph(AgentState)

    builder.add_node("planner", planner_agent)
    builder.add_node("researcher", researcher_agent)
    builder.add_node("analyst", analyst_agent)
    builder.add_node("critic", critic_agent)
    builder.add_node("decision", decision_agent)

    builder.set_entry_point("planner")
    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", "analyst")
    builder.add_edge("analyst", "critic")

    #def should_continue(state):
    #    return "analyst" if "IMPROVE" in state["critique"] else "decision"

    #builder.add_conditional_edges("critic", should_continue)
    builder.add_edge("critic", "decision")
    builder.add_edge("decision", "__end__")

    return builder.compile()

# Get the cached version
graph = get_graph()

# ==============================
# RUN FUNCTION
# ==============================

def run_agent(query):
    # Initial state
    initial_state = {
        "query": query,
        "plan": "",
        "research": "",
        "analysis": "",
        "critique": "",
        "final": "",
        "history": []
    }
    # Invoke the cached graph
    return graph.invoke(initial_state)

# ==============================
# STREAMLIT UI
# ==============================
st.set_page_config(page_title="Agentic AI Assistant", layout="wide")
st.title("Agentic AI Decision Assistant")
st.markdown("""
A dynamic AI agent that performs decision-making and task routing using LangGraph based workflows powered by OpenAI LLM.
""")
st.text("Created by S Ashish Reddy")

#st.caption("Created by S Ashish Reddy")

query = st.text_input("Enter your question:", placeholder="Should I invest in gold right now?")

if st.button("Run Agent"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        # Use a container so the UI feels responsive
        with st.status("Agent is thinking...", expanded=True) as status:
            st.write("Planning...")
            result = run_agent(query)
            status.update(label="Analysis Complete!", state="complete", expanded=False)

        # Display results in Tabs to save space
        tab1, tab2, tab3, tab4 = st.tabs(["Final Decision", "Analysis", "Research", "Plan"])
        
        with tab1:
            st.markdown(result["final"])
        with tab2:
            st.markdown(f"**Critique:** {result['critique']}")
            st.markdown(result["analysis"])
        with tab3:
            st.markdown(result["research"])
        with tab4:
            st.markdown(result["plan"])
