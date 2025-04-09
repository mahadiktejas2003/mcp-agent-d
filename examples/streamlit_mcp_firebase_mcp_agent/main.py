from mcp import ListToolsResult 
import streamlit as st
import asyncio
import json
import os
from pathlib import Path
from firebase_admin import credentials, initialize_app
from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

def init_firebase():
    try:
        # Load Firebase config directly from Streamlit secrets
        firebase_config = json.loads(st.secrets["FIREBASE_SERVICE_ACCOUNT_JSON"])
        
        # Ensure newlines in private key are properly formatted
        firebase_config["private_key"] = firebase_config["private_key"].replace("\\n", "\n")
        
        # Initialize Firebase with in-memory credentials
        cred = credentials.Certificate(firebase_config)
        initialize_app(cred)
        st.success("🔥 Firebase initialized successfully!")
    except Exception as e:
        st.error(f"🚨 Firebase initialization failed: {str(e)}")
        raise

def format_list_tools_result(list_tools_result: ListToolsResult):
    return "\n".join(f"- **{tool.name}**: {tool.description}" for tool in list_tools_result.tools)

async def query_firestore(agent, collection: str, filters: dict = None):
    try:
        return await agent.call_tool(
            name="firestore-query",
            arguments={
                "collection": collection,
                "filters": filters if filters else {}
            }
        )
    except Exception as e:
        st.error(f"🔍 Firestore query failed: {str(e)}")
        return None

async def main():
    # Set OpenAI API key from secrets
    os.environ["OPENAI_API_KEY"] = st.secrets.openai.api_key

    # Initialize core services
    with st.spinner("⚡ Initializing services..."):
        init_firebase()
        app = MCPApp(name="firebase_ev_assistant")
        await app.initialize()

    # Initialize Agent
    firebase_agent = Agent(
        name="firebase_agent",
        instruction="""🔋 You are an EV charging expert with Firebase access. Your capabilities include:
1. Finding nearby charging stations
2. Managing reservations & user accounts
3. Providing energy consumption insights
4. Real-time status updates
5. Smart recommendations based on traffic/usage""",
        server_names=["firebase-mcp"],
    )
    await firebase_agent.initialize()
    llm = await firebase_agent.attach_llm(OpenAIAugmentedLLM)

    # UI Setup
    st.title("🔋 EV Charging Assistant")
    st.caption("🚀 Powered by Firebase MCP & AI")
    
    with st.expander("🛠️ Available Tools"):
        tools = await firebase_agent.list_tools()
        st.markdown(format_list_tools_result(tools))

    # Chat session
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant", 
            "content": "Welcome! How can I assist with EV charging today?"
        }]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("🔍 Type your question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            response = ""
            with st.spinner("🧠 Processing..."):
                satisfied = False
                attempts = 0
                
                while not satisfied and attempts < 3:
                    try:
                        if "nearby" in prompt.lower():
                            response = await query_firestore(
                                firebase_agent, 
                                "Owner",
                                {"field": "owner_location", "op": "near", "value": "user_location"}
                            )
                        elif "reserve" in prompt.lower():
                            response = await firebase_agent.call_tool(
                                name="firestore-create",
                                arguments={
                                    "collection": "Owner/EV_Station/slot",
                                    "data": {"user_email": "user@example.com", "time": "10:00 AM"},
                                },
                            )
                        elif "list users" in prompt.lower():
                            response = await query_firestore(firebase_agent, "User")
                        elif "list owners" in prompt.lower():
                            response = await query_firestore(firebase_agent, "Owner")
                        else:
                            response = await llm.generate_str(
                                message=prompt, 
                                request_params=RequestParams(use_history=True)
                            )

                        if response and "no data" not in str(response).lower():
                            satisfied = True
                        else:
                            attempts += 1
                            st.warning(f"🔁 Retrying... (Attempt {attempts}/3)")

                    except Exception as e:
                        attempts += 1
                        st.error(f"⚠️ Error: {str(e)}")

                if not satisfied:
                    response = "❌ Sorry, I couldn't find that information. Please try rephrasing."

                st.markdown(response)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response
                })

if __name__ == "__main__":
    asyncio.run(main())
