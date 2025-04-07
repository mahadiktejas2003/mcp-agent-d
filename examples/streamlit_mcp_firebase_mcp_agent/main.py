from mcp import ListToolsResult
import streamlit as st
import asyncio
from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

def format_list_tools_result(list_tools_result: ListToolsResult):
    res = ""
    for tool in list_tools_result.tools:
        res += f"- **{tool.name}**: {tool.description}\n\n"
    return res

async def main():
    await app.initialize()

    firebase_agent = Agent(
        name="firebase_agent",
        instruction="""You are an intelligent assistant designed to help users with EV charging-related queries. You have access to Firebase services, including Firestore, and can perform CRUD operations to fetch or update data. Your primary tasks include:

1. Providing information about nearby EV charging stations based on user location.
2. Recommending charging stations based on traffic and user preferences.
3. Offering insights into energy consumption and pricing.
4. Assisting users in booking and reserving charging slots.
5. Sending notifications or updates related to EV charging.

Always ensure your responses are accurate, concise, and relevant to the user's query. If you cannot find the required information, retry or switch tools until satisfactory results are obtained.""",
        server_names=["firebase-mcp"],
    )
    await firebase_agent.initialize()
    llm = await firebase_agent.attach_llm(OpenAIAugmentedLLM)

    tools = await firebase_agent.list_tools()
    tools_str = format_list_tools_result(tools)

    st.title("🔋 EV Assistance Chatbot")
    st.caption("🚀 Powered by Firebase MCP Server and mcp-agent")

    with st.expander("View Tools"):
        st.markdown(tools_str)

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Welcome! How can I assist you with EV charging today?"}
        ]

    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Type your message here..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})

        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            response = ""
            with st.spinner("Fetching data..."):
                satisfied = False
                attempts = 0

                while not satisfied and attempts < 5:  # Retry logic with a maximum of 5 attempts
                    try:
                        if "nearby charging stations" in prompt.lower():
                            response = await firebase_agent.call_tool(
                                name="firestore-query",
                                arguments={
                                    "collection": "Owner",
                                    "filters": {"field": "owner_location", "op": "near", "value": "user_location"},
                                },
                            )
                        elif "reserve a slot" in prompt.lower():
                            response = await firebase_agent.call_tool(
                                name="firestore-create",
                                arguments={
                                    "collection": "Owner/EV_Station/slot",
                                    "data": {"user_email": "example@gmail.com", "time": "10:00 AM"},
                                },
                            )
                        elif "list users" in prompt.lower():
                            response = await firebase_agent.call_tool(
                                name="firestore-query",
                                arguments={"collection": "User"},
                            )
                        elif "list owners" in prompt.lower():
                            response = await firebase_agent.call_tool(
                                name="firestore-query",
                                arguments={"collection": "Owner"},
                            )
                        else:
                            response = await llm.generate_str(
                                message=prompt, request_params=RequestParams(use_history=True)
                            )

                        if response and "no data" not in response.lower():  # Check if the response is satisfactory
                            satisfied = True
                        else:
                            attempts += 1
                            st.warning("Retrying to fetch better results...")

                    except Exception as e:
                        attempts += 1
                        st.error(f"Error occurred: {e}. Retrying...")

            if not satisfied:
                response = "I'm sorry, I couldn't fetch the desired information. Please try rephrasing your query."

            st.markdown(response)

        st.session_state["messages"].append({"role": "assistant", "content": response})

if __name__ == "__main__":
    app = MCPApp(name="firebase_ev_assistant")

    asyncio.run(main())
