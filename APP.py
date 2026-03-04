import streamlit as st
from activity_planner.Agents import get_agent
from activity_planner.QA_Logger import get_logger
import json

st.set_page_config(page_title="Travel Assistant", page_icon="✈️")

# ---------- LOAD AGENT ----------
@st.cache_resource
def load_agent():
    return get_agent()

agent = load_agent()

# ---------- SESSION ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("✈️ Travel Assistant")

# ---------- CHAT HISTORY ----------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

st.sidebar.title("⚙️ Settings")

threads = st.sidebar.text_input("Theards")

# ---------- UI HELPERS ----------
def _list_section(title, items, icon):
    if items:
        with st.expander(f"{icon} {title}", expanded=False):
            for i in items:
                st.markdown(f"- {i}")


# ---------- RENDER ----------
def render_messages(answer: dict):

    if not answer:
        return

    if answer.get("type") == "non_trip":
        with st.chat_message("assistant"):
            st.markdown(answer.get("message", ""))

    elif answer.get("type") == "trip_plan":

        with st.chat_message("assistant"):

            destination = answer.get("destination", "Trip")
            st.subheader(f"🌍 {destination}")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("📅 Duration", answer.get("duration_days", "unknown"))

            with col2:
                st.metric("💰 Budget", answer.get("budget_estimate", "unknown"))

            st.divider()

            _list_section("Activities", answer.get("activities"), "🎯")
            _list_section("Accommodation", answer.get("accommodations"), "🏨")
            _list_section("Transportation", answer.get("transportation"), "🚗")
            _list_section("Itinerary", answer.get("itinerary"), "🗺️")
            _list_section("Culture", answer.get("culture"), "🎎")
            _list_section("Safety", answer.get("safety"), "🛟")
            _list_section("Health", answer.get("health"), "💊")

    else:
        with st.chat_message("assistant"):
            st.json(answer)


# ---------- INPUT ----------
prompt = st.chat_input("Ask about your trip...")

if prompt:
    # show user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # run agent with live status steps
    with st.status("Thinking...", expanded=True) as status:
        st.write("🔍 Searching knowledge base...")
        thread = {"configurable": {"thread_id": f"{threads}"}}
        result = agent.run(prompt, thread=thread)
        answer = result["messages"][-1].content
        # capture any retrieval citation that the agent stored
        citation = result.get("citation")
        if citation:
            # try to format JSON list of citations for display
            try:
                parsed = json.loads(citation)
            except Exception:
                parsed = citation
            st.sidebar.markdown("**🔗 Retrieval citations**")
            st.sidebar.write(parsed)
        # Show how many tool calls were made
        tool_calls = sum(
            1 for m in result["messages"]
            if hasattr(m, "type") and m.type == "tool"
        )
        if tool_calls:
            st.write(f"🛠️ Used {tool_calls} tool(s) to fetch live data")
        st.write("✅ Response ready!")
        status.update(label="Done!", state="complete", expanded=False)

    # Log the Q&A with retrieved context
    try:
        logger = get_logger()
        
        # Extract retrieved context from result
        retrieved_context = []
        if "contents" in result:
            contents = result.get("contents")
            if isinstance(contents, str):
                try:
                    retrieved_context = json.loads(contents)
                except (json.JSONDecodeError, ValueError):
                    retrieved_context = [{"context": contents}]
            elif isinstance(contents, list):
                retrieved_context = contents
        
        # Log the Q&A
        logger.log_qa(
            question=prompt,
            answer=answer,
            retrieved_context=retrieved_context
        )
        print(f"✅ Logged Q&A to: {logger.get_log_file_path()}")
    except Exception as e:
        print(f"⚠️ Error logging Q&A: {e}")
        import traceback
        traceback.print_exc()

    # parse
    try:
        data = json.loads(answer)
        render_messages(data)

        # display citation also in the chat stream if available
        if citation:
            with st.chat_message("assistant"):
                st.markdown("**Sources used for retrieval:**")
                # parsed earlier when adding sidebar
                try:
                    cit_list = json.loads(citation)
                except Exception:
                    cit_list = citation
                st.write(cit_list)

        readable = (
            data.get("destination")
            or data.get("message")
            or "Trip plan generated"
        )

    except Exception:
        with st.chat_message("assistant"):
            st.markdown(answer)
        readable = answer

    # store assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": readable
    })
