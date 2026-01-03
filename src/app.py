import os
import logging
import logging.config
from typing import Any
from uuid import uuid4, UUID
import json
import sys

import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.types import RunnableConfig
from pydantic import BaseModel
from pathlib import Path

load_dotenv()

# Check Gradio version and provide guidance
print(f"Gradio version: {gr.__version__}")

# Parse version to check compatibility
try:
    version_parts = gr.__version__.split('.')
    major_version = int(version_parts[0])
    minor_version = int(version_parts[1]) if len(version_parts) > 1 else 0
    
    if major_version < 4:
        print("⚠️  WARNING: You're using an older version of Gradio.")
        print("   Some features may be limited. Consider upgrading:")
        print("   pip install --upgrade gradio>=4.0.0")
    elif major_version >= 4:
        print("✅ Gradio version is compatible with all features.")
        
except (ValueError, IndexError):
    print("Could not parse Gradio version.")
    
print()  # Add spacing

# There are tools set here dependent on environment variables
from graph import graph, weak_model, search_enabled # noqa

FOLLOWUP_QUESTION_NUMBER = 3
TRIM_MESSAGE_LENGTH = 16  # Includes tool messages
USER_INPUT_MAX_LENGTH = 10000  # Characters

# We need the same secret for data persistance
# If you store sensitive data, you should store your secret in .env
BROWSER_STORAGE_SECRET = "itsnosecret"

try:
    with open('logging-config.json', 'r') as fh:
        config = json.load(fh)
    logging.config.dictConfig(config)
except FileNotFoundError:
    # Fallback logging configuration
    logging.basicConfig(level=logging.INFO)
    
logger = logging.getLogger(__name__)

def load_initial_greeting(filepath="greeting_prompt.txt") -> str:
    """
    Loads the initial greeting message from a specified text file.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        logger.warning(f"Warning: Prompt file '{filepath}' not found.")
        return "Welcome to DIYO! I'm here to help you create amazing DIY projects. What would you like to build today?"

async def chat_fn(user_input: str, history: list, input_graph_state: dict, uuid: UUID, prompt: str, search_enabled: bool, download_website_text_enabled: bool):
    """
    Chat function that works with tuples format for maximum compatibility
    
    Args:
        user_input (str): The user's input message
        history (list): The history of the conversation in tuples format [(user_msg, bot_msg), ...]
        input_graph_state (dict): The current state of the graph
        uuid (UUID): The unique identifier for the current conversation
        prompt (str): The system prompt
    Yields:
        list: Updated history in tuples format
        dict: The final state of the graph
        bool: Whether to trigger follow up questions
    """
    try:
        logger.info(f"Processing user input: {user_input[:100]}...")
        logger.info(f"History format: {type(history)}, length: {len(history) if history else 0}")
        
        # Initialize input_graph_state if None
        if input_graph_state is None:
            input_graph_state = {}
            
        input_graph_state["tools_enabled"] = {
            "download_website_text": download_website_text_enabled,
            "tavily_search_results_json": search_enabled,
        }
        if prompt:
            input_graph_state["prompt"] = prompt

        # Convert tuples history to internal messages format for graph processing
        if not isinstance(history, list):
            history = []

        # Convert history to messages format for graph processing
        internal_messages = convert_from_tuples_format(history)
        logger.info(f"Converted {len(history)} tuples to {len(internal_messages)} internal messages")

        if input_graph_state.get("awaiting_human_input"):
            internal_messages.append(
                ToolMessage(
                    tool_call_id=input_graph_state.pop("human_assistance_tool_id"),
                    content=user_input
                )
            )
            input_graph_state["awaiting_human_input"] = False
        else:
            # New user message
            internal_messages.append(
                HumanMessage(user_input[:USER_INPUT_MAX_LENGTH])
            )

        # Store internal messages in graph state
        input_graph_state["messages"] = internal_messages[-TRIM_MESSAGE_LENGTH:]

        config = RunnableConfig(
            recursion_limit=20,
            run_name="user_chat",
            configurable={"thread_id": str(uuid)}
        )

        output: str = ""
        final_state: dict = input_graph_state.copy()  # Initialize with current state
        waiting_output_seq: list[str] = []
        
        # Add user message to history immediately
        updated_history = history + [(user_input, "")]
        logger.info(f"Updated history length: {len(updated_history)}")

        async for stream_mode, chunk in graph.astream(
                    input_graph_state,
                    config=config,
                    stream_mode=["values", "messages"],
                ):
            if stream_mode == "values":
                final_state = chunk
                if chunk.get("messages") and len(chunk["messages"]) > 0:
                    last_message = chunk["messages"][-1]
                    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                        for msg_tool_call in last_message.tool_calls:
                            tool_name: str = msg_tool_call['name']

                            if tool_name == "tavily_search_results_json":
                                query = msg_tool_call['args']['query']
                                waiting_output_seq.append(f"🔍 Searching for '{query}'...")
                                # Update the last tuple with current status
                                if updated_history:
                                    updated_history[-1] = (user_input, "\n".join(waiting_output_seq))
                                yield updated_history, final_state, False

                            elif tool_name == "download_website_text":
                                url = msg_tool_call['args']['url']
                                waiting_output_seq.append(f"📥 Downloading text from '{url}'...")
                                if updated_history:
                                    updated_history[-1] = (user_input, "\n".join(waiting_output_seq))
                                yield updated_history, final_state, False

                            elif tool_name == "human_assistance":
                                query = msg_tool_call["args"]["query"]
                                waiting_output_seq.append(f"🤖: {query}")

                                # Save state to resume after user provides input
                                final_state["awaiting_human_input"] = True
                                final_state["human_assistance_tool_id"] = msg_tool_call["id"]

                                # Update history and indicate that human input is needed
                                if updated_history:
                                    updated_history[-1] = (user_input, "\n".join(waiting_output_seq))
                                yield updated_history, final_state, True
                                return  # Pause execution, resume in next call

                            else:
                                waiting_output_seq.append(f"🔧 Running {tool_name}...")
                                if updated_history:
                                    updated_history[-1] = (user_input, "\n".join(waiting_output_seq))
                                yield updated_history, final_state, False

            elif stream_mode == "messages":
                msg, metadata = chunk
                # Check for the correct node name from your graph
                node_name = metadata.get('langgraph_node', '')
                if node_name in ["brainstorming_node", "prompt_planning_node", "generate_3d_node", "assistant_node"]:
                    current_chunk_text = ""
                    if isinstance(msg.content, str):
                        current_chunk_text = msg.content
                    elif isinstance(msg.content, list):
                        for block in msg.content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                current_chunk_text += block.get("text", "")
                            elif isinstance(block, str):
                                current_chunk_text += block

                    if current_chunk_text:
                        output += current_chunk_text
                        # Update the last tuple with accumulated output
                        if updated_history:
                            updated_history[-1] = (user_input, output)
                        yield updated_history, final_state, False

        # Final yield with complete response
        if updated_history:
            updated_history[-1] = (user_input, output.strip() if output else "I'm here to help with your DIY projects!")
        logger.info(f"Final response: {output[:100]}...")
        yield updated_history, final_state, True
        
    except Exception as e:
        logger.exception("Exception occurred in chat_fn")
        error_message = "There was an error processing your request. Please try again."
        if not isinstance(history, list):
            history = []
        error_history = history + [(user_input, error_message)]
        # Return safe values instead of gr.skip()
        yield error_history, input_graph_state or {}, False


def convert_to_tuples_format(messages_list):
    """Convert messages format to tuples format for older Gradio versions"""
    if not isinstance(messages_list, list):
        logger.warning(f"Expected list for messages conversion, got {type(messages_list)}")
        return []
        
    tuples = []
    user_msg = None
    
    for msg in messages_list:
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "user":
                user_msg = content
            elif role == "assistant":
                if user_msg is not None:
                    tuples.append((user_msg, content))
                    user_msg = None
                else:
                    # Assistant message without user message, add empty user message
                    tuples.append((None, content))
        elif isinstance(msg, tuple) and len(msg) == 2:
            # Already in tuple format
            tuples.append(msg)
    
    # If there's a hanging user message, add it with empty assistant response
    if user_msg is not None:
        tuples.append((user_msg, ""))
    
    logger.info(f"Converted {len(messages_list)} messages to {len(tuples)} tuples")
    return tuples


def convert_from_tuples_format(tuples_list):
    """Convert tuples format to messages format"""
    if not isinstance(tuples_list, list):
        logger.warning(f"Expected list for tuples conversion, got {type(tuples_list)}")
        return []
        
    messages = []
    for item in tuples_list:
        if isinstance(item, tuple) and len(item) == 2:
            user_msg, assistant_msg = item
            if user_msg and user_msg.strip():
                messages.append({"role": "user", "content": user_msg})
            if assistant_msg and assistant_msg.strip():
                messages.append({"role": "assistant", "content": assistant_msg})
        elif isinstance(item, dict):
            # Already in messages format
            messages.append(item)
    
    logger.info(f"Converted {len(tuples_list)} tuples to {len(messages)} messages")
    return messages

def clear():
    """Clear the current conversation state"""
    return dict(), uuid4()

class FollowupQuestions(BaseModel):
    """Model for langchain to use for structured output for followup questions"""
    questions: list[str]

async def populate_followup_questions(end_of_chat_response: bool, history: list, uuid: UUID):
    """
    Generate followup questions based on chat history in tuples format
    
    Args:
        end_of_chat_response (bool): Whether the chat response has ended
        history (list): Chat history in tuples format [(user, bot), ...]
        uuid (UUID): Session UUID
    """
    if not end_of_chat_response or not history or len(history) == 0:
        return *[gr.skip() for _ in range(FOLLOWUP_QUESTION_NUMBER)], False
    
    # Check if the last tuple has a bot response
    if not history[-1][1]:  # No bot response in the last tuple
        return *[gr.skip() for _ in range(FOLLOWUP_QUESTION_NUMBER)], False
        
    try:
        # Convert tuples format to messages format for LLM processing
        messages = convert_from_tuples_format(history)
        
        if not messages:
            return *[gr.skip() for _ in range(FOLLOWUP_QUESTION_NUMBER)], False
        
        config = RunnableConfig(
            run_name="populate_followup_questions",
            configurable={"thread_id": str(uuid)}
        )
        weak_model_with_config = weak_model.with_config(config)
        follow_up_questions = await weak_model_with_config.with_structured_output(FollowupQuestions).ainvoke([
            ("system", f"suggest {FOLLOWUP_QUESTION_NUMBER} followup questions for the user to ask the assistant. Refrain from asking personal questions."),
            *messages,
        ])
        
        if len(follow_up_questions.questions) != FOLLOWUP_QUESTION_NUMBER:
            logger.warning("Invalid number of followup questions generated")
            return *[gr.Button(visible=False) for _ in range(FOLLOWUP_QUESTION_NUMBER)], False
            
        buttons = []
        for i in range(FOLLOWUP_QUESTION_NUMBER):
            buttons.append(
                gr.Button(follow_up_questions.questions[i], visible=True, elem_classes="chat-tab"),
            )
        return *buttons, False
        
    except Exception as e:
        logger.error(f"Error generating followup questions: {e}")
        return *[gr.Button(visible=False) for _ in range(FOLLOWUP_QUESTION_NUMBER)], False

async def summarize_chat(end_of_chat_response: bool, history: list, sidebar_summaries: dict, uuid: UUID):
    """Summarize chat for tab names using tuples format"""
    should_return = (
        not end_of_chat_response or
        not history or
        len(history) == 0 or
        not history[-1][1] or  # No bot response in last tuple
        isinstance(sidebar_summaries, type(lambda x: x)) or
        uuid in sidebar_summaries
    )
    if should_return:
        return gr.skip(), gr.skip()

    # Convert tuples format to messages format for processing
    messages = convert_from_tuples_format(history)

    # Filter valid messages
    filtered_messages = []
    for msg in messages:
        if isinstance(msg, dict) and msg.get("content") and msg["content"].strip():
            filtered_messages.append(msg)
    
    # If we don't have any valid messages after filtering, provide a default summary
    if not filtered_messages:
        if uuid not in sidebar_summaries:
            sidebar_summaries[uuid] = "New Chat"
        return sidebar_summaries, False

    try:
        config = RunnableConfig(
            run_name="summarize_chat",
            configurable={"thread_id": str(uuid)}
        )
        weak_model_with_config = weak_model.with_config(config)
        summary_response = await weak_model_with_config.ainvoke([
            ("system", "summarize this chat in 7 tokens or less. Refrain from using periods"),
            *filtered_messages,
        ])
        
        if uuid not in sidebar_summaries:
            sidebar_summaries[uuid] = summary_response.content[:50]  # Limit length
            
    except Exception as e:
        logger.error(f"Error summarizing chat: {e}")
        if uuid not in sidebar_summaries:
            sidebar_summaries[uuid] = "Chat Session"
    
    return sidebar_summaries, False

async def new_tab(uuid, gradio_graph, history, tabs, prompt, sidebar_summaries):
    """Create a new chat tab"""
    new_uuid = uuid4()
    new_graph = {}
    
    # Save current tab if it has content
    if history and len(history) > 0:
        if uuid not in sidebar_summaries:
            sidebar_summaries, _ = await summarize_chat(True, history, sidebar_summaries, uuid)
        tabs[uuid] = {
            "graph": gradio_graph,
            "messages": history,  # Store history as-is (tuples format)
            "prompt": prompt,
        }
    
    # Clear suggestion buttons
    suggestion_buttons = [gr.Button(visible=False) for _ in range(FOLLOWUP_QUESTION_NUMBER)]
    
    # Load initial greeting for new chat in tuples format
    greeting_text = load_initial_greeting()
    new_chat_history = [(None, greeting_text)]
    
    new_prompt = prompt if prompt else "You are a helpful DIY assistant."
    
    return new_uuid, new_graph, new_chat_history, tabs, new_prompt, sidebar_summaries, *suggestion_buttons

def switch_tab(selected_uuid, tabs, gradio_graph, uuid, history, prompt):
    """Switch to a different chat tab"""
    try:
        # Save current state if there are messages
        if history and len(history) > 0:
            tabs[uuid] = {
                "graph": gradio_graph if gradio_graph else {},
                "messages": history,  # Store history as-is (tuples format)
                "prompt": prompt
            }

        if selected_uuid not in tabs:
            logger.error(f"Could not find the selected tab in tabs storage: {selected_uuid}")
            return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), *[gr.Button(visible=False) for _ in range(FOLLOWUP_QUESTION_NUMBER)]
            
        selected_tab_state = tabs[selected_uuid]
        selected_graph = selected_tab_state.get("graph", {})
        selected_history = selected_tab_state.get("messages", [])  # This should be tuples format
        selected_prompt = selected_tab_state.get("prompt", "You are a helpful DIY assistant.")
        
        suggestion_buttons = [gr.Button(visible=False) for _ in range(FOLLOWUP_QUESTION_NUMBER)]
        
        return selected_graph, selected_uuid, selected_history, tabs, selected_prompt, *suggestion_buttons
        
    except Exception as e:
        logger.error(f"Error switching tabs: {e}")
        return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), *[gr.Button(visible=False) for _ in range(FOLLOWUP_QUESTION_NUMBER)]

def delete_tab(current_chat_uuid, selected_uuid, sidebar_summaries, tabs):
    """Delete a chat tab"""
    output_history = gr.skip()
    
    # If deleting the current tab, clear the chatbot
    if current_chat_uuid == selected_uuid:
        output_history = []  # Empty tuples list
        
    # Remove from storage
    if selected_uuid in tabs:
        del tabs[selected_uuid]
    if selected_uuid in sidebar_summaries:
        del sidebar_summaries[selected_uuid]
        
    return sidebar_summaries, tabs, output_history

def submit_edit_tab(selected_uuid, sidebar_summaries, text):
    """Submit edited tab name"""
    if text.strip():
        sidebar_summaries[selected_uuid] = text.strip()[:50]  # Limit length
    return sidebar_summaries, ""

def load_mesh(mesh_file_name):
    """Load a 3D mesh file"""
    return mesh_file_name

def get_sorted_3d_model_examples():
    """Get sorted list of 3D model examples"""
    examples_dir = Path("./generated_3d_models")
    
    # Create directory if it doesn't exist
    examples_dir.mkdir(exist_ok=True)
    
    if not examples_dir.exists():
        return []

    # Get all 3D model files with desired extensions
    model_files = [
        file for file in examples_dir.glob("*")
        if file.suffix.lower() in {".obj", ".glb", ".gltf"}
    ]

    # Sort files by creation time (latest first)
    try:
        sorted_files = sorted(
            model_files,
            key=lambda x: x.stat().st_ctime,
            reverse=True
        )
    except (OSError, AttributeError):
        # Fallback to name sorting if stat fails
        sorted_files = sorted(model_files, key=lambda x: x.name, reverse=True)

    # Convert to format [[path1], [path2], ...]
    return [[str(file)] for file in sorted_files]

CSS = """
footer {visibility: hidden}
.followup-question-button {font-size: 12px }
.chat-tab {
    font-size: 12px;
    padding-inline: 0;
}
.chat-tab.active {
    background-color: #654343;
}
#new-chat-button { background-color: #0f0f11; color: white; }

.tab-button-control {
    min-width: 0;
    padding-left: 0;
    padding-right: 0;
}

.sidebar-collapsed {
    display: none !important;
}

.sidebar-replacement {
    background-color: #f8f9fa;
    border-left: 1px solid #dee2e6;
    padding: 10px;
    min-height: 400px;
}

.wrap.sidebar-parent {
    min-height: 2400px !important;
    height: 2400px !important;
}

#main-app {
    height: 4600px;
    overflow-y: auto;
    padding-top: 20px;
}
"""

TRIGGER_CHATINTERFACE_BUTTON = """
function triggerChatButtonClick() {
  const chatTextbox = document.getElementById("chat-textbox");
  if (!chatTextbox) {
    console.error("Error: Could not find element with id 'chat-textbox'");
    return;
  }
  const button = chatTextbox.querySelector("button");
  if (!button) {
    console.error("Error: No button found inside the chat-textbox element");
    return;
  }
  button.click();
}"""

if __name__ == "__main__":
    logger.info("Starting the DIYO interface")
    
    # Check if BrowserState is available
    has_browser_state = hasattr(gr, 'BrowserState')
    logger.info(f"BrowserState available: {has_browser_state}")
    
    if not has_browser_state:
        print("📝 Note: Using session-only state (data won't persist after refresh)")
        print("   For data persistence, upgrade to Gradio 4.0+")
        logger.warning("BrowserState not available in this Gradio version. Using regular State instead.")
        logger.warning("To use BrowserState, upgrade Gradio: pip install gradio>=4.0.0")
    else:
        print("💾 Using persistent browser state (data persists after refresh)")
    
    # Log available Gradio components for debugging
    available_components = []
    for attr_name in dir(gr):
        if attr_name[0].isupper() and not attr_name.startswith('_'):
            available_components.append(attr_name)
    
    logger.info(f"Available Gradio components: {len(available_components)} components detected")
    key_components = ['ChatInterface', 'Sidebar', 'BrowserState', 'MultimodalTextbox']
    for component in key_components:
        status = "✅" if hasattr(gr, component) else "❌"
        logger.info(f"  {status} {component}")
    
    print()  # Add spacing
    
    with gr.Blocks(title="DIYO - DIY Assistant", fill_height=True, css=CSS, elem_id="main-app") as demo:
        # State management - Use BrowserState if available, otherwise regular State
        is_new_user_for_greeting = gr.State(True)
        
        if has_browser_state:
            current_prompt_state = gr.BrowserState(
                value="You are a helpful DIY assistant.",
                storage_key="current_prompt_state",
                secret=BROWSER_STORAGE_SECRET,
            )
            current_uuid_state = gr.BrowserState(
                value=uuid4(),  # Call the function to get an actual UUID
                storage_key="current_uuid_state",
                secret=BROWSER_STORAGE_SECRET,
            )
            current_langgraph_state = gr.BrowserState(
                value={},  # Empty dict instead of dict type
                storage_key="current_langgraph_state",
                secret=BROWSER_STORAGE_SECRET,
            )
            sidebar_names_state = gr.BrowserState(
                value={},  # Empty dict instead of dict type
                storage_key="sidebar_names_state",
                secret=BROWSER_STORAGE_SECRET,
            )
            offloaded_tabs_data_storage = gr.BrowserState(
                value={},  # Empty dict instead of dict type
                storage_key="offloaded_tabs_data_storage",
                secret=BROWSER_STORAGE_SECRET,
            )
            chatbot_message_storage = gr.BrowserState(
                value=[],  # Empty list instead of list type
                storage_key="chatbot_message_storage",
                secret=BROWSER_STORAGE_SECRET,
            )
        else:
            # Fallback to regular State
            current_prompt_state = gr.State("You are a helpful DIY assistant.")
            current_uuid_state = gr.State(uuid4())
            current_langgraph_state = gr.State({})
            sidebar_names_state = gr.State({})
            offloaded_tabs_data_storage = gr.State({})
            chatbot_message_storage = gr.State([])
        
        end_of_assistant_response_state = gr.State(False)
        
        # Header
        with gr.Row(elem_classes="header-margin"):
            gr.Markdown("""
            <div style="display: flex; align-items: center; justify-content: center; text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 20px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
                <h1>🔧 DIYO - Your DIY Assistant 🛠️</h1>
            </div>
            """)

        # System prompt input
        with gr.Row():
            prompt_textbox = gr.Textbox(
                label="System Prompt",
                value="You are a helpful DIY assistant.",
                show_label=True,
                interactive=True,
                placeholder="Enter custom system prompt..."
            )
        
        # Tool settings
        with gr.Row():
            checkbox_search_enabled = gr.Checkbox(
                value=True,
                label="Enable web search",
                show_label=True,
                visible=search_enabled,
                scale=1,
            )
            checkbox_download_website_text = gr.Checkbox(
                value=True,
                show_label=True,
                label="Enable downloading text from URLs",
                scale=1,
            )
            
        # 3D Model display and controls
        with gr.Row():
            with gr.Column(scale=2):
                model_3d_output = gr.Model3D(
                    clear_color=[0.0, 0.0, 0.0, 0.0],
                    label="3D Model Viewer",
                    height=400
                )
            with gr.Column(scale=1):
                model_3d_upload_button = gr.UploadButton(
                    "📁 Upload 3D Model (.obj, .glb, .gltf)",
                    file_types=[".obj", ".glb", ".gltf"],
                )
                model_3d_upload_button.upload(
                    fn=load_mesh,
                    inputs=model_3d_upload_button,
                    outputs=model_3d_output
                )
                
                # Examples with error handling and version compatibility
                try:
                    examples_list = get_sorted_3d_model_examples()
                    if examples_list:
                        examples_kwargs = {
                            "label": "Example 3D Models",
                            "examples": examples_list,
                            "inputs": model_3d_upload_button,
                            "outputs": model_3d_output,
                            "fn": load_mesh,
                        }
                        
                        # Check if cache_examples parameter is supported
                        try:
                            init_params = gr.Examples.__init__.__code__.co_varnames
                            if 'cache_examples' in init_params:
                                examples_kwargs["cache_examples"] = False
                        except Exception:
                            # Parameter not supported, skip it
                            pass
                            
                        gr.Examples(**examples_kwargs)
                except Exception as e:
                    logger.error(f"Error setting up 3D model examples: {e}")
        
        # Chat interface setup - with compatibility checks
        with gr.Row():
            multimodal = False
            
            # Check if MultimodalTextbox is available
            if hasattr(gr, 'MultimodalTextbox') and multimodal:
                textbox_component = gr.MultimodalTextbox
            else:
                textbox_component = gr.Textbox
                multimodal = False  # Force to False if not available
            
            textbox_kwargs = {
                "show_label": False,
                "label": "Message",
                "placeholder": "Type a message...",
                "scale": 1,
                "elem_id": "chat-textbox",
                "lines": 1,
            }
            
            # Check if newer textbox parameters are supported
            try:
                init_params = textbox_component.__init__.__code__.co_varnames
                
                if 'autofocus' in init_params:
                    textbox_kwargs["autofocus"] = True
                if 'submit_btn' in init_params:
                    textbox_kwargs["submit_btn"] = True
                if 'stop_btn' in init_params:
                    textbox_kwargs["stop_btn"] = True
            except Exception as e:
                logger.warning(f"Error checking textbox parameters: {e}")
                # Keep minimal parameters as fallback
            
            textbox = textbox_component(**textbox_kwargs)
            
            # Check if newer Chatbot parameters are supported
            chatbot_kwargs = {
                "height": 400,
                "elem_classes": "main-chatbox"
            }
            
            # Add parameters that might not be available in older versions
            try:
                # Check parameter availability without creating test instance
                init_params = gr.Chatbot.__init__.__code__.co_varnames
                
                # For older Gradio versions, don't try to set type parameter
                # Let it default to 'tuples' format to avoid compatibility issues
                if 'type' in init_params:
                    # Try to set type, but if it fails, let it default
                    try:
                        chatbot_kwargs["type"] = "tuples"  # Use tuples for maximum compatibility
                        logger.info("Using 'tuples' type for chatbot (compatibility mode)")
                    except:
                        logger.warning("Could not set chatbot type, using default")
                else:
                    logger.info("Chatbot 'type' parameter not supported, using default 'tuples' format")
                
                # Check if 'show_copy_button' parameter is supported  
                if 'show_copy_button' in init_params:
                    chatbot_kwargs["show_copy_button"] = True
                    
                # Check if 'scale' parameter is supported
                if 'scale' in init_params:
                    chatbot_kwargs["scale"] = 0
                    
            except Exception as e:
                logger.warning(f"Error checking Chatbot parameters: {e}")
                # Use minimal parameters as fallback
                chatbot_kwargs = {"height": 400}
            
            chatbot = gr.Chatbot(**chatbot_kwargs)
            
        # Follow-up question buttons
        with gr.Row():
            followup_question_buttons = []
            for i in range(FOLLOWUP_QUESTION_NUMBER):
                btn = gr.Button(f"Button {i+1}", visible=False, elem_classes="followup-question-button")
                followup_question_buttons.append(btn)

        # Tab management state
        tab_edit_uuid_state = gr.State("")
        
        # Update prompt state when changed
        prompt_textbox.change(
            fn=lambda prompt: prompt, 
            inputs=[prompt_textbox], 
            outputs=[current_prompt_state]
        )
        
        # Chat History Sidebar (using simple approach for compatibility)
        with gr.Column():
            gr.Markdown("### Chat History")
            
            @gr.render(inputs=[tab_edit_uuid_state, end_of_assistant_response_state, sidebar_names_state, current_uuid_state, chatbot, offloaded_tabs_data_storage])
            def render_chats(tab_uuid_edit, end_of_chat_response, sidebar_summaries, active_uuid, messages, tabs):
                # Ensure sidebar_summaries is a dict
                if not isinstance(sidebar_summaries, dict):
                    sidebar_summaries = {}
                    
                # Current tab button
                current_tab_button_text = sidebar_summaries.get(active_uuid, "Current Chat")
                if active_uuid not in tabs or not tabs[active_uuid]:
                    unique_id = f"current-tab-{active_uuid}-{uuid4()}"
                    gr.Button(
                        current_tab_button_text, 
                        elem_classes=["chat-tab", "active"], 
                        elem_id=unique_id
                    )
                
                # Historical tabs
                for chat_uuid, tab in reversed(tabs.items()):
                    if not tab:  # Skip empty tabs
                        continue
                        
                    elem_classes = ["chat-tab"]
                    if chat_uuid == active_uuid:
                        elem_classes.append("active")
                        
                    button_uuid_state = gr.State(chat_uuid)
                    
                    with gr.Row():
                        # Delete button
                        clear_tab_button = gr.Button(
                            "🗑",
                            scale=0,
                            elem_classes=["tab-button-control"],
                            elem_id=f"delete-btn-{chat_uuid}-{uuid4()}"
                        )
                        clear_tab_button.click(
                            fn=delete_tab,
                            inputs=[
                                current_uuid_state,
                                button_uuid_state,
                                sidebar_names_state,
                                offloaded_tabs_data_storage
                            ],
                            outputs=[
                                sidebar_names_state,
                                offloaded_tabs_data_storage,
                                chatbot
                            ]
                        )
                        
                        # Tab name/edit functionality
                        chat_button_text = sidebar_summaries.get(chat_uuid, str(chat_uuid)[:8])
                        
                        if chat_uuid != tab_uuid_edit:
                            # Edit button
                            set_edit_tab_button = gr.Button(
                                "✎",
                                scale=0,
                                elem_classes=["tab-button-control"],
                                elem_id=f"edit-btn-{chat_uuid}-{uuid4()}"
                            )
                            set_edit_tab_button.click(
                                fn=lambda x: x,
                                inputs=[button_uuid_state],
                                outputs=[tab_edit_uuid_state]
                            )
                            
                            # Tab button
                            chat_tab_button = gr.Button(
                                chat_button_text,
                                elem_id=f"chat-{chat_uuid}-{uuid4()}",
                                elem_classes=elem_classes,
                                scale=2
                            )
                            chat_tab_button.click(
                                fn=switch_tab,
                                inputs=[
                                    button_uuid_state,
                                    offloaded_tabs_data_storage,
                                    current_langgraph_state,
                                    current_uuid_state,
                                    chatbot,
                                    prompt_textbox
                                ],
                                outputs=[
                                    current_langgraph_state,
                                    current_uuid_state,
                                    chatbot,
                                    offloaded_tabs_data_storage,
                                    prompt_textbox,
                                    *followup_question_buttons
                                ]
                            )
                        else:
                            # Edit textbox
                            chat_tab_text = gr.Textbox(
                                chat_button_text,
                                scale=2,
                                interactive=True,
                                show_label=False,
                                elem_id=f"edit-text-{chat_uuid}-{uuid4()}"
                            )
                            chat_tab_text.submit(
                                fn=submit_edit_tab,
                                inputs=[
                                    button_uuid_state,
                                    sidebar_names_state,
                                    chat_tab_text
                                ],
                                outputs=[
                                    sidebar_names_state,
                                    tab_edit_uuid_state
                                ]
                            )
            
            # New chat button and clear button
            with gr.Row():
                new_chat_button = gr.Button("➕ New Chat", elem_id="new-chat-button", scale=1)
                # Check if variant parameter is supported for buttons
                try:
                    clear_button_kwargs = {"scale": 1}
                    if 'variant' in gr.Button.__init__.__code__.co_varnames:
                        clear_button_kwargs["variant"] = "secondary"
                    clear_chat_button = gr.Button("🗑️ Clear Chat", **clear_button_kwargs)
                except Exception as e:
                    logger.warning(f"Error creating clear button with variant: {e}")
                    clear_chat_button = gr.Button("🗑️ Clear Chat", scale=1)
        
        # Clear functionality - implement manually since chatbot.clear() is not available in older Gradio versions
        # We'll handle clearing through the clear chat button instead
        
        # Main chat interface - with extensive compatibility checks
        # Start with minimal required parameters
        chat_interface_kwargs = {
            "chatbot": chatbot,
            "fn": chat_fn,
            "textbox": textbox,
        }
        
        # Check if newer ChatInterface parameters are supported
        try:
            init_params = gr.ChatInterface.__init__.__code__.co_varnames
            logger.info(f"ChatInterface supported parameters: {list(init_params)}")
            
            # Check each parameter individually
            if 'additional_inputs' in init_params:
                chat_interface_kwargs["additional_inputs"] = [
                    current_langgraph_state,
                    current_uuid_state,
                    prompt_textbox,
                    checkbox_search_enabled,
                    checkbox_download_website_text,
                ]
                logger.info("Added additional_inputs to ChatInterface")
                
            if 'additional_outputs' in init_params:
                chat_interface_kwargs["additional_outputs"] = [
                    current_langgraph_state,
                    end_of_assistant_response_state
                ]
                logger.info("Added additional_outputs to ChatInterface")
            else:
                logger.warning("ChatInterface 'additional_outputs' not supported - some features may be limited")
            
            # Use tuples format to match the Chatbot for compatibility
            if 'type' in init_params:
                chat_interface_kwargs["type"] = "tuples"
                logger.info("Added type='tuples' to ChatInterface (matching Chatbot format)")
                
            # Check if 'multimodal' parameter is supported
            if 'multimodal' in init_params:
                chat_interface_kwargs["multimodal"] = multimodal
                logger.info(f"Added multimodal={multimodal} to ChatInterface")
                
        except Exception as e:
            logger.warning(f"Error checking ChatInterface parameters: {e}")
            # Keep minimal parameters as fallback
        
        # Try to create ChatInterface with compatibility handling
        try:
            chat_interface = gr.ChatInterface(**chat_interface_kwargs)
            logger.info("ChatInterface created successfully")
        except TypeError as e:
            logger.error(f"ChatInterface creation failed: {e}")
            logger.info("Falling back to minimal ChatInterface configuration")
            
            # Fallback to absolute minimal configuration
            try:
                minimal_kwargs = {
                    "chatbot": chatbot,
                    "fn": lambda message, history: (message + " (processed)", history + [(message, message + " (processed)")]),
                    "textbox": textbox,
                }
                chat_interface = gr.ChatInterface(**minimal_kwargs)
                logger.warning("Using minimal ChatInterface - advanced features disabled")
            except Exception as fallback_error:
                logger.error(f"Even minimal ChatInterface failed: {fallback_error}")
                # Create manual chat functionality as last resort
                chat_interface = None
                logger.info("Creating manual chat interface as fallback")
                
                # Manual chat submit function
                def manual_chat_submit(message, history, graph_state, uuid_val, prompt, search_enabled, download_enabled):
                    """Manual chat submission when ChatInterface is not available"""
                    try:
                        if not message.strip():
                            return history, "", graph_state
                            
                        # Add user message in tuples format
                        if not isinstance(history, list):
                            history = []
                            
                        # Create response tuple
                        response = f"Manual chat mode: {message} (ChatInterface not available in this Gradio version)"
                        history.append((message, response))
                        
                        return history, "", graph_state
                    except Exception as e:
                        logger.error(f"Error in manual chat: {e}")
                        if not isinstance(history, list):
                            history = []
                        history.append((message, f"Error: {str(e)}"))
                        return history, "", graph_state
                
                # Set up manual chat button
                textbox.submit(
                    fn=manual_chat_submit,
                    inputs=[
                        textbox,
                        chatbot,
                        current_langgraph_state,
                        current_uuid_state,
                        prompt_textbox,
                        checkbox_search_enabled,
                        checkbox_download_website_text
                    ],
                    outputs=[chatbot, textbox, current_langgraph_state]
                )

        # New chat button functionality
        new_chat_button.click(
            new_tab,
            inputs=[
                current_uuid_state,
                current_langgraph_state,
                chatbot,
                offloaded_tabs_data_storage,
                prompt_textbox,
                sidebar_names_state,
            ],
            outputs=[
                current_uuid_state,
                current_langgraph_state,
                chatbot,
                offloaded_tabs_data_storage,
                prompt_textbox,
                sidebar_names_state,
                *followup_question_buttons,
            ]
        )
        
        # Clear chat button functionality
        def clear_current_chat():
            """Clear the current chat and reset state"""
            new_state, new_uuid = clear()
            # Clear followup buttons and return empty tuples list
            cleared_buttons = [gr.Button(visible=False) for _ in range(FOLLOWUP_QUESTION_NUMBER)]
            return [], new_state, new_uuid, *cleared_buttons
        
        clear_chat_button.click(
            fn=clear_current_chat,
            inputs=[],
            outputs=[
                chatbot,
                current_langgraph_state,
                current_uuid_state,
                *followup_question_buttons
            ]
        )

        # Follow-up button functionality
        def click_followup_button(btn):
            buttons = [gr.Button(visible=False) for _ in range(len(followup_question_buttons))]
            return btn, *buttons

        # Handle followup buttons based on whether ChatInterface is available
        if chat_interface is not None:
            for btn in followup_question_buttons:
                try:
                    btn.click(
                        fn=click_followup_button,
                        inputs=[btn],
                        outputs=[
                            chat_interface.textbox if hasattr(chat_interface, 'textbox') else textbox,
                            *followup_question_buttons
                        ]
                    ).success(lambda: None, js=TRIGGER_CHATINTERFACE_BUTTON)
                except Exception as e:
                    logger.warning(f"Error setting up followup button: {e}")
                    # Fallback to basic button functionality
                    btn.click(
                        fn=click_followup_button,
                        inputs=[btn],
                        outputs=[textbox, *followup_question_buttons]
                    )
        else:
            logger.warning("ChatInterface not available - followup buttons will have limited functionality")
            for btn in followup_question_buttons:
                btn.click(
                    fn=click_followup_button,
                    inputs=[btn],
                    outputs=[textbox, *followup_question_buttons]
                )

        # Event handlers for chatbot changes - with compatibility checks
        def setup_change_handler(fn, inputs, outputs, trigger_mode=None):
            """Helper function to set up change handlers with optional trigger_mode"""
            try:
                # Get the change method's parameter names
                change_params = chatbot.change.__code__.co_varnames
                
                if trigger_mode and 'trigger_mode' in change_params:
                    return chatbot.change(fn=fn, inputs=inputs, outputs=outputs, trigger_mode=trigger_mode)
                else:
                    return chatbot.change(fn=fn, inputs=inputs, outputs=outputs)
            except Exception as e:
                logger.warning(f"Error setting up change handler: {e}")
                # Fallback to basic change handler
                try:
                    return chatbot.change(fn=fn, inputs=inputs, outputs=outputs)
                except Exception as fallback_error:
                    logger.error(f"Failed to set up change handler: {fallback_error}")
                    return None
        
        setup_change_handler(
            fn=populate_followup_questions,
            inputs=[
                end_of_assistant_response_state,
                chatbot,
                current_uuid_state
            ],
            outputs=[
                *followup_question_buttons,
                end_of_assistant_response_state
            ],
            trigger_mode="multiple"
        )
        
        setup_change_handler(
            fn=summarize_chat,
            inputs=[
                end_of_assistant_response_state,
                chatbot,
                sidebar_names_state,
                current_uuid_state
            ],
            outputs=[
                sidebar_names_state,
                end_of_assistant_response_state
            ],
            trigger_mode="multiple"
        )
        
        setup_change_handler(
            fn=lambda x: x,
            inputs=[chatbot],
            outputs=[chatbot_message_storage],
            trigger_mode="always_last"
        )

        # Load event handlers - only add these if we have BrowserState
        if has_browser_state:
            @demo.load(
                inputs=[is_new_user_for_greeting, chatbot_message_storage],
                outputs=[chatbot_message_storage, is_new_user_for_greeting]
            )
            def handle_initial_greeting_load(current_is_new_user_flag: bool, existing_chat_history: list):
                """Handle initial greeting when the app loads"""
                if current_is_new_user_flag:
                    greeting_message_text = load_initial_greeting()
                    
                    if not isinstance(existing_chat_history, list):
                        existing_chat_history = []
                    
                    # Always use tuples format for compatibility
                    greeting_entry = (None, greeting_message_text)
                    updated_chat_history = [greeting_entry] + existing_chat_history
                    updated_is_new_user_flag = False
                    logger.info("Greeting added for new user (tuples format).")
                    return updated_chat_history, updated_is_new_user_flag
                else:
                    logger.info("Not a new user or already greeted.")
                    if not isinstance(existing_chat_history, list):
                        existing_chat_history = []
                    return existing_chat_history, False

            @demo.load(inputs=[chatbot_message_storage], outputs=[chatbot])
            def load_messages(history):
                """Load stored messages into chatbot"""
                if isinstance(history, list):
                    return history
                return []

            @demo.load(inputs=[current_prompt_state], outputs=[prompt_textbox])
            def load_prompt(current_prompt):
                """Load stored prompt"""
                if current_prompt:
                    return current_prompt
                return "You are a helpful DIY assistant."
        else:
            # For regular State, add a simple greeting on load
            @demo.load(outputs=[chatbot])
            def load_initial_greeting():
                """Load initial greeting for users without BrowserState"""
                greeting_text = load_initial_greeting()
                # Use tuples format for maximum compatibility
                return [(None, greeting_text)]

    # Launch the application
    demo.launch(debug=True, share=True)