import logging
import os
import uuid
import aiohttp
import json
import httpx
import io 
import requests
from urllib.parse import quote

from typing import Annotated
from typing import TypedDict, List, Optional, Literal

from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from trafilatura import extract

from huggingface_hub import InferenceClient
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

from langchain_core.messages import AIMessage, HumanMessage, AnyMessage, ToolCall, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from langchain_community.tools import TavilySearchResults
from langchain_huggingface import HuggingFacePipeline

from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, START, END, add_messages

from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.checkpoint.memory import MemorySaver

from langgraph.types import Command, interrupt


class State(TypedDict):
    messages: Annotated[list, add_messages]

class DebugToolNode(ToolNode):
    async def invoke(self, state, config=None):
        print("🛠️ ToolNode activated")
        print(f"Available tools: {[tool.name for tool in self.tool_map.values()]}")
        print(f"Tool calls in last message: {state.messages[-1].tool_calls}")
        return await super().invoke(state, config)


logger = logging.getLogger(__name__)
ASSISTANT_SYSTEM_PROMPT_BASE = """"""
search_enabled = bool(os.environ.get("TAVILY_API_KEY"))

try:
    with open('brainstorming_system_prompt.txt', 'r') as file:
        brainstorming_system_prompt = file.read()
except FileNotFoundError:
    print("File 'system_prompt.txt' not found!")
except Exception as e:
    print(f"Error reading file: {e}")

def evaluate_idea_completion(response) -> bool:
    """
    Evaluates whether the assistant's response indicates a complete DIY project idea.
    """
    required_keywords = ["materials", "dimensions", "tools", "steps"]

    if isinstance(response, dict):
        response_text = ' '.join(str(value).lower() for value in response.values())
    elif isinstance(response, str):
        response_text = response.lower()
    else:
        response_text = str(response).lower()

    return all(keyword in response_text for keyword in required_keywords)

@tool
async def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = await interrupt({"query": query})
    return human_response["data"]

@tool
async def download_website_text(url: str) -> str:
    """Download the text from a website"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                downloaded = await response.text()
        result = extract(downloaded, include_formatting=True, include_links=True, output_format='json', with_metadata=True)
        return result or "No text found on the website"
    except Exception as e:
        logger.error(f"Failed to download {url}: {str(e)}")
        return f"Error retrieving website content: {str(e)}"
    
@tool
async def finalize_idea() -> str:
    """Marks the brainstorming phase as complete. This function does nothing else."""
    return "Brainstorming finalized."

tools = [download_website_text, human_assistance, finalize_idea]
memory = MemorySaver()

if search_enabled:
    tavily_search_tool = TavilySearchResults(
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=True,
    )
    tools.append(tavily_search_tool)
else:
    print("TAVILY_API_KEY environment variable not found. Websearch disabled")

# Initialize Hugging Face models
print("Loading transformer models...")

# Option 1: Use Hugging Face Inference API (recommended for production)
def create_hf_inference_model(model_name="microsoft/DialoGPT-medium"):
    """Create a Hugging Face Inference API client"""
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN not found. Some features may not work.")
        return None
    
    return InferenceClient(
        model=model_name,
        token=hf_token,
    )

# Option 2: Load local model (for offline use)
def create_local_model(model_name="microsoft/DialoGPT-small"):
    """Create a local transformer model"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {model_name} on {device}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        
        # Add padding token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        text_generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        return HuggingFacePipeline(pipeline=text_generator)
    except Exception as e:
        print(f"Error loading local model: {e}")
        return None

# Option 3: Use Llama via Hugging Face (requires more resources)
def create_llama_model():
    """Create Llama model - requires significant GPU memory"""
    try:
        model_name = "meta-llama/Llama-2-7b-chat-hf"  # or "meta-llama/Llama-3.2-3B"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True,  # Use 8-bit quantization to save memory
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        text_generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
            torch_dtype=torch.float16,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
        )
        
        return HuggingFacePipeline(pipeline=text_generator)
    except Exception as e:
        print(f"Error loading Llama model: {e}")
        return None

# Choose which model to use
MODEL_TYPE = os.environ.get("MODEL_TYPE", "local")  # Options: "inference", "local", "llama"

if MODEL_TYPE == "inference":
    # Use Hugging Face Inference API
    hf_client = create_hf_inference_model("microsoft/DialoGPT-medium")
    model = hf_client
elif MODEL_TYPE == "llama":
    # Use local Llama model
    model = create_llama_model()
elif MODEL_TYPE == "local":
    # Use local lightweight model
    model = create_local_model("microsoft/DialoGPT-small")
else:
    print("Invalid MODEL_TYPE. Using local model as fallback.")
    model = create_local_model("microsoft/DialoGPT-small")

# Fallback to a simple model if primary model fails
if model is None:
    print("Primary model failed to load. Using fallback model...")
    model = create_local_model("distilgpt2")

# Set all model references to use the same transformer model
weak_model = model
assistant_model = model
prompt_planning_model = model
threed_object_gen_model = model

print(f"Using model type: {MODEL_TYPE}")
print(f"Model loaded successfully: {model is not None}")

# Custom function to generate responses with transformer models
async def generate_with_transformer(prompt_text, messages, max_length=512):
    """Generate response using transformer model"""
    try:
        # Combine messages into a single prompt
        conversation = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                conversation += f"Human: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                if isinstance(msg.content, str):
                    conversation += f"Assistant: {msg.content}\n"
                elif isinstance(msg.content, list):
                    content = " ".join([item.get("text", "") for item in msg.content if isinstance(item, dict)])
                    conversation += f"Assistant: {content}\n"
            elif isinstance(msg, SystemMessage):
                conversation += f"System: {msg.content}\n"
        
        # Add the current prompt
        full_prompt = f"{prompt_text}\n\nConversation:\n{conversation}\nAssistant:"
        
        if MODEL_TYPE == "inference" and hf_client:
            # Use Hugging Face Inference API
            response = await hf_client.text_generation(
                full_prompt,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                stop_sequences=["Human:", "System:"]
            )
            return response
        else:
            # Use local model
            if hasattr(model, 'invoke'):
                response = model.invoke(full_prompt)
                return response
            elif hasattr(model, '__call__'):
                response = model(full_prompt)
                if isinstance(response, list) and len(response) > 0:
                    return response[0].get('generated_text', '').replace(full_prompt, '').strip()
                return str(response)
            else:
                return "Model not properly configured"
                
    except Exception as e:
        logger.error(f"Error generating with transformer: {e}")
        return f"Error generating response: {e}"

# Custom tool calling simulation for transformer models
def simulate_tool_calls(response_text):
    """Simulate tool calls by parsing response text for specific patterns"""
    tool_calls = []
    
    # Look for patterns like "CALL_TOOL: human_assistance(query='...')"
    if "human_assistance" in response_text.lower():
        # Extract query from response
        import re
        pattern = r"human_assistance.*?[\(\"']([^\"']+)[\)\"']"
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            query = match.group(1)
            tool_calls.append({
                "name": "human_assistance",
                "arguments": {"query": query},
                "id": f"call_{uuid.uuid4()}"
            })
    
    if "finalize_idea" in response_text.lower() or "idea finalized" in response_text.lower():
        tool_calls.append({
            "name": "finalize_idea",
            "arguments": {"idea_name": "Generated Idea"},
            "id": f"call_{uuid.uuid4()}"
        })
    
    return tool_calls

class GraphProcessingState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    prompt: str = Field(default_factory=str, description="The prompt to be used for the model")
    tools_enabled: dict = Field(default_factory=dict, description="The tools enabled for the assistant")
    search_enabled: bool = Field(default=True, description="Whether to enable search tools")
    next_stage: str = Field(default="", description="The next stage to execute, decided by the guidance node.")

    tool_call_required: bool = Field(default=False, description="Whether a tool should be called from brainstorming.")
    loop_brainstorming: bool = Field(default=False, description="Whether to loop back to brainstorming for further iteration.")

    # Completion flags for each stage
    idea_complete: bool = Field(default=False)
    brainstorming_complete: bool = Field(default=False)
    planning_complete: bool = Field(default=False)
    drawing_complete: bool = Field(default=False)
    product_searching_complete: bool = Field(default=False)
    purchasing_complete: bool = Field(default=False)

    generated_image_url_from_dalle: str = Field(default="", description="The generated_image_url_from_dalle.")

async def guidance_node(state: GraphProcessingState, config=None):
    print("\n🕵️‍♀️🕵️‍♀️ |  start | progress checking node \n")

    if state.messages:
        last_message = state.messages[-1]
        if isinstance(last_message, HumanMessage):
            print(f"🧑 Human: {last_message.content}\n")
        elif isinstance(last_message, AIMessage):
            if last_message.content:
                if isinstance(last_message.content, list):
                    texts = [item.get('text', '') for item in last_message.content if isinstance(item, dict) and 'text' in item]
                    if texts:
                        print(f"🤖 AI: {' '.join(texts)}\n")
                elif isinstance(last_message.content, str):
                    print(f"🤖 AI: {last_message.content}\n")
        elif isinstance(last_message, SystemMessage):
            print(f"⚙️ System: {last_message.content}\n")
        elif isinstance(last_message, ToolMessage):
            print(f"🛠️ Tool: {last_message.content}\n")
    else:
        print("\n(No messages found.)")

    # Define the order of stages
    stage_order = ["brainstorming", "planning", "drawing", "product_searching", "purchasing"]
    
    # Identify completed and incomplete stages
    completed = [stage for stage in stage_order if getattr(state, f"{stage}_complete", False)]
    incomplete = [stage for stage in stage_order if not getattr(state, f"{stage}_complete", False)]
    
    # Determine the next stage
    if not incomplete:
        return {
            "messages": [AIMessage(content="All DIY project stages are complete!")],
            "next_stage": "end_project",
            "pending_approval_stage": None,
        }
    else:
        next_stage = incomplete[0]
        print(f"Next Stage: {next_stage}")
        print("\n🕵️‍♀️🕵️‍♀️ |  end | progress checking node \n")
        return {
            "messages": [],
            "next_stage": next_stage,
            "pending_approval_stage": None,
        }
        
def guidance_routing(state: GraphProcessingState) -> str:
    print("\n🔀🔀 Routing checkpoint 🔀🔀\n")    
    print(f"Next Stage: {state.next_stage}\n")
    print(f"Brainstorming complete: {state.brainstorming_complete}")
    print(f"Planning complete: {state.planning_complete}")
    print(f"Drawing complete: {state.drawing_complete}")
    print(f"Product searching complete: {state.product_searching_complete}\n")
    
    next_stage = state.next_stage
    if next_stage == "brainstorming":
        return "brainstorming_node"
    elif next_stage == "planning":
        return "prompt_planning_node"
    elif next_stage == "drawing":
        return "generate_3d_node"
    elif next_stage == "product_searching":
        print('\n Product searching stage reached')
        return END
    else:
        return END

async def brainstorming_node(state: GraphProcessingState, config=None):
    print("\n🧠🧠 |  start | brainstorming Node \n")

    if not model:
        return {"messages": [AIMessage(content="Model not available for brainstorming.")]}

    filtered_messages = [
        message for message in state.messages
        if isinstance(message, (HumanMessage, AIMessage, SystemMessage, ToolMessage)) and message.content
    ]

    if not filtered_messages:
        filtered_messages.append(AIMessage(content="No valid messages provided."))

    stage_order = ["brainstorming", "planning", "drawing", "product_searching", "purchasing"]
    completed = [stage for stage in stage_order if getattr(state, f"{stage}_complete", False)]
    incomplete = [stage for stage in stage_order if not getattr(state, f"{stage}_complete", False)]

    if not incomplete:
        print("All stages complete!")
        ai_all_complete_msg = AIMessage(content="All DIY project stages are complete!")
        return {
            "messages": [ai_all_complete_msg],
            "next_stage": "end_project",
            "pending_approval_stage": None,
        }

    guidance_prompt_text = """
You are a warm, encouraging, and knowledgeable AI assistant, acting as a Creative DIY Collaborator. Your primary goal is to guide the user through a friendly and inspiring conversation to finalize ONE specific, viable DIY project idea.

Your Conversational Style & Strategy:
1. Be an Active Listener: Start by acknowledging and validating the user's input.
2. Ask Inspiring, Open-Ended Questions: Make them feel personal and insightful.
3. Act as a Knowledgeable Guide: When a user is unsure, proactively suggest appealing ideas.
4. Guide, Don't Just Gatekeep: When an idea almost meets criteria, guide it towards feasibility.

Critical Criteria for the Final DIY Project Idea:
1. Buildable: Achievable by an average person with basic DIY skills.
2. Common Materials/Tools: Uses only materials and basic tools commonly available.
3. Avoid Specializations: No specialized electronics, 3D printing, or complex machinery.
4. Tangible Product: The final result must be a physical, tangible item.

If you need to ask the user a question, respond with: "CALL_TOOL: human_assistance(query='your question here')"
If an idea is finalized, respond with: "IDEA FINALIZED: [Name of the Idea]"
"""

    if state.prompt:
        final_prompt = "\n".join([guidance_prompt_text, state.prompt, ASSISTANT_SYSTEM_PROMPT_BASE])
    else:
        final_prompt = "\n".join([guidance_prompt_text, ASSISTANT_SYSTEM_PROMPT_BASE])

    try:
        # Generate response using transformer model
        response_text = await generate_with_transformer(final_prompt, filtered_messages)
        
        # Simulate tool calls
        tool_calls = simulate_tool_calls(response_text)
        
        # Create AI message
        ai_message = AIMessage(content=response_text)
        
        updates = {
            "messages": [ai_message],
            "tool_calls": tool_calls,
        }

        print(f'\n🔍 response from brainstorm: {response_text}')
        
        # Check for finalization
        if "IDEA FINALIZED:" in response_text.upper():
            print('✅ final idea')
            updates.update({
                "brainstorming_complete": True,
                "tool_call_required": False,
                "loop_brainstorming": False,
            })
        elif tool_calls:
            print('🛠️ tool call requested at brainstorming node')
            updates.update({
                "tool_call_required": True,
                "loop_brainstorming": False,
            })
        else:
            print('💬 decided to keep brainstorming')
            updates.update({
                "tool_call_required": False,
                "loop_brainstorming": True,
            })

        print("\n🧠🧠 |  end | brainstorming Node \n")
        return updates
        
    except Exception as e:
        print(f"Error: {e}")
        return {
            "messages": [AIMessage(content="Error in brainstorming.")],
            "next_stage": "brainstorming"
        }

async def prompt_planning_node(state: GraphProcessingState, config=None):
    print("\n🚩🚩 |  start | prompt planning Node \n")
    
    if not model:
        return {"messages": [AIMessage(content="Model not available for planning.")]}

    filtered_messages = state.messages
    if not filtered_messages:
        filtered_messages.append(AIMessage(content="No valid messages provided."))

    guidance_prompt_text = """
You are a creative AI assistant acting as a DIY Project Brainstorming & 3D-Prompt Generator. Your mission is to:

1. Brainstorm and refine one specific, viable DIY project idea.
2. Identify the single key component from that idea that should be 3D-modeled.
3. Produce a final, precise text prompt for a 3D-generation endpoint.

Critical Criteria for the DIY Project:
• Buildable by an average person with only basic DIY skills.
• Uses common materials/tools (e.g., wood, screws, glue, paint; hammer, saw, drill).
• No specialized electronics, 3D printers, or proprietary parts.
• Results in a tangible, physical item.

When the DIY idea is fully detailed and meets all criteria, output exactly:
ACCURATE PROMPT FOR MODEL GENERATING: [Your final single-paragraph prompt here]
"""

    if state.prompt:
        final_prompt = "\n".join([guidance_prompt_text, state.prompt, ASSISTANT_SYSTEM_PROMPT_BASE])
    else:
        final_prompt = "\n".join([guidance_prompt_text, ASSISTANT_SYSTEM_PROMPT_BASE])

    try:
        # Generate response using transformer model
        response_text = await generate_with_transformer(final_prompt, filtered_messages)
        
        # Create AI message
        response = AIMessage(content=response_text)
        updates = {"messages": [response]}

        print(f'\nResponse: {response_text}')

        # Check for finalization signal
        if "ACCURATE PROMPT FOR MODEL GENERATING" in response_text:
            dalle_prompt_text = response_text.replace("ACCURATE PROMPT FOR MODEL GENERATING:", "").strip()
            print(f"\n🤖🤖🤖🤖Extracted prompt: {dalle_prompt_text}")

            # For this example, we'll simulate image generation
            # In practice, you would call your image generation API here
            generated_image_url = "https://example.com/generated_image.jpg"  # Placeholder
            
            updates["messages"].append(AIMessage(content=f"Image generation prompt created: {dalle_prompt_text}"))
            
            updates.update({
                "generated_image_url_from_dalle": generated_image_url,
                "planning_complete": True,
                "tool_call_required": False,
                "loop_planning": False,
            })
        else:
            # Check if a tool call was requested
            tool_calls = simulate_tool_calls(response_text)
            if tool_calls:
                updates.update({
                    "tool_call_required": True,
                    "loop_planning": False,
                })
            else:
                updates.update({
                    "tool_call_required": False,
                    "loop_planning": True,
                })

        print("\n🚩🚩 |  end | prompt planning Node \n")
        return updates

    except Exception as e:
        print(f"Error in prompt_planning node: {e}")
        return {
            "messages": [AIMessage(content="Error in prompt_planning node.")],
            "next_stage": state.next_stage or "planning"
        }

async def generate_3d_node(state: GraphProcessingState, config=None):
    print("\n🚀🚀🚀 |  start | Generate 3D Node 🚀🚀🚀\n")    
    
    # Get the image URL
    hardcoded_image_url = state.generated_image_url_from_dalle
    print(f"Using image_url: {hardcoded_image_url}")

    # For this example, we'll simulate 3D generation
    # In practice, you would call your 3D generation API here
    
    try:
        # Simulate 3D model generation
        print("Simulating 3D model generation...")
        
        # Create output directory
        output_dir = "generated_3d_models"
        os.makedirs(output_dir, exist_ok=True)
        
        # Simulate successful generation
        file_name = f"model_{uuid.uuid4()}.glb"
        file_path = os.path.join(output_dir, file_name)
        
        # Create a placeholder file
        with open(file_path, "w") as f:
            f.write("# Simulated 3D model file\n")
        
        print(f"Success: 3D model saved to {file_path}")
        return {
            "messages": [AIMessage(content=f"3D object generation successful: {file_path}")],
            "drawing_complete": True,
            "three_d_model_path": file_path,
            "next_stage": state.get("next_stage") or 'end'
        }
        
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)
        return {"messages": [AIMessage(content=f"Failed to generate 3D object: {error_message}")]}

def define_workflow() -> CompiledStateGraph:
    """Defines the workflow graph"""
    workflow = StateGraph(GraphProcessingState)

    # Add nodes
    workflow.add_node("tools", DebugToolNode(tools))
    workflow.add_node("guidance_node", guidance_node)
    workflow.add_node("brainstorming_node", brainstorming_node)
    workflow.add_node("prompt_planning_node", prompt_planning_node)
    workflow.add_node("generate_3d_node", generate_3d_node)

    # Edges
    workflow.add_conditional_edges(
        "guidance_node",
        guidance_routing,
        {
            "brainstorming_node": "brainstorming_node",
            "prompt_planning_node": "prompt_planning_node",
            "generate_3d_node": "generate_3d_node"
        }
    )

    workflow.add_conditional_edges("brainstorming_node", tools_condition)
    workflow.add_conditional_edges("prompt_planning_node", tools_condition)
    
    workflow.add_edge("tools", "guidance_node")
    workflow.add_edge("brainstorming_node", "guidance_node")
    workflow.add_edge("prompt_planning_node", "guidance_node")
    workflow.add_edge("generate_3d_node", "guidance_node")

    workflow.set_entry_point("guidance_node")
    compiled_graph = workflow.compile(checkpointer=memory)
    
    try:
        img_bytes = compiled_graph.get_graph().draw_mermaid_png()
        with open("graph.png", "wb") as f:
            f.write(img_bytes)
        print("Graph image saved as graph.png")
    except Exception as e:
        print("Can't print the graph:")
        print(e)

    return compiled_graph

graph = define_workflow()

# Example usage function
async def run_diy_assistant(user_input: str):
    """Run the DIY assistant with user input"""
    config = {"configurable": {"thread_id": "1"}}
    
    initial_state = GraphProcessingState(
        messages=[HumanMessage(content=user_input)],
        prompt="",
        tools_enabled={"download_website_text": True, "tavily_search_results_json": search_enabled},
        search_enabled=search_enabled
    )
    
    try:
        result = await graph.ainvoke(initial_state, config)
        return result
    except Exception as e:
        print(f"Error running DIY assistant: {e}")
        return {"error": str(e)}

# Example of how to run
if __name__ == "__main__":
    import asyncio
    
    async def main():
        user_input = "I want to build something for my garden"
        result = await run_diy_assistant(user_input)
        print("Final result:", result)
    
    # asyncio.run(main())
    print("DIY Assistant with transformer models loaded successfully!")
    print(f"Available model: {model}")
    print("Use the graph object to run your workflow.")