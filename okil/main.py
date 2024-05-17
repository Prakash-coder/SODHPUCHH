from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    SummaryIndex,
)
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, SummaryIndex
from llama_index.agent.openai import OpenAIAgent
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.core.objects import ObjectIndex
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI
from llama_index.core.callbacks import CallbackManager
from llama_index.core.node_parser import SentenceSplitter
import os

from nameprocessor import process_name

# Load law documents
law_docs = {}
law_directory = "data"

for filename in os.listdir(law_directory):
    if filename.endswith(".pdf"):
        file_title = process_name(os.path.splitext(filename)[0])
        loaded_data_path = os.path.join("data", file_title)
        if not os.path.exists(loaded_data_path):
            f = os.path.join(law_directory, filename)
            law_docs[file_title] = SimpleDirectoryReader(input_files=[f]).load_data()
    
# Set up OpenAI model and embedding
Settings.llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

# Initialize agents dictionary and query engines
agents = {}
query_engines = {}

# Initialize node parser
node_parser = SentenceSplitter()

# Initialize list to store all nodes
all_nodes = []

# Loop through law documents
for file_title, node in law_docs.items():
    # Define paths for index persistence
    persist_dir = os.path.join("data", file_title)
    
    # Split nodes into sentences
    if not os.path.exists(persist_dir):
        nodes = node_parser.get_nodes_from_documents(node)
        all_nodes.extend(nodes)

    # Build or load vector index
    if not os.path.exists(persist_dir):
        vector_index = VectorStoreIndex(nodes)
        vector_index.storage_context.persist(persist_dir=persist_dir)
    else:
        vector_index = load_index_from_storage(StorageContext.from_defaults(persist_dir=persist_dir))

    # Build summary index
    summary_index = SummaryIndex(nodes)

    # Define query engines
    vector_query_engine = vector_index.as_query_engine(llm=Settings.llm)
    summary_query_engine = summary_index.as_query_engine(llm=Settings.llm)

    # Define query engine tools
    query_engine_tools = [
        QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name="vector_tool",
                description=(
                    f"Useful for questions related to specific laws or rules and regulations related to {file_title}."
                ),
            ),
        ),
        QueryEngineTool(
            query_engine=summary_query_engine,
            metadata=ToolMetadata(
                name="summary_tool",
                description=(
                    f"Useful for any requests that require a holistic summary of everything about {file_title}."
                ),
            ),
        ),
    ]

    # Build agent
    function_llm = OpenAI(model="gpt-4")
    agent = OpenAIAgent.from_tools(
        query_engine_tools,
        llm=function_llm,
        verbose=True,
        system_prompt=f"You are a specialized agent designed to answer queries about {file_title}. You must ALWAYS use at least one of the tools provided when answering a question; do NOT rely on prior knowledge.",
    )

    # Store agent and query engine
    agents[file_title] = agent
    query_engines[file_title] = vector_index.as_query_engine(similarity_top_k=2)

# Define tool for each document agent
all_tools = []
for file_title in law_docs.keys():
    law_summary = (
        f"This content contains law articles about {file_title}. Use this tool if you want to answer any questions about {file_title}.\n"
    )
    doc_tool = QueryEngineTool(
        query_engine=agents[file_title],
        metadata=ToolMetadata(
            name=file_title,
            description=law_summary,
        ),
    )
    all_tools.append(doc_tool)

# Define object index and retriever over these tools
obj_index = ObjectIndex.from_objects(
    all_tools,
    index_cls=VectorStoreIndex,
)

# Initialize top-level OpenAI agent
top_agent = OpenAIAgent.from_tools(
    tool_retriever=obj_index.as_retriever(similarity_top_k=3),
    system_prompt="You are an agent designed to answer queries about a set of law documents. Please always use the tools provided to answer a question. Do not rely on prior knowledge.",
    verbose=True,
)

# Build base index and query engine for all nodes
base_index = VectorStoreIndex(all_nodes)
base_query_engine = base_index.as_query_engine(similarity_top_k=4)

while True:
    prompt = input("Enter a prompt (q to quit): ")
    if prompt.lower() == "q":
        break
    
    # Call the base query engine to process the user's prompt
    response = top_agent.query(prompt)
    
    # Print the response generated by the agent
    print(response)
