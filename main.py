from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from pydantic import BaseModel
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.query_pipeline import QueryPipeline
from prompts import context, code_parser_template
from code_reader import code_reader
from dotenv import load_dotenv
import ast
import os
import re
 
# import the .env file
load_dotenv()
 
# create an LLM with the model mistral of Ollama
llm = Ollama(model = "mistral", request_timeout = 360.0)
 
# responsible for taking the file and transforming it to a more suitable format
# Type of the result : markdown
parser = LlamaParse(result_type = "markdown")
# create a dictionary which stores the key-value pairs representing by the type of the file and the parser created
# For each type of file, we can use different parser to parse


file_extractor = {".pdf":parser}
# read files in the ./data with parser created
# store the parsed and loaded files into
# the variable "documents"
documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()
 
# use the embedding modle bge-m3
embed_model = resolve_embed_model("local:BAAI/bge-m3")
# vector store index of documents
vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
# convert the vector store index into a query engine which receives a query,
# search the documents or vectors from the vector index
query_engine = vector_index.as_query_engine(llm=llm)
 
tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="api_documentation",
            description="this gives documentation about code for an API. Use this for reading docs for API"
        ),
    ),
    # this tool is to open and read a file
    code_reader,
]
 
code_llm = Ollama(model="codellama", request_timeout = 3600.0)
agent = ReActAgent.from_tools(tools, llm=code_llm, verbose=True, context=context)
 
class CodeOutput(BaseModel):
    code: str
    description: str
    filename: str
parser = PydanticOutputParser(CodeOutput)
json_prompt_str = parser.format(code_parser_template)
json_prompt_tmpl = PromptTemplate(json_prompt_str)
output_pipeline = QueryPipeline(chain=[json_prompt_tmpl, llm])
 
while(prompt := input("Enter a prompt (q to quit): ")) != "q":
    retries = 0
 
    while retries < 3:
        try:
            result = agent.query(prompt)
            print("Agent query result:", result)
            next_result = output_pipeline.run(response=result)
            print("Output pipeline result:", next_result)
            cleaned_json = ast.literal_eval(str(next_result).replace("assistant:", ""))
            print("Cleaned JSON:", cleaned_json)
            break
        except Exception as e:
            retries += 1
            print(f"Error occured, retry #{retries}:", e)
 
    if retries >=3:
        print("Unable to process request, try again...")
        continue
 
    print("Code generated")
    print(cleaned_json["code"])
 
    print("\n\nDescription",cleaned_json["description"])
 
    filename = cleaned_json["filename"]
    filename = re.sub(r'[\\/*?:"<>|]', "_", filename)
 
    try:
        with open(os.path.join("output",filename), "w") as f:
            f.write(cleaned_json["code"])
        print("Saved file", filename)
    except:
        print("Error saving file...")