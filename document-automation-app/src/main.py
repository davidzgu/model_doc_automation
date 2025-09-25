from automation.processor import DocumentProcessor
from langchain_utils.chain import LangChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_utils.python_tools import python_tool

# def main():
#     # Initialize the document processor
#     document_processor = DocumentProcessor()
    
#     # Initialize the LangChain
#     lang_chain = LangChain()
    
#     # Example input document
#     input_document = "path/to/input/document.pdf"
    
#     # Process the document
#     processed_document = document_processor.process_document(input_document)
    
#     # Create a processing chain
#     processing_chain = lang_chain.create_chain()
    
#     # Run the chain on the processed document
#     output = lang_chain.run_chain(processing_chain, processed_document)
    
#     # Output the results
#     print("Output:", output)


def main_func():



    llm = ChatOpenAI(
        model="llama31", 
        openai_api_base="http://localhost:11434/v1",
        openai_api_key="not-needed"
    )

    template = """
    You are an assistant writer that generates a report based on the user prompt and optional script outputs.

    User Prompt:
    {user_prompt}

    Script Output:
    {script_output}

    Generate a simple document below:
    """

    doc_prompt = PromptTemplate(
        input_variables=["user_prompt", "script_output"],
        template=template
    )
    
    chain = doc_prompt | llm


    user_prompt = "Generate a summary report of the testing results. Note that the threshold of the result is 4 and the metric is that if output <= threshold, then the test failed."
    script_output = python_tool("./document-automation-app/src/input/pricing_model.py")

    document = chain.invoke(input={"user_prompt":user_prompt, "script_output":script_output})
    print(document.content)

if __name__ == "__main__":
    # print(python_tool("./document-automation-app/src/input/pricing_model.py"))
    main_func()