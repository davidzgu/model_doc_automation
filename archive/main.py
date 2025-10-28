from automation.processor import DocumentProcessor
from langchain_utils.chain import LangChain

def main():
    # Initialize the document processor
    document_processor = DocumentProcessor()
    
    # Initialize the LangChain
    lang_chain = LangChain()
    
    # Example input document
    input_document = "path/to/input/document.pdf"
    
    # Process the document
    processed_document = document_processor.process_document(input_document)
    
    # Create a processing chain
    processing_chain = lang_chain.create_chain()
    
    # Run the chain on the processed document
    output = lang_chain.run_chain(processing_chain, processed_document)
    
    # Output the results
    print("Output:", output)

if __name__ == "__main__":
    main()