from langchain.prompts.prompt import PromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from config import Config
from rag.document_knowledge import DocumentKnowledge

class Rag:
    rag_prompt_template = PromptTemplate(
    input_variables=["context", "input"],
    template="""
    You are a knowledge-driven assistant for requirements engineering. Use the following context to enhance your response:
    Context: {context}
    Query: {input}
    Provide a detailed, structured response incorporating the context, following chain-of-thought reasoning.
    """
    )
    doc_prompt = PromptTemplate(
        input_variables=["page_content"],
        template="{page_content}"
    )

    @classmethod
    def get_rag_chain(cls):
        combine_docs_chain = create_stuff_documents_chain(
            llm=Config.get_llm(),
            prompt=cls.rag_prompt_template,
            document_prompt=cls.doc_prompt
        )
        return create_retrieval_chain(
            retriever=DocumentKnowledge.get_vector_document().as_retriever(search_kwargs={"k": 3}),
            combine_docs_chain=combine_docs_chain
        )