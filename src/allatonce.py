# All At One Place Implementation of Medic.AI
import os
import warnings
from load_chunk import load_pdf, text_split
from langchain_community.embeddings import HuggingFaceEmbeddings 
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from openai import OpenAI
from dotenv import load_dotenv
from langchain.llms.openai import OpenAI as LangchainOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


load_dotenv()

warnings.filterwarnings("ignore")

class MedicAI:
    def __init__(self, data_path):
        """
        Just Pass Data Path to the Class and it will handle the rest
        """
        self.data_path = data_path
        self.extracted_data = load_pdf(data=self.data_path)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # def create_embeddings(self):
    #     return self.embeddings.embed_query("hey there")
    def setup_pinecone(self, api_key, index_name = "medicoai"):
        pc = Pinecone(api_key=api_key)
        return pc 

    def create_pinecone_index(self, api_key, index_name = "medicoai"):
        pc = Pinecone(api_key = api_key)
        pc.create_index(
            name=index_name,
            dimension= 384, 
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ) 
        )
        return pc 
    
    def add_embeddings_to_pinecone(documents, index_name, embedding):
        docsearch = PineconeVectorStore.from_documents(
           documents= documents,
           index_name= index_name,
           embedding= embedding
        )
        return docsearch
    
    def get_docsearch_from_pinecone(self, index_name, embeddings):
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )
        return docsearch
    
    def setup_local_gemma(self):
        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        llm = LangchainOpenAI(
        openai_api_key="lm-studio",
        openai_api_base="http://localhost:1234/v1",
        model_name="gemma-2-2b-instruct",
        temperature=0.7
        )
        return client, llm 
    
    def setup_rag_chain(self,llm,vectorstore):
        template = """
        You are Dr. Medic, a medical expert and assistant for question-answering tasks. Your goal is to provide accurate and helpful answers to the user's questions.

        Important Guidelines:

        Answer the question based on the provided context.
        If you don't know the answer, say so and don't try to make something up.
        If the user greets you with "hello" or "hi", respond with a friendly greeting and ask how you can help, don't go through context for messages like these.
        Keep your answers concise and focused on the question.
        Avoid discussing the context or providing unnecessary information.
        Use plain text, bold text, and bullets as needed to make your answers clear and easy to read.
        Do not use code blocks or ``` tags.

        Context: {context}
        Question: {question}
        Answer:
        """
        rag_prompt = PromptTemplate.from_template(template)

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | rag_prompt
            | llm
            | StrOutputParser()
        )
        return rag_chain
    
    def enter_qa(self, rag_chain):
        while True:
            try:
                print("About to invoke the rag_chain")
                question = input("Enter your prompt (type 'exit' to exit the loop): ")
                if question.lower() == "exit":
                    break
                for chunk in rag_chain.stream(question):
                    print(chunk, end="", flush=True)
                print("\nJust finished invoking the rag_chain")
            except Exception as e:
                print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    medic_ai = MedicAI('..//Data/')
    # embeddings = medic_ai.create_embeddings()
    # print(len(embeddings))
    api_key = os.getenv('PINECONE_API_KEY')
    pc = medic_ai.setup_pinecone(api_key=api_key)
    print(type(pc))
    # new_index = medic_ai.create_pinecone_index(api_key=api_key)
    # print(type(new_index)) # Index already exists.
    vectorstore = medic_ai.get_docsearch_from_pinecone(index_name="medicoai", embeddings=medic_ai.embeddings)
    # print(vectorstore.similarity_search("What is acne ?"))
    llm = medic_ai.setup_local_gemma()
    rag_chain = medic_ai.setup_rag_chain(llm=llm[1], vectorstore=vectorstore)
    medic_ai.enter_qa(rag_chain=rag_chain)