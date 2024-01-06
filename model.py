from langchain_community.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings

def load_llm(model_type = "llama", model_id = "TheBloke/Llama-2-7B-Chat-GGML"):
    llm = CTransformers(
        model=model_id,
        model_type = model_type,
        max_new_tokens = 512,
        temperature = 0.5
    )

    return llm

def get_embedding_model(model_name='sentence-transformers/all-MiniLM-L6-v2'):
    return HuggingFaceEmbeddings(model_name=model_name,
                                 model_kwargs={'device':'cpu'})