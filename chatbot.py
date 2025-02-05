import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Load dataset
df = pd.read_csv("uae_real_estate_2024.csv")

# Fill missing values
df.fillna("", inplace=True)

def format_property_text(row):
    return (f"{row['title']}. Located in {row['displayAddress']}. "
            f"{row['bedrooms']} bedrooms, {row['bathrooms']} bathrooms. "
            f"Price: {row['price']} AED. {row['furnishing']} furnished. "
            f"Type: {row['type']}. {row['description']}")

df["embedding_text"] = df.apply(format_property_text, axis=1)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

property_chunks = []
for index, row in df.iterrows():
    chunks = text_splitter.split_text(row["embedding_text"])
    for chunk in chunks:
        property_chunks.append(Document(page_content=chunk, metadata={"id": index}))

print(f"Generated {len(property_chunks)} chunks from {len(df)} properties.")

from langchain.vectorstores import Chroma

api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

print(api_base)
print(api_key)
print(api_version)
print(embedding_deployment)

from langchain_openai import AzureOpenAIEmbeddings
embedding_model = AzureOpenAIEmbeddings(
    model=embedding_deployment,
    azure_endpoint=api_base,
    api_key=api_key,
    openai_api_version=api_version
)

from langchain_chroma import Chroma
from uuid import uuid4
CHROMA_PATH = r"chroma_db"
vector_store = Chroma(
    collection_name="uae_real_estate",
    embedding_function=embedding_model,
    persist_directory=CHROMA_PATH,
)

uuids = [str(uuid4()) for _ in range(len(property_chunks))]

# Function to add documents in batches
def add_documents_in_batches(vector_store, documents, ids, batch_size=100):
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        try:
            vector_store.add_documents(documents=batch_docs, ids=batch_ids)
            print(f"Batch {i // batch_size + 1} indexed successfully!")
        except Exception as e:
            print(f"Failed to index batch {i // batch_size + 1}: {e}")
            time.sleep(60)  # Wait before retrying

# Add documents to the vector store in batches
add_documents_in_batches(vector_store, property_chunks, uuids)

from langchain_openai import AzureChatOpenAI
model = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
chat_model = AzureChatOpenAI(
    azure_deployment=model,  
    api_version=api_version,
    api_key=api_key,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

def search_properties(query):
    results = vector_store.similarity_search(query, k=5)
    return results

def generate_response(query):
    results = search_properties(query)

    if not results:
        return "I couldn't find any properties matching your criteria."

    # Format results for GPT-4o
    property_info = "\n".join([f"- {r.page_content}" for r in results])

    # Ask GPT-4o to summarize
    prompt = f"Based on the user query '{query}', here are some matching properties:\n{property_info}\nSummarize professionally."
    response = chat_model.invoke(prompt)

    return response
