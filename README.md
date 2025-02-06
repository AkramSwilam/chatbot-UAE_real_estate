# Chatbot Documentation

## Overview
This chatbot is designed to help users search for real estate properties in the UAE. It leverages a dataset of property listings, processes the data into searchable chunks, stores them in a vector database, and uses Azure OpenAI for generating responses. The chatbot is accessible through a FastAPI endpoint and a Streamlit-based user interface.

## Setup Instructions

### Prerequisites
- Python 3.8+
- Required Python libraries:
  - `pandas`
  - `fastapi`
  - `pydantic`
  - `langchain`
  - `langchain_openai`
  - `langchain_chroma`
  - `streamlit`
  - `requests`
  - `uuid`
  - `os`
  - `time`

### Environment Variables
Set the following environment variables before running the chatbot:
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_KEY`
- `AZURE_OPENAI_API_VERSION`
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`
- `AZURE_OPENAI_DEPLOYMENT_NAME`

### Installation
1. Clone the repository or download the script.
2. Install the required dependencies
3. Ensure the dataset (`uae_real_estate_2024.csv`) is available in the same directory.
4. Run the FastAPI backend:
   ```bash
   uvicorn chatbot:app --host 0.0.0.0 --port 8000
   ```
5. Start the Streamlit frontend:
   ```bash
   streamlit run streamlit.py
   ```

## Functionality

### Data Processing
1. Loads the dataset (`uae_real_estate_2024.csv`) using Pandas.
2. Fills missing values with empty strings.
3. Formats each property listing into a structured text format.
4. Splits text into smaller chunks using `RecursiveCharacterTextSplitter`.
5. Stores processed property chunks into a `Chroma` vector database.

### Vector Storage
- Uses `AzureOpenAIEmbeddings` to generate embeddings for property data.
- Stores these embeddings in `Chroma` for efficient similarity searches.
- Documents are added in batches to prevent memory issues.

### Chatbot Interaction
1. Users enter a search query in the Streamlit interface.
2. The FastAPI backend processes the query:
   - Searches for similar property listings in `Chroma`.
   - Uses GPT-4o (`AzureChatOpenAI`) to summarize and return a response.
3. The response is displayed to the user.

## API Endpoints
### `POST /search`
#### Request Body
```json
{
  "query": "3-bedroom apartment in Dubai Marina"
}
```
#### Response
```json
{
  "response": "Here are some properties that match your criteria: ..."
}
```

## User Interface (Streamlit)
- A simple UI allows users to enter a search query.
- Sends requests to the FastAPI backend.
- Displays property search results in a readable format.

## Error Handling
- Handles missing API keys and environment variables.
- Catches indexing errors when adding documents to `Chroma`.
- Implements retry logic for batch document addition.
- Provides user-friendly warnings in Streamlit.

## Future Enhancements
- Implement authentication for API access.
- Improve property ranking using more advanced search techniques.
- Add support for additional query filters (e.g., price range, location).
- Deploy backend on a cloud server for scalability.

## Conclusion
This chatbot effectively allows users to search for real estate properties using natural language. By leveraging `LangChain`, `Azure OpenAI`, and `Chroma`, it provides efficient search and response generation. The FastAPI backend ensures smooth processing, while the Streamlit UI makes the chatbot user-friendly.

