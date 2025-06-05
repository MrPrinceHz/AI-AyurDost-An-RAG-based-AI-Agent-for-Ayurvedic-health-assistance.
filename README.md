# AI AyurDost

AI AyurDost is a chatbot powered by LangChain, FAISS, and HuggingFace models. It provides AI-driven responses based on context-aware retrieval, leveraging a vector store for efficient question-answering.

## Features
- **Conversational Memory**: Maintains chat history for better context-aware responses.
- **FAISS Vector Store**: Efficient document retrieval for accurate answers.
- **HuggingFace LLM**: Utilizes `mistralai/Mistral-7B-Instruct-v0.3` for response generation.
- **Streamlit UI**: Provides an interactive interface for users.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/MrPrinceHz/AI-AyurDost-An-RAG-based-AI-Agent-for-Ayurvedic-health-assistance.
   cd AI-AyurDost-An-RAG-based-AI-Agent-for-Ayurvedic-health-assistance
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```sh
   export HF_TOKEN=your_huggingface_token
   ```

## Usage
Run the Streamlit app:
```sh
streamlit run app.py
```

## Project Structure
```
├── app.py                # Main application script
├── vectorstore/          # FAISS vector database
├── requirements.txt      # Required dependencies
├── README.md             # Documentation
```

## How It Works
1. **FAISS Vector Store** loads pre-processed knowledge for retrieval.
2. **LangChain ConversationalRetrievalChain** retrieves relevant context.
3. **HuggingFace LLM** generates context-aware responses.
4. **Streamlit Interface** provides user interaction.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss.

## License
This project is licensed under the MIT License.

