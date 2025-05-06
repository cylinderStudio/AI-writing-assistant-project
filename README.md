
![genai-assistant-ui](https://github.com/user-attachments/assets/b2955ece-2557-4810-93bd-0e1f85cc0fa7)

# AI Writing Assistant Project

This project is an experiment to create a generative AI assistant that could help a company’s marketing, advertising, or PR teams quickly generate documents that conform to style, tone, and brand guidelines.

**This repo contains the Python scripts used to:**
1. Create training data
2. Fine-tune a base model with training data and save as GGUF
3. Clean, chunk and upsert data for RAG vector database
4. Create a UI with inference logic
    - `app.py`: model inference only
    - `app-plus-rag.py`: inference + vector database/RAG

**Model Training**
Fine-tuning was performed on Meta Llama 3.1 8B Instruct, using Google Colab and the [Unsloth](https://github.com/unslothai/unsloth) framework.

**RAG**
Retrieval Augmented Generation is provided with [Pinecone](https://www.pinecone.io/).

**Inference**
Fine-tuned model is stored on Hugging Face and uses [Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index). The user interface was built with [Gradio](https://www.gradio.app/).
