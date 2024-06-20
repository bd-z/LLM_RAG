# LLM_RAG: Empowering GPT for Domain Questions

The workflow for integrating Retrieval-Augmented Generation with Large Language Models includes:

1. Preparing a knowledge base and generating word embeddings.
2. Setting up a retrieval system, processing queries, and integrating retrieved documents with context.
3. Deploying locally or via cloud APIs using Python code or GUI tools like Ollama ANYTHINGLLM.
4. Utilizing the LLM to generate and post-process responses.

Conclusion:
In this test, the Python version (GPT-3.5-turbo + OpenAI text-embedding-ada-002 + FAISS) outperforms the GUI version. For the GUI, using LanceDB as the vector database, GPT-3.5-turbo with OpenAI text-embedding-3-large surpasses Llama2 with nomic-embed-text.


![image](https://github.com/bd-z/LLM_RAG/assets/56706046/50d5d729-6485-433d-85b0-3ccdf38fe517)

![image](https://github.com/bd-z/LLM_RAG/assets/56706046/d2e2b85f-6654-46f6-8696-e262acd391ae)
