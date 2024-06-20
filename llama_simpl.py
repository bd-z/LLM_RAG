# Created on Fri May 24 20:32:35 2024
# @author: zhang
import pandas as pd
import os
import openai
from openai import OpenAI
import faiss
import numpy as np

def read_excel_files(folder_path):
    all_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            file_path = os.path.join(folder_path, filename)
            data = pd.read_excel(file_path)
            all_data.append(data)
    return all_data

folder_path = "D:\zbd\Llama"
excel_data = read_excel_files(folder_path)
# format and embedding
data_texts = [df.to_string(index=False) for df in excel_data]

client = OpenAI(
  api_key=os.environ.get("OPENAI_API_KEY"),
)

def get_text_embeddings(text):
    response = client.embeddings.create(
        input=text,
        model ="text-embedding-ada-002"  # choose embedding model
    )
    return response.data[0].embedding 

embeddings = [get_text_embeddings(text) for text in data_texts]

# convert embeddings to NP array which is needed by FAISS
embedding_dim = len(embeddings[0])
index = faiss.IndexFlatL2(embedding_dim)  # contruct vector index
index.add(np.array(embeddings).astype(np.float32))  # add embeddings vector to index

# save index for future use
faiss.write_index(index, 'faiss_index.index')

def query_knowledge_base(query, top_k=1):
    query_embedding = np.array(get_text_embeddings(query)).astype(np.float32)
    D, I = index.search(query_embedding.reshape(1, -1), top_k)
    return I[0], D[0]

def retrieve_relevant_data(indices, data_texts):
    return [data_texts[i] for i in indices]

def query_language_model(prompt):
    response = client.chat.completions.create(    
        model="gpt-3.5-turbo",        
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

# user query
user_query = "what i the PsreisstufeNummer of ProduktNummer 11477"
indices, distances = query_knowledge_base(user_query)
relevant_data = retrieve_relevant_data(indices, data_texts)
# combine with LLM to generate answer
combined_prompt = f"Here is the data related to your query:\n{relevant_data}\n\nBased on this data, {user_query}"
response = query_language_model(combined_prompt)
print(f"Response from the language model: {response}")

user_query = "what is the Preis of ProduktNummer 11477?"
user_query = "which ProduktNummers have Preis 12?"
user_query = "do you know Transdev Vertrieb GmbH"
print(user_query)
#user_query = "what is the filename that I can find ProduktNummers and Preis info?"