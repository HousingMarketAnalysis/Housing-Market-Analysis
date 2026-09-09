import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


def Chatbot():
    file_path = "./Dataset/manipulated dataset 2.xlsx"
    # Use 2nd row as column headers
    df = pd.read_excel(file_path, header=1)

    # Step 2: Convert rows into text (only use 2nd row headers and values)
    documents = []
    for i, row in df.iterrows():
        text_parts = [f"{col}: {row[col]}" for col in df.columns]
        text = " | ".join(text_parts)
        documents.append(text)

    # Step 3: Create embeddings with SentenceTransformers
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(documents, convert_to_numpy=True)

    # Step 4: Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Step 5: Function to answer questions
    def chatbot(query, top_k=1):
        query_vec = model.encode([query], convert_to_numpy=True)
        distances, indices = index.search(query_vec, top_k)

        results = [documents[i] for i in indices[0]]
        answer = "\n".join(results)
        return answer

    # Step 6: Chat loop
    print("Chatbot ready! Ask anything about your Excel data. Type 'exit' to quit.\n")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break
        response = chatbot(query)
        print("Chatbot:\n", response, "\n")


if __name__ == '__main__':
    Chatbot()
