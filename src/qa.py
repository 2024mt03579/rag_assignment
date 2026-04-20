import logging

from openai import OpenAI

client = OpenAI()


def answer_with_rag(db, query):
    docs = db.similarity_search(query, k=4)
    #print("DEBUG DOC COUNT:", len(docs))

    #for d in docs:
        #print("TEXT:", d.page_content[:100])
        #print("META:", d.metadata)

    if not docs:
        return "No relevant information found in documents.", []

    context = "\n\n".join([doc.page_content for doc in docs])

    sources = [doc.metadata.get("source", "Unknown") for doc in docs]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You MUST answer ONLY from the provided context. "
                    "If the answer is not in the context, say: 'Not found in documents.'"
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{query}"
            }
        ],
        temperature=0
    )

    usage = response.usage
    # Log token usage
    logging.info(f"Prompt tokens: {usage.prompt_tokens}")
    logging.info(f"Completion tokens: {usage.completion_tokens}")
    logging.info(f"Total tokens: {usage.total_tokens}")

    # Log sources
    logging.info(f"Sources used: {list(set(sources))}")

    answer = response.choices[0].message.content

    return answer, list(set(sources))