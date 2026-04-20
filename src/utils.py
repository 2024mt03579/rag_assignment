from openai import OpenAI

client = OpenAI()

def call_llm(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

def detect_intent(query):
    prompt = f"""
    Classify this query into one of:
    QA, SUMMARIZE, QUIZ, COMPLIANCE

    Query: {query}

    Return only one word.
    """

    result = call_llm(prompt).strip().upper()
    return result