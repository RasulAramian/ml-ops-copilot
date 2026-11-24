
def build_prompt(query, context):
    return f"""
You are an ML engineering assistant.

Answer the user's question using only the provided context.

Rules:
1. Do not introduce unsupported facts or assumptions.
2. If the context does not contain enough information,
   clearly state that you do not have enough information.
3. Cite the source filename when providing factual claims.
4. Do not invent citations.

Context:
{context}

Question:
{query}

Answer:
"""
