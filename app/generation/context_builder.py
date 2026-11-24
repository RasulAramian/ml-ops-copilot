
def build_context(ranked_results, top_k=3):
    selected_results = ranked_results[:top_k]

    context_parts = []

    for index, result in enumerate(selected_results, start=1):
        chunk = result["chunk"]

        source = (
            f"[Source {index}: {chunk['filename']}]\n"
            f"{chunk['text']}"
        )

        context_parts.append(source)

    return "\n\n".join(context_parts)
