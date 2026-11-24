import re
from uuid import uuid4


def split_sentences(text):
    text = re.sub(r'(?m)^\s*\d+\.\s+', '', text)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def sentence_aware_chunking(
    text,
    document_id,
    filename,
    chunk_size=500,
    overlap=0
):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    sentences = split_sentences(text)

    chunks = []
    current_sentences = []
    current_word_count = 0
    chunk_index = 0

    for sentence in sentences:
        sentence_word_count = len(sentence.split())

        if (
            current_word_count + sentence_word_count > chunk_size
            and current_sentences
        ):
            chunk_text = " ".join(current_sentences)

            chunks.append({
                "chunk_id": str(uuid4()),
                "document_id": document_id,
                "chunk_index": chunk_index,
                "filename": filename,
                "text": chunk_text,
                "word_count": len(chunk_text.split())
            })

            chunk_index += 1

            current_sentences = (
                current_sentences[-overlap:]
                if overlap > 0 else []
            )

            current_word_count = sum(
                len(sentence.split())
                for sentence in current_sentences
            )

        current_sentences.append(sentence)
        current_word_count += sentence_word_count

    if current_sentences:
        chunk_text = " ".join(current_sentences)

        chunks.append({
            "chunk_id": str(uuid4()),
            "document_id": document_id,
            "chunk_index": chunk_index,
            "filename": filename,
            "text": chunk_text,
            "word_count": len(chunk_text.split())
        })

    return chunks
