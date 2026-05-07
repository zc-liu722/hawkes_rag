from __future__ import annotations

import os

import _bootstrap  # noqa: F401
from hawkes_rag import HawkesMemoryStore


def simple_embedding(text: str) -> list[float]:
    """Tiny deterministic embedding for the example fallback.

    Replace this with sentence-transformers, OpenAI embeddings, or BGE in real
    use. Keeping the fallback local makes the example runnable in a fresh clone.
    """
    text = text.lower()
    return [
        float("max" in text or "dog" in text or "park" in text),
        float("python" in text or "code" in text or "package" in text),
        float("paper" in text or "hawkes" in text or "memory" in text),
    ]


def main() -> None:
    store = HawkesMemoryStore(beta=0.4)
    store.add("The user's dog is named Max.", simple_embedding("dog Max"))
    store.add("The user is writing a Hawkes-RAG paper.", simple_embedding("Hawkes memory paper"))
    store.add("The user mentioned Python packaging once.", simple_embedding("Python package"))

    user_message = "What should I remember about Max this weekend?"
    retrieved = store.retrieve(simple_embedding(user_message), top_k=2)
    context = "\n".join(f"- {r.memory.content}" for r in retrieved)

    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            messages=[
                {"role": "system", "content": f"Use these memories when relevant:\n{context}"},
                {"role": "user", "content": user_message},
            ],
        )
        print(response.choices[0].message.content)
    elif os.getenv("ANTHROPIC_API_KEY"):
        import anthropic

        client = anthropic.Anthropic()
        response = client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest"),
            max_tokens=300,
            system=f"Use these memories when relevant:\n{context}",
            messages=[{"role": "user", "content": user_message}],
        )
        print(response.content[0].text)
    else:
        print("No API key found. Retrieved context:")
        print(context)


if __name__ == "__main__":
    main()
