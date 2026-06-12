import asyncio
from app.services.language.translator import TranslatorService

async def main():
    translator = TranslatorService()
    question = "what is RAG?"
    answer_a = "Retrieval Augmented Generation"
    answer_b = "RAG stands for Retrieval-Augmented Generation. It is an AI framework that improves the quality of LL..."
    
    prompt = f"""You are a semantic fact-checker. Your only job is to determine if two answers convey the same core facts.

RULES:
- Ignore all differences in phrasing, word order, verbosity, or style.
- Focus ONLY on whether the key factual claim(s) in Answer A are present in Answer B.
- A short, correct answer (e.g. "Paris") MUST match a longer one ("The capital is Paris.").
- Partial answers that contain the correct fact still count as a match.
- Output ONLY the single word YES or NO. No explanation, no punctuation.

Question: {question}
Answer A: {answer_a}
Answer B: {answer_b}

Are the core facts in Answer A and Answer B equivalent?"""

    client = translator._get_client()
    response = await client.chat.completions.create(
        model=translator._model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=5,
    )
    print("OUTPUT:", repr(response.choices[0].message.content))

asyncio.run(main())
