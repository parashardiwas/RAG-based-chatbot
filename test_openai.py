import asyncio
from app.services.llm.generator import LLMGenerator

async def main():
    gen = LLMGenerator()
    try:
        res = await gen.generate("What is the capital of France?", "France is a country in Europe. Its capital is Paris.", "en")
        print("Success:", res.answer)
    except Exception as e:
        print("Exception:", type(e).__name__, str(e))

asyncio.run(main())
