import asyncio
import aiohttp

async def test_compare():
    async with aiohttp.ClientSession() as session:
        url = "http://localhost:8000/api/v1/compare"
        payload = {
            "question": "What is RAG?",
            "user_answer": "Retrieval Augmented Generation is a technique to improve LLMs.",
            "language": "en"
        }
        async with session.post(url, json=payload) as resp:
            print("Status:", resp.status)
            print("Response:", await resp.json())

asyncio.run(test_compare())
