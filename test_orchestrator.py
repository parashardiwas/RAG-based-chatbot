import asyncio
from app.core.orchestrator import get_orchestrator
from app.config import get_settings

async def main():
    s = get_settings()
    print("Provider config:", s.llm_provider)
    orch = await get_orchestrator()
    res = await orch.process_text_query("What does the document intro say regarding topics XYZ?", language="en")
    print("Model used:", res.model_used)
    print("Answer:", res.answer[:100])

asyncio.run(main())
