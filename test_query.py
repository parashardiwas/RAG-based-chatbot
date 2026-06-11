import asyncio
from app.core.orchestrator import Orchestrator
from app.db.database import async_session_maker

async def main():
    orch = Orchestrator()
    await orch.init(async_session_maker)
    res = await orch.process_text_query("What is the capital of France?")
    print(res)

asyncio.run(main())
