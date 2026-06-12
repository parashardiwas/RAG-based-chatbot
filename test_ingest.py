import asyncio
from app.db.database import async_session_maker, init_db
from app.core.orchestrator import get_orchestrator
import os

async def main():
    await init_db()
    orc = await get_orchestrator()
    await orc.init(async_session_maker, None)
    
    with open("test.txt", "w") as f:
        f.write("This is a test document. It has some sentences. Hopefully it ingests.")
        
    res = await orc.ingest_document("test.txt", "test.txt", "text/plain")
    print("Ingestion result:", res)

asyncio.run(main())
