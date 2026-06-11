import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

async def main():
    engine = create_async_engine("postgresql+asyncpg://postgres:postgres@localhost:5432/ragbot")
    async with engine.connect() as conn:
        result = await conn.execute(text("SELECT content FROM document_chunks WHERE content ILIKE '%laws of motion%';"))
        rows = result.fetchall()
        for row in rows:
            print("--- CHUNK ---")
            print(row[0][:500])

asyncio.run(main())
