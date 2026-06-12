import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
import os

DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/ragbot"
engine = create_async_engine(DATABASE_URL)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession)

async def check():
    async with AsyncSessionLocal() as session:
        result = await session.execute(text("SELECT id, question FROM qa_pairs"))
        for row in result:
            print(row.id, repr(row.question))

asyncio.run(check())
