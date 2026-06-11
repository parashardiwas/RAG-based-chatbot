import asyncio
from app.db.database import async_session_maker
from app.db.models import QAPair
from sqlalchemy import select

async def main():
    async with async_session_maker() as session:
        # Get one QA pair
        res = await session.execute(select(QAPair).limit(1))
        qa = res.scalar_one_or_none()
        if not qa:
            print("No QA pair found")
            return
        
        chunk_id_str = str(qa.id)
        
        # Now try to query it by string
        try:
            stmt = select(QAPair).where(QAPair.id == chunk_id_str)
            res2 = await session.execute(stmt)
            qa2 = res2.scalar_one_or_none()
            print("Query by string returned:", bool(qa2))
        except Exception as e:
            print("Error:", e)

asyncio.run(main())
