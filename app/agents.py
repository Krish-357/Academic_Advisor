import asyncio
from app.llm_client import LLMClient
from app.vectordb import VectorDB

class AgentManager:
    def __init__(self):
        self.llm = LLMClient()
        self.vdb = VectorDB(storage_path='server/data/memory.json')

    async def handle_user_message(self, user_id: str, text: str, context: dict):
        # Retrieve top-k memories
        memories = self.vdb.retrieve(user_id, top_k=5)

        # Run two agents in parallel: academic advisor and career counselor
        tasks = [
            self.llm.complete_system_prompt('academic_advisor', text, memories, context),
            self.llm.complete_system_prompt('career_counselor', text, memories, context),
        ]
        results = await asyncio.gather(*tasks)

        # Upsert the new user message to memory
        try:
            self.vdb.upsert(user_id, text)
        except Exception:
            pass

        combined = {
            'academic': results[0],
            'career': results[1],
            'memories': memories
        }
        return combined
