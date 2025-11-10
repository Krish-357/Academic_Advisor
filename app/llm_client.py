import os
import httpx
import asyncio
from dotenv import load_dotenv

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

# Default Gemini API Key (used if .env not loaded)
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAcMI2kxmZfufVVjX3-By1hDb3i8KKPkk8")


class LLMClient:
    def __init__(self):
        """Select provider: Gemini if key available, else mock."""
        if GEMINI_KEY and GEMINI_KEY.strip():
            self.provider = "gemini"
        else:
            self.provider = "mock"

    async def complete_system_prompt(self, agent_role, user_text, memories, context):
        """Generate a response using Gemini (or mock if key missing)."""
        system_prompt = self._compose_prompt(agent_role, user_text, memories, context)

        if self.provider == "gemini":
            return await self._call_gemini(system_prompt)
        else:
            return self._mock_response(agent_role, user_text, memories)

    # -----------------------
    # Prompt Builder
    # -----------------------
    def _compose_prompt(self, role, message, memories, context):
        mem_text = "\n".join(memories) if memories else "No prior memories."
        if role == "academic_advisor":
            return (
                f"You are an academic advisor.\n"
                f"Student context:\n{mem_text}\n\n"
                f"User question: {message}\n\n"
                f"Provide detailed guidance on suitable courses, majors, study plans, "
                f"and online learning resources."
            )
        elif role == "career_counselor":
            return (
                f"You are a career counselor.\n"
                f"Student context:\n{mem_text}\n\n"
                f"User question: {message}\n\n"
                f"Provide detailed insights on career paths, in-demand skills, "
                f"job roles, and practical steps for career growth."
            )
        return message

    # -----------------------
    # Gemini API (Google AI)
    # -----------------------
    async def _call_gemini(self, prompt_text):
        """Call Gemini 2.0 Flash with retry/backoff to handle rate limits."""
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.0-flash:generateContent?key={GEMINI_KEY}"
        )

        headers = {"Content-Type": "application/json"}
        body = {
            "contents": [
                {"role": "user", "parts": [{"text": prompt_text}]}
            ]
        }

        async with httpx.AsyncClient(timeout=60) as client:
            for attempt in range(3):  # retry up to 3 times
                try:
                    response = await client.post(url, headers=headers, json=body)

                    if response.status_code == 200:
                        data = response.json()
                        return (
                            data.get("candidates", [{}])[0]
                            .get("content", {})
                            .get("parts", [{}])[0]
                            .get("text", "No response text from Gemini.")
                        )

                    # If rate limit (429), wait & retry
                    if response.status_code == 429:
                        wait_time = 2 ** attempt  # exponential backoff (1s, 2s, 4s)
                        print(f"[Warning] Gemini rate limit hit. Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue

                    return f"[Gemini API Error {response.status_code}] {response.text}"

                except Exception as e:
                    return f"[Gemini Request Error: {e}]"

            # If all retries fail
            return (
                "Gemini service is currently busy. "
                "Please try again in a few seconds."
            )

    # -----------------------
    # Mock Mode (no API key)
    # -----------------------
    def _mock_response(self, role, text, memories):
        if role == "academic_advisor":
            return f"(mock) Academic advice for: {text}\nBased on memories: {memories}"
        return f"(mock) Career advice for: {text}\nBased on memories: {memories}"
