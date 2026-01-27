import asyncio
import json
from typing import Any, Dict, List


class EventBroker:
    def __init__(self):
        self.subscribers: List[asyncio.Queue] = []
        self.lock = asyncio.Lock()

    async def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        async with self.lock:
            self.subscribers.append(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue):
        async with self.lock:
            if queue in self.subscribers:
                self.subscribers.remove(queue)

    async def publish(self, event: Dict[str, Any]):
        async with self.lock:
            for queue in list(self.subscribers):
                await queue.put(event)

    @staticmethod
    def format_sse(data: Dict[str, Any]) -> str:
        return f"data: {json.dumps(data)}\n\n"


event_broker = EventBroker()
