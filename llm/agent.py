"""Claude agent with episodic memory tools (remember / recall / sleep).

The agent decides what to store and when to look things up; the memory
service holds the text and the backend holds the index. Interactive use:

    python -m llm.agent --backend tfidf --db out/agent_memory.sqlite

Credentials come from the environment (ANTHROPIC_API_KEY or an
`ant auth login` profile). Type `/sleep [n]` to run consolidation between
"sessions", `/quit` to exit.
"""

from __future__ import annotations

import argparse

import anthropic
from anthropic import beta_tool

from .backends import BACKENDS
from .memory_service import MemoryService

MODEL = "claude-opus-5-5"

SYSTEM = """You are an assistant with a long-term episodic memory that persists across \
conversations. You do not remember earlier conversations on your own; anything you \
want to keep must be stored with the `remember` tool, and anything from the past must \
be looked up with `recall`.

Store one self-contained fact per `remember` call, written so it makes sense without \
the conversation (names, not pronouns). Store facts about the user and their world \
that they would expect you to know later; skip small talk.

When a question may depend on something said earlier, call `recall` with a short cue \
before answering. Recall results come with a confidence score; if nothing relevant \
comes back, say that you do not remember rather than guessing."""


def build_tools(service: MemoryService) -> list:
    @beta_tool
    def remember(fact: str) -> str:
        """Store one self-contained fact in long-term memory.

        Args:
            fact: The fact, phrased so it is understandable on its own.
        """
        mid = service.remember(fact)
        return f"stored as {mid}"

    @beta_tool
    def recall(cue: str, k: int = 3) -> str:
        """Look up memories matching a partial cue. Returns up to k memories, best first.

        Args:
            cue: A short description of what to look for, e.g. "Alice's dentist appointment".
            k: Maximum number of memories to return.
        """
        hits = service.recall(cue, k=max(1, min(int(k), 10)))
        if not hits:
            return "no matching memories"
        return "\n".join(f"[{h.memory_id}] (confidence {h.score:.2f}) {h.text}" for h in hits)

    @beta_tool
    def sleep(n_blocks: int = 1) -> str:
        """Run offline consolidation of stored memories. Use only when the user asks you to rest or sleep.

        Args:
            n_blocks: Number of consolidation blocks to run.
        """
        service.sleep(max(1, int(n_blocks)))
        return f"slept {n_blocks} block(s); total {service.sleep_blocks}"

    return [remember, recall, sleep]


class MemoryAgent:
    def __init__(self, service: MemoryService, client: anthropic.Anthropic | None = None,
                 model: str = MODEL, effort: str = "medium") -> None:
        self.service = service
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.tools = build_tools(service)
        self.messages: list = []

    def new_session(self) -> None:
        """Start a fresh conversation: only the memory service carries over."""
        self.messages = []

    def send(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})
        runner = self.client.beta.messages.tool_runner(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM,
            tools=self.tools,
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=self.messages,
        )
        final = None
        for message in runner:
            final = message
            # mirror history: the runner keeps its own copy and does not expose it
            self.messages.append({"role": "assistant", "content": message.content})
            tool_response = runner.generate_tool_call_response()
            if tool_response is not None:
                self.messages.append(tool_response)
        if final is None:
            return ""
        if final.stop_reason == "refusal":
            return "[declined]"
        return "".join(b.text for b in final.content if b.type == "text")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=sorted(BACKENDS), default="tfidf")
    ap.add_argument("--db", default=":memory:", help="SQLite file for the content store")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    args = ap.parse_args()

    service = MemoryService(BACKENDS[args.backend](), db_path=args.db)
    agent = MemoryAgent(service, effort=args.effort)
    print(f"memory agent ({args.backend}, {service.count()} memories). /sleep [n], /new, /quit")
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line == "/quit":
            break
        if line == "/new":
            agent.new_session()
            print("(new session)")
            continue
        if line.startswith("/sleep"):
            parts = line.split()
            service.sleep(int(parts[1]) if len(parts) > 1 else 1)
            agent.new_session()
            print(f"(slept; total {service.sleep_blocks} blocks; new session)")
            continue
        print(agent.send(line))
    service.close()


if __name__ == "__main__":
    main()
