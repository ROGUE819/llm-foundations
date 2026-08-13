import asyncio
import json
from typing import Literal
from datetime import UTC, datetime
from pathlib import Path
from rich.console import Console

from llm_foundations.client import LLMClient
from llm_foundations.config import get_settings
from llm_foundations.ledger import SessionLedger
from llm_foundations.pricing import cost_for
from llm_foundations.types import Message, TextDelta


class Conversation:
    def __init__(self, system: str | None = None) -> None:
        self._system = system
        self._turns: list[Message] = []

    def add(self, role: Literal["user", "assistant"], content: str) -> None:
        self._turns.append(Message(role=role, content=content))

    def to_messages(self) -> list[Message]:
        if self._system is None:
            return list(self._turns)
        return [Message(role="system", content=self._system), *self._turns]

    def reset(self) -> None:
        self._turns.clear()

    def set_system(self, system: str) -> None:
        self._system = system
        self._turns.clear()

    def drop_last(self) -> None:
        if self._turns:
            self._turns.pop()


def _handle_command(
    line: str,
    conv: Conversation,
    ledger: SessionLedger,
    console: Console,
) -> bool:
    """Returns True if the loop should exit."""
    parts = line.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else None

    if cmd in ("/quit", "/exit"):
        return True

    if cmd == "/help":
        console.print(
            "[dim]/reset  /system <path>  /cost  /tokens  /save  /quit[/dim]"
        )

    elif cmd == "/reset":
        conv.reset()
        console.print("[dim]conversation cleared[/dim]")

    elif cmd == "/system":
        if arg is None:
            console.print("[red]usage: /system <path>[/red]")
        else:
            path = Path(arg)
            if not path.is_file():
                console.print(f"[red]no such file: {path}[/red]")
            else:
                conv.set_system(path.read_text(encoding="utf-8").strip())
                console.print(f"[dim]system prompt loaded from {path}[/dim]")

    elif cmd == "/cost":
        console.print(f"[dim]{ledger.summary()}[/dim]")

    elif cmd == "/tokens":
        chars = sum(len(m.content) for m in conv.to_messages())
        console.print(f"[dim]~{chars // 4:,} tokens in context (estimate)[/dim]")

    elif cmd == "/save":
        Path("sessions").mkdir(exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out = Path("sessions") / f"{stamp}.json"
        out.write_text(
            json.dumps(
                {
                    "messages": [m.model_dump() for m in conv.to_messages()],
                    "ledger": ledger.model_dump(mode="json"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        console.print(f"[dim]saved to {out}[/dim]")

    else:
        console.print(f"[red]unknown command: {cmd}[/red] — try /help")

    return False

async def _run() -> None:
    print("loop started")
    settings = get_settings()
    client = LLMClient(settings.provider(), settings.llm_max_tokens, settings.llm_temperature)
    conv = Conversation()
    ledger = SessionLedger()
    console = Console()

    try:
        while True:
            try:
                line = input("› ").strip()
            except (KeyboardInterrupt, EOFError):
                break

            if not line:
                continue
            if line.startswith("/"):
                if _handle_command(line, conv, ledger, console):
                    break
                continue

            conv.add("user", line)   
            try:  
                async for event in client.stream(conv.to_messages()):
                    if isinstance(event, TextDelta):
                        console.print(event.text, end="")
                    else:
                        if event.content:
                            conv.add("assistant", event.content)
                            cost = (
                                cost_for(event.usage, event.model, free_tier=settings.provider().is_free_tier)
                                if event.usage else None
                            )
                            ledger.record(event, cost)
                            usage_str = (
                                f"{event.usage.input_tokens} in / {event.usage.output_tokens} out"
                                if event.usage else "no usage"
                            )
                            cost_str = f"${cost.shadow_usd:.4f}" if cost else "unpriced"
                            console.print(f"\n[dim]{usage_str} · {cost_str} · {event.latency_ms:.0f}ms[/dim]\n")
            except (KeyboardInterrupt, asyncio.CancelledError):
                conv.drop_last()
                console.print("\n[dim]interrupted[/dim]")
                continue
                        
    finally:
        console.print(f"\n[dim]Session: {ledger.summary()}[/dim]")

def main() -> None:
    asyncio.run(_run())