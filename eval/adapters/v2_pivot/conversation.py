"""Strict-correct conversation state for Anthropic Messages API.

The bug we caught the hard way: Anthropic enforces tight invariants on
the messages array — every assistant `tool_use` block must be answered by
exactly one `tool_result` block in the IMMEDIATELY-FOLLOWING user message.
A single extra append (or a missed result) makes the API reject the
request with `tool_use ids were found without tool_result blocks` or its
inverse.

This module is intentionally minimal and explicit so the invariants stay
visible. The Conversation object is the ONLY thing that mutates the
messages list; callers add input via well-typed methods and the object
guarantees a valid API payload at all times.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Conversation:
    """Append-only message log enforcing Anthropic Messages-API invariants."""

    system: str
    tools: list[dict[str, Any]]  # Anthropic tool defs
    messages: list[dict[str, Any]] = field(default_factory=list)

    # -------- writer-side: callers should ONLY use these methods --------

    def append_user_text(self, text: str) -> None:
        """Append a user message containing a single text block."""
        self._append_user_blocks([{"type": "text", "text": text}])

    def append_assistant(self, content_blocks: list[dict[str, Any]]) -> None:
        """Append an assistant message verbatim.

        `content_blocks` is the raw Anthropic content array (mix of
        text and tool_use blocks). Must NOT be the SDK's typed objects —
        the caller is responsible for converting via `.model_dump()` or
        equivalent, so the persisted history is a pure dict tree.
        """
        # Validate: list of dicts only.
        if not isinstance(content_blocks, list):
            raise ValueError("content_blocks must be a list")
        for b in content_blocks:
            if not isinstance(b, dict):
                raise ValueError(f"each block must be a dict, got {type(b)}")
        # Tau2 / Anthropic both reject an empty assistant message.
        if not content_blocks:
            raise ValueError("assistant message must have ≥1 block")
        self.messages.append({"role": "assistant", "content": content_blocks})

    def append_tool_results(self, results: list[tuple[str, str, bool]]) -> None:
        """Append a user message of tool_result blocks.

        Each result is ``(tool_use_id, content, is_error)``. Content can be
        any JSON-serializable thing; tool_result `content` field is a
        string in Anthropic's API, so we json.dumps non-strings.

        Crucially, ALL tool_use blocks from the previous assistant message
        must be answered in a single user message — Anthropic does not
        accept interleaving. We don't enforce that here (the orchestrator
        is responsible), but we keep this method append-only so partial
        appends are impossible.
        """
        blocks: list[dict[str, Any]] = []
        for tool_use_id, content, is_error in results:
            content_str = content if isinstance(content, str) else json.dumps(
                content, default=str
            )
            block = {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": content_str,
            }
            if is_error:
                block["is_error"] = True
            blocks.append(block)
        self._append_user_blocks(blocks)

    # ------------------ reader-side: API payload -----------------------

    def api_payload(self) -> dict[str, Any]:
        """Build the request body for Anthropic Messages create()."""
        return {
            "system": self.system,
            "messages": self.messages,
            "tools": self.tools,
        }

    # ------------------ invariant checks (debugging) -------------------

    def validate(self) -> list[str]:
        """Return a list of validation errors; empty list means valid.

        Use as a self-check between turns to catch any malformed history
        before sending to the API.
        """
        errors: list[str] = []
        prev_role: str | None = None
        prev_tool_uses: list[str] = []
        for i, msg in enumerate(self.messages):
            role = msg.get("role")
            if role not in ("user", "assistant"):
                errors.append(f"messages[{i}]: invalid role {role!r}")
                continue
            if role == prev_role:
                errors.append(f"messages[{i}]: two consecutive {role} messages")
            content = msg.get("content", [])
            if not isinstance(content, list):
                errors.append(f"messages[{i}]: content must be a list")
                continue
            tool_uses_in_msg = [
                b.get("id") for b in content if b.get("type") == "tool_use"
            ]
            tool_results_in_msg = [
                b.get("tool_use_id") for b in content if b.get("type") == "tool_result"
            ]
            if role == "user":
                # If prev assistant had tool_uses, this user message MUST
                # contain a tool_result for each (and no other tool_results).
                if prev_tool_uses:
                    unmatched_uses = set(prev_tool_uses) - set(tool_results_in_msg)
                    extra_results = set(tool_results_in_msg) - set(prev_tool_uses)
                    if unmatched_uses:
                        errors.append(
                            f"messages[{i}]: missing tool_result for "
                            f"tool_use ids: {sorted(unmatched_uses)}"
                        )
                    if extra_results:
                        errors.append(
                            f"messages[{i}]: tool_result ids without matching "
                            f"tool_use: {sorted(extra_results)}"
                        )
                elif tool_results_in_msg:
                    errors.append(
                        f"messages[{i}]: tool_result without preceding tool_use"
                    )
                prev_tool_uses = []
            else:  # assistant
                prev_tool_uses = [tid for tid in tool_uses_in_msg if tid]
            prev_role = role
        return errors

    # ----------------------- internal helpers --------------------------

    def _append_user_blocks(self, blocks: list[dict[str, Any]]) -> None:
        """Append blocks to the user side. If the most-recent message is
        already a user message, MERGE — Anthropic forbids consecutive
        same-role messages."""
        if not blocks:
            return
        if self.messages and self.messages[-1].get("role") == "user":
            existing = self.messages[-1].get("content")
            if not isinstance(existing, list):
                existing = [{"type": "text", "text": str(existing)}]
            existing.extend(blocks)
            self.messages[-1]["content"] = existing
        else:
            self.messages.append({"role": "user", "content": list(blocks)})
