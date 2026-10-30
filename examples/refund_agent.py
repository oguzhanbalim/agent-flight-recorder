"""A minimal agent loop with two synthetic tools. It takes any Messages-API-shaped client."""

from __future__ import annotations

import json
from typing import Any

MODEL = "claude-opus-5-5"
TOOLS = [
    {
        "name": "lookup_order",
        "description": "Look up an order by id.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "issue_refund",
        "description": "Refund an order.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}, "amount": {"type": "number"}},
            "required": ["order_id", "amount"],
        },
    },
]


def run_tool(name: str, tool_input: dict[str, Any]) -> str:
    if name == "lookup_order":
        return json.dumps({"order_id": tool_input["order_id"], "total": 40, "status": "delivered"})
    if name == "issue_refund":
        return json.dumps({"refunded": tool_input["amount"]})
    return json.dumps({"error": f"unknown tool {name}"})


def run_agent(client: Any, tool_runner: Any = run_tool, max_steps: int = 10) -> Any:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Order ORD-1042 arrived broken. Please refund it."}
    ]
    for _ in range(max_steps):
        response = client.messages.create(
            model=MODEL, max_tokens=2048, tools=TOOLS, messages=messages
        )
        if response.stop_reason != "tool_use":
            return response
        messages.append({"role": "assistant", "content": response.content})
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": tool_runner(block.name, block.input),
                    }
                    for block in response.content
                    if block.type == "tool_use"
                ],
            }
        )
    raise RuntimeError("agent did not finish")
