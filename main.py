"""
Customer Support AI Agent — Starter Code
==========================================
Your task is to complete this file by implementing all sections marked
with # TODO comments.

Reference the step-by-step solution files and INSTRUCTIONS.md for guidance.
Do NOT copy the solution directly — work through each section yourself.

Run locally (after filling in config values):
  uv run main.py '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'

Deploy to AgentCore:
  agentcore deploy

Invoke deployed agent:
  agentcore invoke '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'
"""

# ── Imports ───────────────────────────────────────────────────────────────────
# These imports are provided. Do not remove them.
from strands import Agent, tool
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.memory import MemoryClient
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamable_http_client
import argparse, json
import os, asyncio, boto3
from strands.hooks import (
    HookProvider, AfterInvocationEvent, HookRegistry, MessageAddedEvent,
)
import logging
import uuid
from typing import Dict
from bedrock_agentcore.tools.code_interpreter_client import code_session
from strands_tools.browser import AgentCoreBrowser


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("CSAI_Agent")

# ── TODO 1 — App Initialisation ───────────────────────────────────────────────
app = BedrockAgentCoreApp()

# Suppress interactive tool-consent prompts (required in headless deployments).
os.environ["BYPASS_TOOL_CONSENT"] = "true"


# ── TODO 2 — Configuration ────────────────────────────────────────────────────
GATEWAY_URL = "https://customersupportgateway-4imuuwmspl.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"
KB_ID       = "TCVMNXNMJT"
REGION      = "us-east-1"
MEMORY_ID   = "CustomerSupportMemory-cOVPUY4WKx"


# ── TODO 3 — Model and Clients ────────────────────────────────────────────────
model_id = "global.amazon.nova-2-lite-v1:0"

model = BedrockModel(model_id=model_id)
memory_client = MemoryClient(region_name=REGION)
_bedrock_runtime = boto3.client("bedrock-agent-runtime", region_name=REGION)


# ── TODO 4 — Namespace Helper ─────────────────────────────────────────────────
def get_namespaces(mem_client: MemoryClient, memory_id: str) -> Dict:
    """Return a dict mapping strategy type -> namespace template string."""
    strategies = mem_client.get_memory_strategies(memory_id)
    return {s["type"]: s["namespaces"][0] for s in strategies}


# ── TODO 5 — Memory Hook ──────────────────────────────────────────────────────
class MemoryHook(HookProvider):
    """Long-term memory hook for the customer support agent."""

    def __init__(
        self,
        actor_id: str,
        session_id: str,
        memory_client: MemoryClient,
        memory_id: str,
    ):
        self.actor_id = actor_id
        self.session_id = session_id
        self.memory_client = memory_client
        self.memory_id = memory_id
        self.namespaces = get_namespaces(memory_client, memory_id)

    @staticmethod
    def _is_tool_content(content: list) -> bool:
        return any("toolResult" in block or "toolUse" in block for block in content)

    @staticmethod
    def _extract_text(content: list) -> str:
        return " ".join(block["text"] for block in content if "text" in block).strip()

    def retrieve_customer_context(self, event: MessageAddedEvent):
        """Retrieve relevant memories and prepend them to the user message."""
        messages = event.agent.messages
        if not messages:
            return

        last_message = messages[-1]
        if last_message.get("role") != "user":
            return

        content = last_message.get("content", [])
        if not content or self._is_tool_content(content):
            return

        query = self._extract_text(content)
        if not query:
            return

        memories_by_type = {}
        for strategy_type, namespace_template in self.namespaces.items():
            namespace = namespace_template.format(actorId=self.actor_id)
            try:
                hits = self.memory_client.retrieve_memories(
                    memory_id=self.memory_id,
                    namespace=namespace,
                    query=query,
                    top_k=5,
                )
            except Exception as e:
                logger.warning("retrieve_memories failed for %s: %s", strategy_type, e)
                continue

            texts = []
            for hit in hits:
                text = hit.get("content", {}).get("text") or hit.get("text")
                if text:
                    texts.append(text)
            if texts:
                memories_by_type[strategy_type] = texts

        if not memories_by_type:
            return

        lines = [
            f"[{strategy_type}] {text}"
            for strategy_type, texts in memories_by_type.items()
            for text in texts
        ]
        context_block = "Customer Context:\n" + "\n".join(lines)
        last_message["content"] = [{"text": f"{context_block}\n\n{query}"}]

    def save_support_interaction(self, event: AfterInvocationEvent):
        """Save the completed turn to memory after the agent responds."""
        messages = event.agent.messages

        last_user_text, last_assistant_text = None, None
        for message in reversed(messages):
            role = message.get("role")
            content = message.get("content", [])
            text = self._extract_text(content)

            if role == "assistant" and text and last_assistant_text is None:
                last_assistant_text = text
            elif (
                role == "user"
                and text
                and not self._is_tool_content(content)
                and last_user_text is None
            ):
                last_user_text = text

            if last_user_text and last_assistant_text:
                break

        if not last_user_text or not last_assistant_text:
            return

        try:
            self.memory_client.create_event(
                memory_id=self.memory_id,
                actor_id=self.actor_id,
                session_id=self.session_id,
                messages=[
                    (last_user_text, "USER"),
                    (last_assistant_text, "ASSISTANT"),
                ],
            )
        except Exception as e:
            logger.warning("create_event failed: %s", e)

    def register_hooks(self, registry: HookRegistry) -> None:  # type: ignore
        """Register both memory callbacks."""
        registry.add_callback(MessageAddedEvent, self.retrieve_customer_context)
        registry.add_callback(AfterInvocationEvent, self.save_support_interaction)


# ── TODO 6 — Knowledge Base Tool ─────────────────────────────────────────────
@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the Amazon product catalog and support knowledge base.
    Use this for product specifications, return policies, warranty
    information, loyalty program details, and order status definitions.

    Args:
        query: The question or topic to search for

    Returns:
        Relevant information retrieved from the knowledge base
    """
    if not KB_ID or KB_ID.startswith("<"):
        return "Knowledge base not configured."

    try:
        resp = _bedrock_runtime.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": query},
        )
    except Exception as e:
        return f"Knowledge base lookup failed: {e}"

    results = resp.get("retrievalResults", [])
    if not results:
        return "No relevant information found in the knowledge base."

    chunks = [
        r["content"]["text"]
        for r in results
        if r.get("content", {}).get("text")
    ]
    return "\n---\n".join(chunks)


# ── TODO 7 — Loyalty Discount Tool (Code Interpreter) ────────────────────────
@tool
def calculate_loyalty_discount(
    loyalty_points: int,
    tier: str,
    order_total: float,
    product_category: str = "standard",
) -> str:
    """
    Calculate the loyalty discount for a customer order using the
    AgentCore Code Interpreter. Runs exact arithmetic in a secure sandbox.

    Args:
        loyalty_points:   Customer's current points balance
        tier:             Customer tier — Silver, Gold, or Platinum
        order_total:      Order total in USD
        product_category: standard, device, or fresh

    Returns:
        Full discount breakdown and final price
    """
    code = f"""
import json

earn_rates = {{"standard": 1, "device": 2, "fresh": 5}}
tier_rates = {{"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}}

loyalty_points = {loyalty_points}
tier = {tier!r}
order_total = {order_total}
product_category = {product_category!r}

tier_discount_pct = tier_rates.get(tier, 0.0)

POINTS_PER_DOLLAR = 100
REDEMPTION_STEP = 500

max_value_cap = order_total * 0.5
max_points_by_cap = int(max_value_cap * POINTS_PER_DOLLAR)
usable_points = min(loyalty_points, max_points_by_cap)
points_redeemed = (usable_points // REDEMPTION_STEP) * REDEMPTION_STEP
points_value = points_redeemed / POINTS_PER_DOLLAR

after_points = max(order_total - points_value, 0)
tier_discount_amount = after_points * tier_discount_pct
final_total = round(after_points - tier_discount_amount, 2)

total_savings = round(order_total - final_total, 2)
points_earned = int(order_total * earn_rates.get(product_category, 1))
remaining_points = loyalty_points - points_redeemed

result = {{
    "points_redeemed": points_redeemed,
    "tier_discount_pct": round(tier_discount_pct * 100, 1),
    "final_total": final_total,
    "total_savings": total_savings,
    "points_earned": points_earned,
    "remaining_points": remaining_points,
}}
print(json.dumps(result))
"""

    try:
        with code_session(REGION) as session:
            response = session.invoke(
                "executeCode",
                {"code": code, "language": "python", "clearContext": True},
            )
            for event in response.get("stream", []):
                result = event.get("result", {})
                for item in result.get("content", []):
                    if item.get("type") == "text":
                        return item["text"]
            raise RuntimeError("Code interpreter returned no output")

    except Exception as e:
        tier_rates = {"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}
        tier_discount_pct = tier_rates.get(tier, 0.0)
        final_total = round(order_total * (1 - tier_discount_pct), 2)
        fallback = {
            "points_redeemed": 0,
            "tier_discount_pct": round(tier_discount_pct * 100, 1),
            "final_total": final_total,
            "total_savings": round(order_total - final_total, 2),
            "points_earned": 0,
            "remaining_points": loyalty_points,
            "note": f"Code interpreter unavailable ({e}); tier-only discount applied.",
        }
        return json.dumps(fallback)


# ── TODO 8 — Agent Entrypoint ─────────────────────────────────────────────────
@app.entrypoint
async def invoke(payload, context=None):
    """
    Main handler called by AgentCore for every incoming request.

    Expected payload keys:
      prompt      (str, required) — the customer's message
      customer_id (str, optional) — unique customer identifier
      session_id  (str, optional) — session identifier; generated if absent
    """
    try:
        user_input = payload.get("prompt", "")
        actor_id = payload.get("customer_id", "anonymous")
        session_id = payload.get("session_id") or str(uuid.uuid4())

        memory_hook = MemoryHook(
            actor_id=actor_id,
            session_id=session_id,
            memory_client=memory_client,
            memory_id=MEMORY_ID,
        )
        agent_core_browser = AgentCoreBrowser(region=REGION)

        tools = [search_knowledge_base, calculate_loyalty_discount, agent_core_browser.browser]

        gateway_client = MCPClient(lambda: streamable_http_client(GATEWAY_URL))
        with gateway_client:
            try:
                gateway_tools = gateway_client.list_tools_sync()
                tools.extend(gateway_tools)

                logger.info(
                    "Gateway connected successfully. Loaded %d tools.",
                    len(gateway_tools),
                )

            except TimeoutError:
                logger.exception("Gateway tool loading timed out")

            except ConnectionError:
                logger.exception("Gateway connection failed")

            except Exception as exc:
                logger.exception(
                    "Gateway tool loading failed: %s", exc
                )

            agent = Agent(
                model=model,
                system_prompt=(
                    "You are a helpful, concise e-commerce customer support "
                    "assistant. Use the order/refund tools for account-specific "
                    "actions, the knowledge base tool for policy/product "
                    "questions, the discount calculator for loyalty math, and "
                    "the browser tool for live web lookups. Always ground "
                    "factual claims in tool output."
                ),
                tools=tools,
                hooks=[memory_hook],
            )
            response = agent(user_input)

        try:
            return response.message["content"][0]["text"]
        except (AttributeError, KeyError, IndexError, TypeError):
            return str(response)

    except Exception as e:
        logger.error("invoke failed: %s", e)
        return f"Sorry, something went wrong: {e}"


# ── CLI entry point (do not modify) ──────────────────────────────────────────
def main():
    """Run one invocation from the command line for local testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=str)
    args = parser.parse_args()
    response = asyncio.run(invoke(json.loads(args.payload)))
    print(response)


if __name__ == "__main__":
    app.run()
    # Uncomment the line below and comment app.run() for local CLI testing:
    # main()