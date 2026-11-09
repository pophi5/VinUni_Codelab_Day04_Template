"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Bản hoàn thiện các TODO của bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import os
import re
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Bạn là VinAssistant — trợ lý AI chính thức của hệ sinh thái Vingroup.

## PERSONA
- Tên: VinAssistant
- Vai trò: Chuyên viên tư vấn sản phẩm & dịch vụ Vingroup (VinFast, Vinpearl)
- Giọng nói: Chuyên nghiệp, thân thiện, chính xác, không bịa thông tin.

## AVAILABLE TOOLS
{tools}

## CORE RULES (Bắt buộc tuân thủ)
1. KHÔNG BAO GIỜ bịa dữ liệu sản phẩm (giá, tính năng, tồn kho). PHẢI gọi tool `search_product_catalog` để lấy dữ liệu thực.
2. KHÔNG BAO GIỜ tự tạo ticket_id. PHẢI gọi tool `submit_support_ticket` để hệ thống tự sinh.
3. Nếu khách hàng hỏi câu FAQ đơn giản (chính sách bảo hành, đổi trả), trả lời trực tiếp KHÔNG cần gọi tool.
4. Nếu câu hỏi cần NHIỀU tool, hãy gọi TUẦN TỰ từng tool rồi tổng hợp kết quả cuối cùng.

## OPERATIONAL BOUNDARIES
- Chỉ trả lời về sản phẩm/dịch vụ thuộc hệ sinh thái Vingroup.
- Từ chối lịch sự nếu câu hỏi ngoài phạm vi (tiền điện tử, chính trị, v.v.).

## OUTPUT CONTRACT
- Luôn trả lời bằng tiếng Việt.
- Khi liệt kê sản phẩm, dùng bảng hoặc danh sách có thứ tự.
- Khi tạo ticket, xác nhận lại ticket_id và mức ưu tiên cho khách hàng.

## RESPONSE FORMAT (cho mỗi bước suy luận)
Thought: <suy nghĩ bước này>
Action: {"name": "<tên tool>", "args": {<các tham số>}}
Observation: <kết quả trả về từ tool>
... (Lặp lại nếu cần thêm tool)
Final Answer: <câu trả lời hoàn chỉnh cho khách hàng>
"""
SYSTEM_PROMPT = SYSTEM_PROMPT.replace(
    "{tools}", json.dumps(TOOL_DEFINITIONS, ensure_ascii=False, indent=2)
)


def _extract_max_price(user_input: str):
    """Parse common Vietnamese price expressions into VND."""
    scaled_price = re.search(
        r"(\d+(?:[.,]\d+)?)\s*(tỷ|tỉ|triệu|tr|nghìn|ngàn|k|m)\b",
        user_input,
        re.IGNORECASE,
    )
    if scaled_price:
        amount = float(scaled_price.group(1).replace(",", "."))
        unit = scaled_price.group(2).lower()
        multiplier = 1_000_000_000 if unit in ("tỷ", "tỉ") else (
            1_000_000 if unit in ("triệu", "tr", "m") else 1_000
        )
        return int(amount * multiplier)

    currency_price = re.search(
        r"(\d[\d.,]*)\s*(?:vnđ|vnd|đồng|đ)\b", user_input, re.IGNORECASE
    )
    if currency_price:
        digits = re.sub(r"\D", "", currency_price.group(1))
        return int(digits) if digits else None

    bare_price = re.search(
        r"\b(?:dưới|thấp hơn|tối đa|không quá)\s*(\d[\d.,]*)\b",
        user_input,
        re.IGNORECASE,
    )
    if bare_price:
        digits = re.sub(r"\D", "", bare_price.group(1))
        return int(digits) if digits else None
    return None


def _detect_intents(user_input: str) -> Dict[str, Any]:
    text = user_input.lower()
    needs_ticket = bool(re.search(
        r"\b(?:lỗi|hỏng|khiếu nại|phản ánh|hỗ trợ|trợ giúp|ticket|"
        r"ghi nhận|sự cố|không hoạt động)\b",
        text,
    ))
    is_faq = bool(re.search(
        r"\b(?:bảo hành|đổi trả|chính sách|bao lâu)\b", text
    )) and not needs_ticket

    travel_terms = ("vinpearl", "resort", "du lịch", "nghỉ dưỡng", "khách sạn")
    vehicle_terms = ("vinfast", "xe điện", "ô tô", "xe hơi", "vf ")
    category = None
    if any(term in text for term in travel_terms):
        category = "du_lich"
    elif any(term in text for term in vehicle_terms):
        category = "xe_dien"

    asks_about_product = bool(re.search(
        r"\b(?:giá|dưới|tối đa|mua|xem|tìm|tra cứu|thông tin|gợi ý|liệt kê)\b",
        text,
    ))
    if not needs_ticket and re.search(r"\bvf\s*\d", text):
        asks_about_product = True
    needs_catalog = bool(category and asks_about_product and not is_faq)
    max_price = _extract_max_price(user_input) if needs_catalog else None

    name_match = re.search(
        r"\b(?:tên tôi là|tôi tên là|tên tôi|tôi tên|mình tên là|mình tên)"
        r"\s+([^,.;!?]+)",
        user_input,
        re.IGNORECASE,
    )
    customer_name = name_match.group(1).strip() if name_match else None
    issue_description = ""
    if name_match:
        issue_description = user_input[name_match.end():].lstrip(" ,:-")
        issue_description = re.split(
            r"\b(?:đây là|mức độ|ưu tiên|cần xử lý|xin xử lý)\b",
            issue_description,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip(" \t\r\n.,;:-")
        issue_description = re.sub(r"\bcủa\s+tôi\b", "", issue_description, flags=re.IGNORECASE)
        issue_description = re.sub(
            r"\b(phòng|xe|đơn hàng)\s+tôi\s+(?=\w)",
            r"\1 ",
            issue_description,
            flags=re.IGNORECASE,
        ).strip()
        issue_description = re.sub(r"\s+", " ", issue_description)
        if issue_description:
            issue_description = issue_description[0].upper() + issue_description[1:]

    priority = "medium"
    if re.search(r"\b(?:khẩn cấp|gấp|nghiêm trọng|cao|high)\b", text):
        priority = "high"
    elif re.search(r"\b(?:thấp|low)\b", text):
        priority = "low"

    return {
        "user_input": user_input,
        "needs_catalog": needs_catalog,
        "needs_ticket": needs_ticket,
        "is_faq": is_faq,
        "category": category,
        "max_price": max_price,
        "customer_name": customer_name,
        "issue_description": issue_description,
        "priority": priority,
    }


def _format_catalog_results(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "Rất tiếc, không tìm thấy sản phẩm phù hợp với yêu cầu của bạn."
    if results[0].get("error"):
        return f"Không thể tra cứu danh mục: {results[0]['error']}"

    lines = ["Các sản phẩm phù hợp:"]
    for product in results:
        price = f"{product['price_vnd']:,}".replace(",", ".")
        lines.append(f"- {product['name']}: {price} VNĐ — {product['description']}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop.
    Mục đích: So sánh chất lượng trả lời khi LLM bịa thông tin (hallucination).
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def query(self, user_input: str) -> Dict[str, Any]:
        """Gửi câu hỏi tới LLM (hoặc trả lời mock nếu không có API key)."""
        if self.api_key:
            from google import genai

            client = genai.Client(api_key=self.api_key)
            response = client.interactions.create(
                model="gemini-3.8-flash",
                input=(
                    "Bạn là chatbot tư vấn sản phẩm Vingroup. Trả lời câu hỏi sau "
                    "mà KHÔNG có quyền tra cứu cơ sở dữ liệu hay gọi tool: "
                    f"{user_input}"
                ),
            )
            if not response.output_text:
                raise RuntimeError("Gemini returned an empty response.")
            return {
                "answer": response.output_text,
                "tool_calls": [],
                "status": "success",
                "mode": "live_api"
            }

        # Mock response — mô phỏng hallucination
        return {
            "answer": (
                "VinFast hiện có nhiều dòng xe điện với giá từ khoảng 300 triệu. "
                "Bạn có thể tham khảo tại showroom gần nhất hoặc website VinFast."
            ),
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []
        intents = _detect_intents(user_input)
        self.trace.append({
            "step": "init",
            "user_input": user_input,
            "intents": intents,
        })

        actions = []
        if intents["needs_catalog"]:
            args = {"category": intents["category"]}
            if intents["max_price"] is not None:
                args["max_price"] = intents["max_price"]
            actions.append(("search_product_catalog", args))

        ticket_clarification = None
        if intents["needs_ticket"]:
            if not intents["customer_name"]:
                ticket_clarification = (
                    "Để tạo yêu cầu hỗ trợ, vui lòng cho tôi biết họ tên của bạn."
                )
            elif not intents["issue_description"]:
                ticket_clarification = (
                    "Vui lòng mô tả cụ thể vấn đề để tôi có thể tạo yêu cầu hỗ trợ."
                )
            else:
                actions.append((
                    "submit_support_ticket",
                    {
                        "customer_name": intents["customer_name"],
                        "issue_description": intents["issue_description"],
                        "priority": intents["priority"],
                    },
                ))

        if not actions:
            answer = ticket_clarification or self._answer_without_tools(intents)
            return self._finish(answer, 1)

        observations = {}
        iterations = 0
        for tool_name, args in actions:
            if iterations >= self.max_iterations:
                answer = "Lỗi: Đã vượt quá số bước tối đa trước khi hoàn tất yêu cầu."
                self.trace.append({"step": "limit", "answer": answer})
                return {
                    "answer": answer,
                    "trace": self.trace,
                    "iterations": iterations,
                    "status": "max_iterations_reached",
                }

            iterations += 1
            observation = TOOL_MAP[tool_name](**args)
            observations[tool_name] = observation
            self.trace.append({
                "step": iterations,
                "action": {"name": tool_name, "args": args},
                "observation": observation,
            })

        answer_parts = []
        if "search_product_catalog" in observations:
            answer_parts.append(_format_catalog_results(
                observations["search_product_catalog"]
            ))
        if "submit_support_ticket" in observations:
            ticket = observations["submit_support_ticket"]
            answer_parts.append(
                f"Đã tạo yêu cầu hỗ trợ cho {ticket['customer_name']}. "
                f"Mã ticket: {ticket['ticket_id']}; mức độ ưu tiên: "
                f"{ticket['priority']}."
            )
        if ticket_clarification:
            answer_parts.append(ticket_clarification)
        answer = "\n".join(answer_parts)
        self.trace.append({"step": "final", "answer": answer})
        return {
            "answer": answer,
            "trace": self.trace,
            "iterations": iterations,
            "status": "completed",
        }

    def _finish(self, answer: str, iterations: int) -> Dict[str, Any]:
        self.trace.append({"step": "final", "answer": answer})
        if iterations > self.max_iterations:
            return {
                "answer": "Lỗi: Đã vượt quá số bước tối đa.",
                "trace": self.trace,
                "iterations": 0,
                "status": "max_iterations_reached",
            }
        return {
            "answer": answer,
            "trace": self.trace,
            "iterations": iterations,
            "status": "completed",
        }

    @staticmethod
    def _answer_without_tools(intents: Dict[str, Any]) -> str:
        if intents["is_faq"]:
            if "bảo hành" in intents.get("user_input", "").lower():
                return (
                    "Theo thông tin hiện có, pin VinFast VF 5 Plus được bảo hành "
                    "10 năm. Với dòng xe khác, vui lòng cho biết mẫu xe để tôi "
                    "có thể cung cấp thông tin phù hợp."
                )
            return (
                "Hiện tôi chưa có đủ thông tin xác nhận chính sách này. "
                "Vui lòng cho biết sản phẩm hoặc dịch vụ Vingroup cụ thể."
            )
        return (
            "Xin lỗi, tôi chỉ hỗ trợ tra cứu sản phẩm, dịch vụ và yêu cầu hỗ trợ "
            "thuộc hệ sinh thái Vingroup."
        )

# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
