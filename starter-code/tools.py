import json
import os
from typing import List, Dict, Any
from datetime import datetime

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "raw-data")

# ---------------------------------------------------------------------------
# Tool #1: search_product_catalog
# Đọc file product_catalog.json và lọc theo category, max_price.
# ---------------------------------------------------------------------------

def search_product_catalog(category: str, max_price: int = 999999999999) -> List[Dict[str, Any]]:
    """
    Tra cứu sản phẩm/dịch vụ Vingroup theo danh mục và giá tối đa.
    
    Args:
        category: Loại sản phẩm ('xe_dien' hoặc 'du_lich').
        max_price: Giá tối đa (VNĐ). Mặc định không giới hạn.
    
    Returns:
        Danh sách sản phẩm phù hợp điều kiện.
    """
    catalog_file = os.path.join(RAW_DATA_DIR, "product_catalog.json")
    if not os.path.exists(catalog_file):
        return [{"error": "Product catalog file not found."}]

    with open(catalog_file, "r", encoding="utf-8") as f:
        products = json.load(f)

    results = [
        p for p in products
        if p["category"].lower() == category.lower()
        and p["price_vnd"] <= max_price
    ]
    return results



# ---------------------------------------------------------------------------
# Tool #2: submit_support_ticket
# Tạo ticket mới và lưu vào support_tickets.json.
# ---------------------------------------------------------------------------

def submit_support_ticket(
    customer_name: str,
    issue_description: str,
    priority: str = "medium"
) -> Dict[str, Any]:
    """
    Ghi nhận yêu cầu hỗ trợ của khách hàng vào hệ thống ticket.
    
    Args:
        customer_name: Tên khách hàng.
        issue_description: Mô tả vấn đề cần hỗ trợ.
        priority: Mức độ ưu tiên ('low', 'medium', 'high'). Mặc định 'medium'.
    
    Returns:
        Thông tin ticket vừa tạo bao gồm ticket_id, status.
    """
    tickets_file = os.path.join(RAW_DATA_DIR, "support_tickets.json")

    # Load existing tickets
    existing_tickets = []
    if os.path.exists(tickets_file):
        with open(tickets_file, "r", encoding="utf-8") as f:
            existing_tickets = json.load(f)

    # Generate new ticket ID
    today = datetime.now().strftime("%Y%m%d")
    seq = len(existing_tickets) + 1
    ticket_id = f"TK-{today}-{seq:03d}"

    new_ticket = {
        "ticket_id": ticket_id,
        "customer_name": customer_name,
        "issue_description": issue_description,
        "priority": priority.lower(),
        "status": "open",
        "created_at": datetime.now().isoformat() + "+07:00",
        "category": "general"
    }

    existing_tickets.append(new_ticket)

    # Persist to file
    with open(tickets_file, "w", encoding="utf-8") as f:
        json.dump(existing_tickets, f, indent=2, ensure_ascii=False)

    return {
        "ticket_id": ticket_id,
        "customer_name": customer_name,
        "priority": priority.lower(),
        "status": "open",
        "message": f"Ticket {ticket_id} đã được tạo thành công cho khách hàng {customer_name}."
    }


# ---------------------------------------------------------------------------
# TOOL_DEFINITIONS — JSON Schemas mô tả cho LLM
# JSON Schema mô tả tên, chức năng và tham số của từng tool.
# ---------------------------------------------------------------------------
TOOL_DEFINITIONS = [
    {
        "name": "search_product_catalog",
        "description": "Tra cứu sản phẩm/dịch vụ Vingroup (xe điện VinFast, resort Vinpearl) theo danh mục và giá tối đa.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Loại sản phẩm: 'xe_dien' cho xe điện VinFast, 'du_lich' cho gói du lịch Vinpearl.",
                    "enum": ["xe_dien", "du_lich"]
                },
                "max_price": {
                    "type": "integer",
                    "description": "Giá tối đa tính bằng VNĐ. Ví dụ: 600000000 cho 600 triệu."
                }
            },
            "required": ["category"]
        }
    },
    {
        "name": "submit_support_ticket",
        "description": "Ghi nhận yêu cầu hỗ trợ khách hàng vào hệ thống ticket. Dùng khi khách hàng phản ánh lỗi, khiếu nại hoặc cần trợ giúp.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_name": {
                    "type": "string",
                    "description": "Họ tên đầy đủ của khách hàng."
                },
                "issue_description": {
                    "type": "string",
                    "description": "Mô tả chi tiết vấn đề khách hàng gặp phải."
                },
                "priority": {
                    "type": "string",
                    "description": "Mức độ ưu tiên: 'low' (thấp), 'medium' (trung bình), 'high' (cao/khẩn cấp).",
                    "enum": ["low", "medium", "high"],
                    "default": "medium"
                }
            },
            "required": ["customer_name", "issue_description"]
        }
    }
]


# ---------------------------------------------------------------------------
# TOOL_MAP — Ánh xạ tên tool → hàm thực thi
# ---------------------------------------------------------------------------

TOOL_MAP = {
    "search_product_catalog": search_product_catalog,
    "submit_support_ticket": submit_support_ticket
}
