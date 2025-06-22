import re
import dateparser

def validate_email(email: str) -> bool:
    """Validate email format using regex."""
    if not email:
        return False
    pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    return re.match(pattern, email) is not None

def validate_phone(phone: str) -> bool:
    """Validate phone number (simple version: digits, +, -, spaces allowed)."""
    if not phone:
        return False
    pattern = r"^[\d\+\-\s]{7,15}$"
    return re.match(pattern, phone) is not None

def parse_date(date_str: str):
    """
    Parse date string to a datetime.date object.
    Supports natural language (e.g. 'next Monday') using dateparser.
    Returns None if parsing fails.
    """
    if not date_str:
        return None
    dt = dateparser.parse(date_str)
    if dt:
        return dt.date()
    return None