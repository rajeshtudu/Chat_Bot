import re
import dateparser

def validate_email(email):
    pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    return re.match(pattern, email) is not None

def validate_phone(phone):
    # Basic validation: must be 7-15 digits, optionally with +, spaces, or dashes
    pattern = r"^\+?[\d\s\-]{7,15}$"
    return re.match(pattern, phone) is not None

def parse_date(date_text):
    parsed = dateparser.parse(date_text)
    return parsed