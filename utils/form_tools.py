import streamlit as st
from utils.validation import validate_email, validate_phone, parse_date

def book_appointment_form(*args, **kwargs):
    """
    A simple conversational form for booking an appointment.
    Returns a confirmation string or error messages.
    """

    st.markdown("### Book an Appointment")

    # Collect inputs with validation
    name = st.text_input("Full Name")
    phone = st.text_input("Phone Number")
    email = st.text_input("Email Address")
    date_str = st.text_input("Appointment Date (e.g. 2023-12-31 or 'next Monday')")

    errors = []

    if name.strip() == "":
        errors.append("Name is required.")

    if phone and not validate_phone(phone):
        errors.append("Invalid phone number format.")

    if email and not validate_email(email):
        errors.append("Invalid email address format.")

    parsed_date = None
    if date_str:
        parsed_date = parse_date(date_str)
        if parsed_date is None:
            errors.append("Invalid date format. Please enter a full date like YYYY-MM-DD or relative date like 'next Monday'.")
    else:
        errors.append("Appointment date is required.")

    # Submit button
    if st.button("Confirm Appointment"):
        if errors:
            for err in errors:
                st.error(err)
            return "Appointment booking failed due to validation errors."
        else:
            # Here you could save the appointment data to a DB or send an email, etc.
            confirmation_msg = (
                f"Thank you {name}! Your appointment is booked for {parsed_date.strftime('%Y-%m-%d')}.\n"
                f"We will contact you at {phone} or {email} if needed."
            )
            st.success(confirmation_msg)
            return confirmation_msg

    # Return empty string if form not submitted
    return ""