import streamlit as st
from utils.validation import validate_email, validate_phone, parse_date

def book_appointment_form(_):
    with st.form("appointment_form"):
        st.subheader("📅 Book an Appointment")
        name = st.text_input("Full Name")
        phone = st.text_input("Phone Number")
        email = st.text_input("Email Address")
        date_input = st.text_input("Preferred Appointment Date (e.g., 'next Monday')")

        submitted = st.form_submit_button("Submit")

        if submitted:
            errors = []

            if not name:
                errors.append("Name is required.")
            if not validate_phone(phone):
                errors.append("Invalid phone number format.")
            if not validate_email(email):
                errors.append("Invalid email format.")

            parsed_date = parse_date(date_input)
            if not parsed_date:
                errors.append("Could not understand the appointment date.")

            if errors:
                for error in errors:
                    st.error(error)
                return "❌ Failed to book appointment. Please correct the errors."

            st.success(f"✅ Appointment booked for {name} on {parsed_date.strftime('%Y-%m-%d')}")
            return {
                "name": name,
                "phone": phone,
                "email": email,
                "date": parsed_date.strftime('%Y-%m-%d')
            }
