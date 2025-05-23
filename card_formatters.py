# card_formatters.py

def format_recommend_card(card_data, index):
    """
    Formats the summary card for a recommended major.
    """
    major_name = card_data.get('major_name_en', 'N/A')
    uni_name = card_data.get('university_name_en', 'N/A')
    degree = card_data.get('degree_level', 'N/A')
    duration = card_data.get('duration_years', 'N/A')

    text = f"🎓 *{major_name}*\n"
    text += f"🏛️ _{uni_name}_\n"
    if degree and duration:
        text += f"📚 {degree} - {duration} years\n"
    elif degree:
        text += f"📚 {degree}\n"
    # Add more fields as needed for the summary
    return text

def format_details_card(card_data):
    """
    Formats the detailed information card for a major.
    """
    major_name = card_data.get('major_name_en', 'N/A')
    uni_name = card_data.get('university_name_en', 'N/A')
    faculty_name = card_data.get('faculty_name_en', 'N/A')
    degree = card_data.get('degree_level', 'N/A')
    duration = card_data.get('duration_years', 'N/A')
    tuition = card_data.get('tuition_fee_usd_per_year', 'N/A')
    entry_req_en = card_data.get('entry_requirements_en', 'N/A')
    career_opp_en = card_data.get('career_opportunities_en', 'N/A')
    uni_logo = card_data.get('university_logo_url', '') # Get logo URL

    text = ""
    if uni_logo: # If there's a logo, you might be able to send it as a photo with caption.
                 # For now, this just adds it as text. Telegram Markdown for images is tricky.
                 # Consider sending photo separately then text, or just uni_name.
        text += f"🏛️ *{uni_name}*\n" # (Logo: {uni_logo})\n" # Markdown for image is not direct
    else:
        text += f"🏛️ *{uni_name}*\n"

    text += f"🎓 *Major: {major_name}*\n\n"

    if faculty_name and faculty_name != 'N/A':
        text += f"Faculty: {faculty_name}\n"
    if degree and degree != 'N/A':
        text += f"Degree: {degree}\n"
    if duration and duration != 'N/A':
        text += f"Duration: {duration} years\n"
    if tuition and tuition != 'N/A':
        text += f"Tuition Fee (USD/Year): {tuition}\n\n"

    if entry_req_en and entry_req_en != 'N/A':
        text += f"*Entry Requirements (English):*\n{entry_req_en}\n\n"

    if career_opp_en and career_opp_en != 'N/A':
        text += f"*Career Opportunities (English):*\n{career_opp_en}\n\n"

    # You can add more fields from `card_data` here as needed
    # For example:
    # scholarship_en = card_data.get('scholarship_info_en', 'N/A')
    # if scholarship_en and scholarship_en != 'N/A':
    #     text += f"*Scholarship Information (English):*\n{scholarship_en}\n\n"

    return text