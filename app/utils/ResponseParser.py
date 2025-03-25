import re


def parse_medical_scan_result(raw_result: str) -> dict:
    """
    Parse the raw markdown-formatted LLM response into a structured JSON.
    Removes trailing hyphens and other inconsistencies.
    """
    # Define the expected fields
    expected_fields = [
        "Scan Type",
        "Organ",
        "Tumor Type",
        "Tumor Subclass",
        "Detailed Description",
        "Possible Causes",
        "Clinical Insights"
    ]

    result_dict = {}

    # Try to extract information using regex patterns
    for field in expected_fields:
        # Look for field name followed by colon and any text until the next field or end
        pattern = rf"\*?\*?\s*{re.escape(field)}\s*:?\*?\*?\s*(.*?)(?=\*?\*?\s*(?:{"|".join([re.escape(f) for f in expected_fields])})\s*:|\Z)"
        match = re.search(pattern, raw_result, re.DOTALL)
        if match:
            # Clean up the extracted text
            value = match.group(1).strip()
            # Remove mark-down formatting and extra whitespace
            value = re.sub(r'\*\*|\*|\n+', ' ', value)
            value = re.sub(r'\s+', ' ', value).strip()
            # Remove trailing hyphens and any surrounding whitespace
            value = re.sub(r'\s*-+\s*$', '', value)
            # Convert field name to snake_case for JSON
            field_key = field.lower().replace(' ', '_')
            result_dict[field_key] = value
        else:
            # If field not found, include it as empty
            field_key = field.lower().replace(' ', '_')
            result_dict[field_key] = ""

    # Add a disclaimer field if present in the original text
    disclaimer_pattern = r"(?:Disclaimer|Important\s+Disclaimer)\s*:?\s*(.*?)(?=\Z)"
    disclaimer_match = re.search(disclaimer_pattern, raw_result, re.DOTALL)
    if disclaimer_match:
        disclaimer_text = disclaimer_match.group(1).strip()
        disclaimer_text = re.sub(r'\*\*|\*|\n+', ' ', disclaimer_text)
        disclaimer_text = re.sub(r'\s+', ' ', disclaimer_text).strip()
        # Remove trailing hyphens here too
        disclaimer_text = re.sub(r'\s*-+\s*$', '', disclaimer_text)
        result_dict["disclaimer"] = disclaimer_text

    # Extract any additional response to the user's question (after the structured analysis)
    user_response_pattern = r"(?:To answer your question|In response to your question|Regarding your question).*?:(.*?)(?=\Z)"
    user_response_match = re.search(user_response_pattern, raw_result, re.DOTALL)
    if user_response_match:
        response_text = user_response_match.group(1).strip()
        response_text = re.sub(r'\s+', ' ', response_text).strip()
        result_dict["llm_response"] = response_text

    return result_dict
