import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Applied when the login screen is shown — hides the sidebar and centres the form.
_LOGIN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Cormorant+Garamond:wght@400;600&display=swap');

html, body, [class*="css"], p, li, span, div {
    font-family: 'DM Sans', sans-serif;
}

/* Hide sidebar entirely on the login screen */
[data-testid="stSidebar"] { display: none !important; }

/* Centre the form in the wide-layout main area */
.main .block-container {
    max-width: 420px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    padding-top: 5rem !important;
    background-color: #2D1B4E !important;
}
.stApp { background-color: #2D1B4E; }

/* Wordmark */
.insyte-login-wordmark {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 2.4rem;
    font-weight: 600;
    color: #F2EFF8 !important;
    letter-spacing: 0.06em;
    margin: 0 0 2px 0;
    line-height: 1.2;
}
.insyte-login-sub {
    font-size: 0.72rem;
    color: #C4A8FF !important;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin: 0 0 2rem 0;
}

/* Divider */
hr { border-color: #4A3070 !important; }

/* Input */
.stTextInput > div > div > input {
    background-color: #1F1238;
    border: 1px solid #4A3070;
    color: #F2EFF8;
    border-radius: 6px;
    font-family: 'DM Sans', sans-serif;
}
.stTextInput > div > div > input:focus {
    border-color: #C4A8FF;
    box-shadow: 0 0 0 2px rgba(196,168,255,0.2);
}
.stTextInput > label p { color: #B8AECE !important; }

/* Button */
.stButton > button {
    background-color: #C4A8FF;
    color: #1F1238;
    border: none;
    border-radius: 6px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    letter-spacing: 0.04em;
    transition: background 0.15s;
}
.stButton > button:hover {
    background-color: #D4B8FF;
    color: #1F1238;
    border: none;
}

/* Disclaimer */
.insyte-disclaimer {
    background-color: #1F1238;
    border: 1px solid #4A3070;
    border-radius: 6px;
    padding: 12px 14px;
    margin-top: 1.6rem;
    font-size: 0.75rem;
    color: #B8AECE;
    line-height: 1.6;
}
.insyte-disclaimer a { color: #C4A8FF; text-decoration: none; }

div[data-testid="stAlert"] { border-radius: 6px; }
.stMarkdown p { color: #F2EFF8; }
</style>
"""


def check_password() -> bool:
    """
    Render a password gate. Returns True only when the user has authenticated.
    st.set_page_config is intentionally NOT called here — appv2.py calls it
    first (before this function) so there is exactly one call per render cycle.
    """
    if st.session_state.get("authenticated"):
        return True

    st.markdown(_LOGIN_CSS, unsafe_allow_html=True)

    st.markdown(
        '<p class="insyte-login-wordmark">Insyte</p>'
        '<p class="insyte-login-sub">Document Intelligence</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    password = st.text_input("Access password", type="password", key="pw_input")

    if st.button("Enter", use_container_width=True):
        expected = os.getenv("INSYTE_PASSWORD", "")
        if not expected:
            st.error("INSYTE_PASSWORD is not set in environment.")
        elif password == expected:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")

    st.markdown(
        '<div class="insyte-disclaimer">'
        "Research prototype — not for use with real client data. "
        "Submit synthetic or anonymized documents only. "
        "Outputs are not clinically validated and should not inform clinical decisions. "
        'Report issues to <a href="mailto:hello@tryinsyte.ca">hello@tryinsyte.ca</a>.'
        "</div>",
        unsafe_allow_html=True,
    )

    return False
