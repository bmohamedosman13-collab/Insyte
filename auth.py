import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

_LOGIN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Cormorant+Garamond:wght@400;600&display=swap');

html, body, [class*="css"], p, li, span, div {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'Cormorant Garamond', serif !important;
}

.stApp { background-color: #0C0820; }
.main  { background-color: #0C0820; }
.main .block-container {
    background-color: #0C0820;
    max-width: 420px;
    padding-top: 5rem;
}

/* Wordmark */
.insyte-login-wordmark {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 2.4rem;
    font-weight: 600;
    color: #F5EFE0 !important;
    letter-spacing: 0.06em;
    margin: 0 0 2px 0;
    line-height: 1.2;
}
.insyte-login-sub {
    font-size: 0.72rem;
    color: #C9A84C !important;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin: 0 0 2rem 0;
}

/* Divider */
hr { border-color: #35265A !important; }

/* Input */
.stTextInput > div > div > input {
    background-color: #160F2E;
    border: 1px solid #35265A;
    color: #F5EFE0;
    border-radius: 6px;
    font-family: 'DM Sans', sans-serif;
}
.stTextInput > div > div > input:focus {
    border-color: #C9A84C;
    box-shadow: 0 0 0 2px rgba(201,168,76,0.2);
}
.stTextInput > label p { color: #C4B89A !important; }

/* Button */
.stButton > button {
    background-color: #C9A84C;
    color: #0C0820;
    border: none;
    border-radius: 6px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    letter-spacing: 0.04em;
    transition: background 0.15s;
}
.stButton > button:hover {
    background-color: #D9BF6E;
    color: #0C0820;
    border: none;
}

/* Disclaimer box */
.insyte-disclaimer {
    background-color: #160F2E;
    border: 1px solid #35265A;
    border-radius: 6px;
    padding: 12px 14px;
    margin-top: 1.6rem;
    font-size: 0.75rem;
    color: #C4B89A;
    line-height: 1.6;
}
.insyte-disclaimer a {
    color: #C9A84C;
    text-decoration: none;
}

/* Error / info alerts */
div[data-testid="stAlert"] { border-radius: 6px; }
.stMarkdown p { color: #F5EFE0; }
</style>
"""


def check_password() -> bool:
    """
    Render a password gate. Returns True only when the user has authenticated.
    Nothing in the calling app should render until this returns True.
    """
    if st.session_state.get("authenticated"):
        return True

    st.set_page_config(page_title="Insyte", layout="centered")
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
