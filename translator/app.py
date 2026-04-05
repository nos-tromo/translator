"""Streamlit frontend for the Translator service.

Renders a two-column layout where source text and translation sit side-by-side.
Language pairs are selected via dropdowns that mirror the layout. Plain text can
be entered directly or uploaded as a .txt file. On submission the payload is
POSTed to the FastAPI backend and the result is written back into the output
column without shifting the page layout.
"""

import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Translator", page_icon="🌐", layout="wide")
st.title("🌐 Translator")


@st.cache_data(show_spinner=False)
def load_languages() -> list[dict]:
    """Fetch the list of supported languages from the backend and sort by name.

    Results are cached for the lifetime of the Streamlit session so that the
    backend is only queried once per page load.

    Returns:
        list[dict]: Sorted list of ``{"code": str, "name": str}`` dicts, or an
            empty list if the request fails.
    """
    try:
        r = requests.get(f"{BACKEND_URL}/languages", timeout=10)
        r.raise_for_status()
        return sorted(r.json(), key=lambda x: x["name"])
    except Exception as e:
        st.error(f"Could not load language list from backend: {e}")
        return []


languages = load_languages()
lang_name_to_code = {lang["name"]: lang["code"] for lang in languages}
lang_names = list(lang_name_to_code.keys())

# ── Language selectors ─────────────────────────────────────────────────────────

AUTO_DETECT = "Auto-detect"
source_lang_col, target_lang_col = st.columns(2)

with source_lang_col:
    source_lang_options = [AUTO_DETECT] + lang_names
    source_lang_name = st.selectbox("Source language", source_lang_options, index=0)

with target_lang_col:
    default_index = lang_names.index("English") if "English" in lang_names else 0
    target_lang_name = st.selectbox("Target language", lang_names, index=default_index)

# ── Text areas ─────────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader("Upload a .txt file (optional)", type=["txt"])
if uploaded_file:
    file_text = uploaded_file.read().decode("utf-8")
else:
    file_text = ""

input_col, output_col = st.columns(2)

with input_col:
    text = st.text_area(
        "Text to translate",
        value=file_text,
        height=300,
        placeholder="Enter or paste text here…",
    )

translation_placeholder = output_col.empty()
translation_placeholder.text_area(
    "Translation",
    value="",
    height=300,
    disabled=True,
    placeholder="Translation will appear here…",
)

# ── Translate ──────────────────────────────────────────────────────────────────

if st.button("Translate", type="primary", disabled=not text.strip()):
    with st.spinner("Translating…"):
        try:
            payload = {
                "text": text.strip(),
                "target_lang": lang_name_to_code[target_lang_name],
            }
            if source_lang_name != AUTO_DETECT:
                payload["source_lang"] = lang_name_to_code[source_lang_name]
            r = requests.post(f"{BACKEND_URL}/translate", json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
        except requests.exceptions.HTTPError as e:
            st.error(f"Backend error ({e.response.status_code}): {e.response.text}")
            st.stop()
        except Exception as e:
            st.error(f"Request failed: {e}")
            st.stop()

    detected = data.get("detected_language", {})
    translation = data.get("translation", "")

    detected_name = detected.get("name", "")
    detected_flag = detected.get("flag", "")
    if detected_name:
        label = (
            "Detected source language"
            if source_lang_name == AUTO_DETECT
            else "Source language"
        )
        st.info(f"{label}: {detected_flag} {detected_name}")

    translation_placeholder.text_area(
        "Translation",
        value=translation,
        height=300,
        placeholder="Translation will appear here…",
    )
