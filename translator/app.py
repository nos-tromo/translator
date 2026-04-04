import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="TranslateGemma", page_icon="🌐", layout="centered")
st.title("🌐 TranslateGemma")


@st.cache_data(show_spinner=False)
def load_languages() -> list[dict]:
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

# ── Input ──────────────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader("Upload a .txt file (optional)", type=["txt"])
if uploaded_file:
    file_text = uploaded_file.read().decode("utf-8")
else:
    file_text = ""

text = st.text_area(
    "Text to translate",
    value=file_text,
    height=200,
    placeholder="Enter or paste text here…",
)

default_index = lang_names.index("English") if "English" in lang_names else 0
target_lang_name = st.selectbox("Target language", lang_names, index=default_index)

# ── Translate ──────────────────────────────────────────────────────────────────

if st.button("Translate", type="primary", disabled=not text.strip()):
    with st.spinner("Translating…"):
        try:
            payload = {
                "text": text.strip(),
                "target_lang": lang_name_to_code[target_lang_name],
            }
            r = requests.post(
                f"{BACKEND_URL}/translate", json=payload, timeout=120
            )
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
        st.info(f"Detected source language: {detected_flag} {detected_name}")

    st.subheader("Translation")
    st.text_area("", value=translation, height=200, label_visibility="collapsed")
    st.code(translation, language=None)
