import csv, os
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st
from datetime import datetime
from urllib.parse import parse_qs, urlparse

# --- LLM & language detection ---
import os
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    oai = OpenAI(api_key=OPENAI_API_KEY)
else:
    oai = None


# ---- paths ----
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
LOGFILE = DATA_DIR / "checkins.csv"

def init_logfile():
    """اگر فایل وجود ندارد، هدر بنویس."""
    if not LOGFILE.exists():
        with LOGFILE.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["ts_utc", "user_id", "role", "lang", "mood", "text"])

def log_event(role: str, text: str, user_id: str = "demo", mood=None, lang_code=None):
    """نوشتن یک ردیف در CSV با رعایت نکات ویندوز (newline, utf-8)."""
    clean = (text or "").replace("\r", " ").replace("\n", " ").strip()
    row = [
        datetime.utcnow().isoformat(timespec="seconds"),
        user_id,
        role,
        (lang_code or "English"),
        (mood if mood is not None else ""),
        clean
    ]
    with LOGFILE.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(row)

# ===================== Helpers (LANG, CRISIS, EXERCISES, DISCLAIMER) =====================

EX_LIB = {
    "grounding": {
        "en": "2-minute grounding: 5 things you see, 4 feel, 3 hear, 2 smell, 1 taste.",
        "fa": "گراندینگ ۲ دقیقه‌ای: ۵ چیزی که می‌بینی، ۴ لمس می‌کنی، ۳ می‌شنوی، ۲ بو می‌کنی، ۱ می‌چشی.",
        "nl": "Gronding 2 min: 5 dingen die je ziet, 4 voelt, 3 hoort, 2 ruikt, 1 proeft."
    },
    "breathing": {
        "en": "Box breathing ×5: inhale 4s, hold 4s, exhale 4s, hold 4s.",
        "fa": "تنفس جعبه‌ای ×۵: ۴ ثانیه دم، ۴ نگه‌دار، ۴ بازدم، ۴ نگه‌دار.",
        "nl": "Box breathing ×5: 4s in, 4s vasthouden, 4s uit, 4s vasthouden."
    },
    "gratitude": {
        "en": "Gratitude note: write 3 tiny good moments from today.",
        "fa": "قدردانی: ۳ لحظه خوب خیلی کوچک از امروزت را بنویس.",
        "nl": "Dankbaarheid: schrijf 3 kleine fijne momenten van vandaag op."
    }
}

def lang_code_of(text: str, fallback="en"):
    try:
        from langdetect import detect
        l = detect(text or "")
        if l.startswith("fa"): return "fa"
        if l.startswith("nl"): return "nl"
        return "en"
    except Exception:
        return fallback

CRISIS_EN = ["suicide","kill myself","end it","self harm","hopeless","harm others"]
CRISIS_FA = ["خودکشی","به زندگیم پایان","کشتن خود","آسیب به خود","ناامید","صدمه به دیگران"]
CRISIS_NL = ["zelfmoord","mezelf doden","einde maken","zelfbeschadiging","hopeloos","anderen pijn doen"]

def is_crisis_text(text: str, l: str) -> bool:
    t = (text or "").lower()
    keys = CRISIS_EN if l=="en" else (CRISIS_FA if l=="fa" else CRISIS_NL)
    return any(k in t for k in keys)

DISCLAIMER_EN = ("I'm an AI well-being coach, not a substitute for professional care. "
                 "If you're in immediate danger, contact local emergency services.")
DISCLAIMER_FA = ("من همراه رفاه روان هستم و جایگزین درمانگر نیستم. "
                 "اگر در خطر فوری هستی، با اورژانس تماس بگیر.")
DISCLAIMER_NL = ("Ik ben een AI-welzijnscoach, geen vervanging voor professionele zorg. "
                 "Bel bij direct gevaar de hulpdiensten.")

# =========================================================================================
# === LLM well-being coach with guardrails ===
def llm_support(user_history: list[dict], user_mood: int|None, ui_lang: str) -> str:
    """
    user_history: [{"role":"user"/"assistant","content":"..."}]
    ui_lang: "en"|"fa"|"nl" → پاسخ در همین زبان
    خروجی: بازتاب احساس + پیشنهاد پشتیبانی
    """
    if not oai:
        return {
            "fa": "من در حال حاضر در حالت محدود کار می‌کنم. متشکرم که با من صحبت کردی. می‌توانی در مورد احساست بیشتر بنویسی.",
            "nl": "Ik werk momenteel in beperkte modus. Bedankt dat je met me praat. Je kunt meer over je gevoelens schrijven.",
            "en": "I'm currently in limited mode. Thank you for talking to me. You can write more about your feelings."
        }[ui_lang]
    
    sys = {
        "role": "system",
        "content": (
            "You are a brief, compassionate AI well-being coach. "
            "You are NOT a therapist; do NOT diagnose or give medical advice. "
            "If you detect crisis or self-harm intent, STOP and advise contacting local emergency services. "
            "Keep replies under 120 words. Reflect feelings in 1–2 sentences, then provide supportive guidance. "
            "No clinical labels. Respond in language code: " + ui_lang + "."
        )
    }
    msgs = [sys] + user_history[-6:]
    try:
        r = oai.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=240,
            messages=msgs
        )
        text = r.choices[0].message.content.strip()
    except Exception:
        return {
            "fa": "فعلاً مشکل فنی پیش آمده. بیاییم ۶۰ ثانیه تنفس جعبه‌ای انجام بدیم: ۴ دم، ۴ نگه‌دار، ۴ بازدم، ۴ نگه‌دار.",
            "nl": "Er is tijdelijk een technisch probleem. 60s box breathing: 4s in, 4s vasthouden, 4s uit, 4s vasthouden.",
            "en": "Temporary issue. Try 60s box breathing: inhale 4s, hold 4s, exhale 4s, hold 4s."
        }[ui_lang]

    text = text[:800]
    banned = ["diagnose","prescribe","medication","تشخیص","دارو","voorschrijven"]
    if any(b in text.lower() for b in banned):
        text = {
            "fa": "بیاییم یک تمرین غیرپزشکی انجام بدیم: تنفس کوتاه، بعد ۳ نکته‌ی خوب امروزت رو بنویس.",
            "nl": "Laten we iets niet-medisch doen: kort ademen, daarna 3 fijne momenten noteren.",
            "en": "Let's keep it non-medical: brief breathing, then write 3 small good moments from today."
        }[ui_lang]

    return text


st.set_page_config(page_title="AI Well-being Coach", page_icon="💙")
init_logfile()

query_params = st.query_params
company = query_params.get("company", [""])[0] if isinstance(query_params.get("company"), list) else query_params.get("company", "")
uid = query_params.get("uid", ["demo"])[0] if isinstance(query_params.get("uid"), list) else query_params.get("uid", "demo")
preferred_lang = query_params.get("lang", [""])[0] if isinstance(query_params.get("lang"), list) else query_params.get("lang", "")

if preferred_lang.lower() in ["en","fa","nl"]:
    lang = "English" if preferred_lang.lower()=="en" else ("فارسی" if preferred_lang.lower()=="fa" else "English")

# --------------------------
# Sidebar: language + reset
# --------------------------
lang = st.sidebar.radio(
    "🌐 انتخاب زبان / Language / Taal",
    ["فارسی", "English", "Nederlands"],
    horizontal=True
)

# --- Mini analytics + Export ---
st.sidebar.markdown("### Activity")

try:
    if LOGFILE.exists():
        df = pd.read_csv(LOGFILE)
        total = len(df)
        st.sidebar.write(f"Total interactions: **{total}**")

        last_bot = df[(df["role"] == "assistant") & df["mood"].notna()].copy()
        last_mood_value = None
        if not last_bot.empty:
            last_bot["mood"] = pd.to_numeric(last_bot["mood"], errors="coerce")
            lm = last_bot["mood"].dropna().tail(1)
            if not lm.empty:
                last_mood_value = int(lm.values[0])
        st.sidebar.write(f"Last suggested mood: **{last_mood_value or '—'}**")

        try:
            df["ts_utc"] = pd.to_datetime(df["ts_utc"], errors="coerce")
            trend = df.set_index("ts_utc").resample("D")["role"].count()
            st.sidebar.line_chart(trend)
        except Exception:
            pass

        st.sidebar.download_button(
            label="Download CSV",
            data=LOGFILE.read_bytes(),
            file_name="checkins.csv",
            mime="text/csv"
        )
    else:
        st.sidebar.write("No interactions yet.")
except Exception as e:
    st.sidebar.warning(f"Log preview unavailable: {e}")

if st.sidebar.button("Reset / شروع دوباره"):
    for k in ("messages", "mood", "consent", "exercise_given", "mood_section_shown", "welcomed", "closed_once", "ask_feeling_shown"):
        if k in st.session_state: 
            del st.session_state[k]
    st.rerun()

# --------------------------
# Translations
# --------------------------
T = {
    "English": {
        "title": "Your mental health, 5 minutes a day",
        "disclaimer": ("I'm an AI well-being coach, not a substitute for professional care. "
                       "If you're in immediate danger, contact local emergency services."),
        "intro": "Hi! I'm your AI well-being companion. May I support you today?",
        "consent": "I agree / موافقم",
        "ask_mood": "On a scale of 1–5, how do you feel right now?",
        "ask_text": "Tell me one line about your current feeling.",
        "crisis": ("I'm sensing you might be going through something very difficult. "
                   "Your safety matters. If you're in immediate danger, contact local emergency services. "
                   "Would you like a quick grounding exercise now?"),
        "exercise_ground": ("2-minute grounding: name 5 things you see, 4 you feel, "
                            "3 you hear, 2 you smell, 1 you taste. Ready?"),
        "exercise_breath": ("Box breathing (×5): inhale 4s, hold 4s, exhale 4s, hold 4s. "
                            "Keep shoulders relaxed."),
        "exercise_grat": ("Gratitude note: write 3 small wins or tiny nice moments from today."),
        "summary": "Thanks for checking in. I'm here anytime. Would you like a daily reminder?",
        "mood_label": "Mood",
        "placeholder": "Type here...",
    },
    "فارسی": {
        "title": "سلامت روانت، روزی ۵ دقیقه",
        "disclaimer": ("من یک همراه هوشمند رفاه روان هستم و جایگزین درمانگر نیستم. "
                       "اگر در خطر فوری هستی، با خدمات اورژانسی تماس بگیر."),
        "intro": "سلام! من همراه دیجیتال سلامت روان تو هستم. امروز کمکت کنم؟",
        "consent": "موافقم / I agree",
        "ask_mood": "از ۱ تا ۵ الان حالت چطوره؟",
        "ask_text": "یک جمله درباره حال و هوای الان بنویس.",
        "crisis": ("انگار شرایط خیلی سختی رو تجربه می‌کنی. امنیتت مهمه. "
                   "اگر خطری هست، لطفاً فوراً با اورژانس تماس بگیر. "
                   "دوست داری یک تمرین گراندینگ سریع انجام بدیم؟"),
        "exercise_ground": ("گراندینگ ۲ دقیقه‌ای: ۵ چیز که می‌بینی، ۴ چیز که لمس می‌کنی، "
                            "۳ صدایی که می‌شنوی، ۲ بویی که حس می‌کنی، ۱ مزه‌ای که می‌چشی."),
        "exercise_breath": ("تنفس جعبه‌ای (۵ بار): ۴ ثانیه دم، ۴ ثانیه نگه‌دار، ۴ ثانیه بازدم، ۴ ثانیه نگه‌دار. "
                            "شانه‌ها رها."),
        "exercise_grat": ("قدردانی: ۳ موفقیت کوچک یا لحظه‌ی خوب امروزت رو بنویس."),
        "summary": "مرسی از چک‌این امروز. هر زمان خواستی اینجا هستم. یادآوری روزانه می‌خوای؟",
        "mood_label": "حالت",
        "placeholder": "اینجا بنویس...",
    },
    "Nederlands": {
        "title": "Je mentale gezondheid, 5 minuten per dag",
        "disclaimer": ("Ik ben een AI-welzijnscoach, geen vervanging voor professionele zorg. "
                       "Bel bij direct gevaar de hulpdiensten."),
        "intro": "Hoi! Ik ben je AI-welzijnscoach. Mag ik je vandaag ondersteunen?",
        "consent": "Ik ga akkoord / I agree",
        "ask_mood": "Op een schaal van 1-5, hoe voel je je nu?",
        "ask_text": "Vertel me in één zin hoe je je nu voelt.",
        "crisis": ("Ik merk dat je misschien iets heel moeilijks doormaakt. "
                   "Je veiligheid is belangrijk. Bel bij direct gevaar de hulpdiensten. "
                   "Wil je nu een korte grondings oefening doen?"),
        "exercise_ground": ("2-min gronding: noem 5 dingen die je ziet, 4 die je voelt, "
                            "3 die je hoort, 2 die je ruikt, 1 die je proeft."),
        "exercise_breath": ("Box breathing (×5): 4s in, 4s vasthouden, 4s uit, 4s vasthouden. "
                            "Houd je schouders ontspannen."),
        "exercise_grat": ("Dankbaarheid: schrijf 3 kleine overwinningen of fijne momenten van vandaag op."),
        "summary": "Bedankt voor je check-in. Ik ben er altijd. Wil je een dagelijkse herinnering?",
        "mood_label": "Stemming",
        "placeholder": "Typ hier...",
    }
}

# --------------------------
# Exercise selection
# --------------------------
def choose_response(mood: int | None, l: str):
    if not mood:
        return T[l]["ask_mood"]
    if mood <= 2:
        return T[l]["exercise_ground"]
    if mood == 3:
        return T[l]["exercise_breath"]
    return T[l]["exercise_grat"]

# --------------------------
# Session init
# --------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "consent" not in st.session_state:
    st.session_state.consent = False
if "mood" not in st.session_state:
    st.session_state.mood = None
if "welcomed" not in st.session_state:
    st.session_state.welcomed = False
if "mood_section_shown" not in st.session_state:
    st.session_state.mood_section_shown = False
if "exercise_given" not in st.session_state:
    st.session_state.exercise_given = False
if "closed_once" not in st.session_state:
    st.session_state.closed_once = False
if "ask_feeling_shown" not in st.session_state:
    st.session_state.ask_feeling_shown = False

# --------------------------
# UI: header + disclaimer
# --------------------------
st.title(T[lang]["title"])
st.caption(T[lang]["disclaimer"])

# Consent gate
if not st.session_state.get("consent", False):
    st.info(T[lang]["intro"])
    if st.button(T[lang]["consent"]):
        st.session_state.consent = True
        st.session_state.welcomed = True
        st.session_state.mood_section_shown = False
        st.session_state.exercise_given = False
        st.session_state.ask_feeling_shown = False
        st.session_state.messages = []
        st.rerun()
    st.stop()

# Mood selection section
if st.session_state.get("consent", False) and not st.session_state.get("mood_section_shown", False):
    st.markdown("### " + T[lang]["mood_label"])
    cols = st.columns(5)
    for i, c in enumerate(cols, start=1):
        if c.button(str(i), key=f"mood_{i}"):
            st.session_state.mood = i
            st.session_state.mood_section_shown = True

            if not st.session_state.get("exercise_given", False):
                st.session_state.exercise_given = True
                exercise = choose_response(st.session_state.mood, lang)
                st.session_state.messages.append({"role": "assistant", "text": exercise})
                log_event("assistant", exercise, user_id=uid, lang_code=lang, mood=i)

            if not st.session_state.get("ask_feeling_shown", False):
                st.session_state.ask_feeling_shown = True
                ask_feeling = T[lang]["ask_text"]
                st.session_state.messages.append({"role": "assistant", "text": ask_feeling})
                log_event("assistant", ask_feeling, user_id=uid, lang_code=lang, mood=i)

            st.rerun()

# Render chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["text"])

# Chat input (LLM-enabled)
user_text = st.chat_input(T[lang]["placeholder"])
if user_text:
    st.session_state.messages.append({"role": "user", "text": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)
    log_event("user", user_text, user_id=uid, lang_code=lang, mood=st.session_state.get("mood"))

    if not st.session_state.get("mood_section_shown", False):
        st.session_state.mood_section_shown = True

    ui_l = lang_code_of(user_text, fallback=("fa" if lang == "فارسی" else ("nl" if lang == "Nederlands" else "en")))
    lang_short = "fa" if ui_l == "fa" else ("nl" if ui_l == "nl" else "en")

    if is_crisis_text(user_text, lang_short):
        crisis_reply = (
            (DISCLAIMER_FA if ui_l == "fa" else DISCLAIMER_NL if ui_l == "nl" else DISCLAIMER_EN)
            + "\n\n"
            + (EX_LIB["grounding"]["fa"] if ui_l == "fa" else EX_LIB["grounding"]["nl"] if ui_l == "nl" else EX_LIB["grounding"]["en"])
        )
        with st.chat_message("assistant"):
            st.markdown(crisis_reply)
        st.session_state.messages.append({"role": "assistant", "text": crisis_reply})
        log_event("assistant", crisis_reply, user_id=uid, lang_code=lang, mood=st.session_state.get("mood"))

    else:
        hist = [{"role": m["role"], "content": m["text"]} for m in st.session_state.messages[-6:]]
        reply_text = llm_support(hist, st.session_state.get("mood"), ui_l)

        with st.chat_message("assistant"):
            st.markdown(reply_text)
        st.session_state.messages.append({"role": "assistant", "text": reply_text})
        log_event("assistant", reply_text, user_id=uid, lang_code=lang, mood=st.session_state.get("mood"))

        if not st.session_state.get("closed_once", False):
            closing = T[lang]["summary"]
            with st.chat_message("assistant"):
                st.markdown(closing)
            st.session_state.messages.append({"role": "assistant", "text": closing})
            log_event("assistant", closing, user_id=uid, lang_code=lang, mood=st.session_state.get("mood"))
            st.session_state.closed_once = True

    st.rerun()

