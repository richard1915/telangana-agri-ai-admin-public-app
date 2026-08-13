"""
Farmer Awareness Assistant (free-first, Groq -> Gemini -> static FAQ)
=======================================================================

A chat-style assistant that answers farmer/student questions about safe
and effective chemical use -- separate from the MOA/ML pipeline, which
optimizes doses but doesn't explain *why* a recommendation matters to
someone without an agriculture background.

Provider chain, in order:
  1. Groq (GROQ_API_KEY) -- free tier, no cost, tries several current
     model IDs in sequence. Get a free key at https://console.groq.com/keys
     Chosen as primary since it's free and (once configured) has been
     more reliable than Gemini's rapidly-changing model lineup.
  2. Gemini (GEMINI_API_KEY) -- tried as a secondary backup, also across
     several model IDs, in case Groq is unavailable/rate-limited.
  3. Static FAQ -- always available, no key required at all.

Both providers deprecate model IDs frequently (Groq retired
llama-3.3-70b-versatile and llama-3.1-8b-instant in June 2026; Google has
been retiring Gemini IDs ahead of their own announced shutdown dates) --
so both are configured as a list of candidates to try in order, not a
single hardcoded model, to reduce how often this breaks going forward.
"""

import os
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
_GROQ_MODEL_ENV = os.getenv("GROQ_MODEL", "")
# openai/gpt-oss-120b and -20b are Groq's current recommended general-purpose
# models as of mid-2026 (replacing the deprecated llama-3.3-70b-versatile /
# llama-3.1-8b-instant). Check https://console.groq.com/docs/models if these
# also start failing.
GROQ_MODEL_CANDIDATES = list(dict.fromkeys(
    ([_GROQ_MODEL_ENV] if _GROQ_MODEL_ENV else [])
    + ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
))
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
_GEMINI_MODEL_ENV = os.getenv("GEMINI_MODEL", "")
GEMINI_MODEL_CANDIDATES = list(dict.fromkeys(
    ([_GEMINI_MODEL_ENV] if _GEMINI_MODEL_ENV else [])
    + ["gemini-3.5-flash", "gemini-flash-latest", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
))

SYSTEM_PROMPT = (
    "You are a farmer awareness assistant for a precision-agriculture project in "
    "Telangana, India. Answer in plain, simple language a farmer or agriculture "
    "student would understand -- avoid jargon, keep answers short (3-6 sentences "
    "unless asked for detail), and focus on practical safety and correct usage of "
    "fertilizers and pesticides. Always mention when a question needs a real "
    "agronomist, doctor, or local agriculture extension officer instead of just "
    "an app -- especially for poisoning symptoms, medical emergencies, or anything "
    "with legal/regulatory implications (banned chemicals, export/import rules). "
    "Do not invent specific product names, dosages, or regulations you are not "
    "confident about -- say so plainly instead of guessing."
)

# Static fallback content -- always available, no API key required. Keeps
# the awareness feature useful even offline / without Gemini configured.
STATIC_FAQ = [
    {
        "question": "Why shouldn't I just use more fertilizer to be safe?",
        "answer": (
            "More isn't always better. Once soil already has enough nutrients, "
            "extra fertilizer doesn't raise yield much further -- it just runs off "
            "into groundwater and nearby water bodies, costs money, and can actually "
            "hurt soil health over time by changing its pH or salt balance."
        ),
    },
    {
        "question": "Is it safe to mix pesticide with less water to save money?",
        "answer": (
            "No -- pesticide labels specify a concentration for a reason. Using less "
            "water than recommended makes the mixture too strong (can damage the crop "
            "or be unsafe to handle); using more water than recommended dilutes it "
            "below the effective dose, so pests survive and you've wasted the "
            "application entirely. Always follow the label's water ratio."
        ),
    },
    {
        "question": "What should I do if I feel dizzy or unwell after spraying chemicals?",
        "answer": (
            "Stop spraying immediately, move to fresh air, remove contaminated "
            "clothing, and wash exposed skin with soap and water. If symptoms don't "
            "improve quickly, or if you swallowed anything, go to the nearest hospital "
            "or call a poison control helpline right away -- this app cannot give "
            "medical advice, and this is not something to wait out."
        ),
    },
    {
        "question": "How long should I wait between applying pesticide and harvesting?",
        "answer": (
            "This is called the 'pre-harvest interval' and it's different for every "
            "chemical and crop -- it's printed on the product label. Harvesting too "
            "early can leave unsafe residue on the crop. If the label is missing or "
            "unclear, ask your local agriculture extension officer rather than guess."
        ),
    },
    {
        "question": "Can I reuse empty pesticide containers for water or storage?",
        "answer": (
            "No, never. Even after rinsing, pesticide containers can retain enough "
            "residue to be dangerous. Triple-rinse them, puncture them so they can't "
            "be reused, and dispose of them according to local guidelines -- don't "
            "burn them or throw them in a water source."
        ),
    },
]


def is_gemini_configured():
    return bool(GEMINI_API_KEY)


def is_groq_configured():
    return bool(GROQ_API_KEY)


def _try_gemini(question, conversation_history):
    """Try each candidate Gemini model in order; return on first success."""
    contents = []
    for turn in (conversation_history or []):
        contents.append({
            "role": "user" if turn["role"] == "user" else "model",
            "parts": [{"text": turn["text"]}],
        })
    contents.append({"role": "user", "parts": [{"text": question}]})

    payload = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 400},
    }

    last_error = None
    for model in GEMINI_MODEL_CANDIDATES:
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        try:
            response = requests.post(
                endpoint,
                headers={"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY},
                json=payload,
                timeout=15,
            )
            if response.status_code == 404:
                last_error = f"'{model}' returned 404 (deprecated/unavailable) -- trying next model."
                print(f"Gemini: {last_error}")
                continue
            response.raise_for_status()
            data = response.json()
            answer = data["candidates"][0]["content"]["parts"][0]["text"]
            return {"available": True, "answer": answer.strip(), "provider": f"Gemini ({model})"}
        except Exception as e:
            last_error = str(e)
            print(f"Gemini model '{model}' error: {e}")
            continue

    return {"available": False, "error": last_error or "All Gemini models failed."}


def _try_groq(question, conversation_history):
    """Free provider -- OpenAI-compatible chat completions API. Tries each
    candidate model in order in case one has been deprecated/rate-limited."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (conversation_history or []):
        messages.append({
            "role": "user" if turn["role"] == "user" else "assistant",
            "content": turn["text"],
        })
    messages.append({"role": "user", "content": question})

    last_error = None
    for model in GROQ_MODEL_CANDIDATES:
        try:
            response = requests.post(
                GROQ_ENDPOINT,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {GROQ_API_KEY}"},
                json={"model": model, "messages": messages, "temperature": 0.4, "max_tokens": 400},
                timeout=15,
            )
            if response.status_code in (400, 404):
                # 400 with an unknown/decommissioned model, or 404 -- try the next candidate.
                last_error = f"'{model}' returned {response.status_code} (deprecated/unavailable) -- trying next model."
                print(f"Groq: {last_error}")
                continue
            response.raise_for_status()
            data = response.json()
            answer = data["choices"][0]["message"]["content"]
            return {"available": True, "answer": answer.strip(), "provider": f"Groq ({model})"}
        except Exception as e:
            last_error = str(e)
            print(f"Groq model '{model}' error: {e}")
            continue

    return {"available": False, "error": last_error or "All Groq models failed."}


def ask_awareness_bot(question, conversation_history=None):
    """
    Ask the awareness assistant a question, trying Groq first (free,
    reliable), then Gemini (across several model IDs, only if configured),
    then falling back to a static FAQ if neither is configured or both fail.

    conversation_history is an optional list of {"role": "user"|"model",
    "text": ...} dicts for multi-turn context.

    Returns {"available": True, "answer": ..., "provider": ...} on
    success, or {"available": False, "error": ..., "fallback_faq":
    STATIC_FAQ} if nothing worked -- callers should show the static FAQ
    in that case rather than leaving the user with nothing.
    """
    errors = []

    if is_groq_configured():
        result = _try_groq(question, conversation_history)
        if result.get("available"):
            return result
        errors.append(f"Groq: {result.get('error')}")

    if is_gemini_configured():
        result = _try_gemini(question, conversation_history)
        if result.get("available"):
            return result
        errors.append(f"Gemini: {result.get('error')}")

    if not errors:
        return {
            "available": False,
            "error": "No AI provider configured (set GEMINI_API_KEY or GROQ_API_KEY in .env) -- showing static awareness FAQ instead.",
            "fallback_faq": STATIC_FAQ,
        }

    return {
        "available": False,
        "error": "Could not get a response from any configured provider (" + "; ".join(errors) + ") -- showing static awareness FAQ instead.",
        "fallback_faq": STATIC_FAQ,
    }
