from __future__ import annotations

import html
import re
import time

import streamlit as st

from model.classifier import predict_email


st.set_page_config(page_title="Email Threat Intelligence Scanner", layout="wide")


def _render_styles() -> None:
    # Keep CSS minimal and theme-friendly; Streamlit handles most light/dark differences.
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at 12% 18%, rgba(255, 198, 164, 0.18) 0, transparent 33%),
                    radial-gradient(circle at 88% 12%, rgba(126, 187, 255, 0.15) 0, transparent 30%);
            }
            .chip {
                display: inline-block;
                padding: 0.18rem 0.5rem;
                border-radius: 999px;
                font-size: 0.82rem;
                font-weight: 700;
                margin-right: 0.35rem;
                border: 1px solid rgba(128, 128, 128, 0.35);
            }
            .chip.fraud { background: rgba(255, 120, 95, 0.18); }
            .chip.injection { background: rgba(95, 144, 255, 0.2); }
            .summary-box {
                border: 1px solid rgba(128, 128, 128, 0.35);
                border-radius: 14px;
                padding: 0.9rem 1rem;
                margin: 0.6rem 0 0.9rem 0;
                background: rgba(255, 255, 255, 0.03);
            }
            .summary-title {
                font-size: 0.9rem;
                font-weight: 700;
                opacity: 0.85;
                margin-bottom: 0.35rem;
                text-transform: uppercase;
                letter-spacing: 0.04em;
            }
            .summary-title.big {
                font-size: 1.25rem;
                font-weight: 900;
                opacity: 1;
                margin-bottom: 0.6rem;
                letter-spacing: 0.08em;
            }
            .summary-primary {
                font-size: 1.4rem;
                font-weight: 800;
                margin-bottom: 0.15rem;
            }
            .summary-secondary {
                font-size: 0.95rem;
                opacity: 0.8;
            }
            .prob-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.65rem;
                margin-top: 0.35rem;
            }
            .prob-card {
                border: 1px solid rgba(128, 128, 128, 0.35);
                border-radius: 12px;
                padding: 0.75rem 0.85rem;
                background: rgba(255, 255, 255, 0.02);
            }
            .prob-card.highest {
                border-color: rgba(255, 193, 7, 0.7);
                box-shadow: 0 0 0 1px rgba(255, 193, 7, 0.18) inset;
            }
            .prob-label {
                font-size: 0.86rem;
                font-weight: 700;
                opacity: 0.8;
            }
            .prob-value {
                font-size: 1.7rem;
                font-weight: 800;
                margin-top: 0.1rem;
            }
            .prob-hint {
                font-size: 0.78rem;
                opacity: 0.7;
                margin-top: 0.15rem;
            }
            .highlight-box {
                border: 1px solid rgba(128, 128, 128, 0.35);
                border-radius: 10px;
                padding: 0.8rem;
                white-space: pre-wrap;
                line-height: 1.5;
            }
            .mark-fraud {
                background: rgba(255, 120, 95, 0.23);
                border-radius: 0.3rem;
                padding: 0.03rem 0.2rem;
                font-weight: 700;
            }
            .mark-injection {
                background: rgba(95, 144, 255, 0.28);
                border-radius: 0.3rem;
                padding: 0.03rem 0.2rem;
                font-weight: 700;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_state() -> None:
    if "scan_result" not in st.session_state:
        st.session_state.scan_result = None
    if "scan_error" not in st.session_state:
        st.session_state.scan_error = ""


def _highlight_text(text: str, fraud_matches: list[str], injection_matches: list[str]) -> str:
    if not text:
        return ""

    term_type: dict[str, str] = {}
    for term in fraud_matches:
        term_type[term.lower()] = "fraud"
    for term in injection_matches:
        term_type[term.lower()] = "injection"

    if not term_type:
        return html.escape(text)

    sorted_terms = sorted(term_type.keys(), key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(term) for term in sorted_terms), re.IGNORECASE)

    output: list[str] = []
    cursor = 0
    for match in pattern.finditer(text):
        start, end = match.span()
        output.append(html.escape(text[cursor:start]))
        term = match.group(0)
        css = "mark-injection" if term_type.get(term.lower()) == "injection" else "mark-fraud"
        output.append(f'<span class="{css}">{html.escape(term)}</span>')
        cursor = end

    output.append(html.escape(text[cursor:]))
    return "".join(output)


def _format_pct(probability: float) -> str:
    """Format probability as a percentage with two decimal places."""
    pct = max(0.0, float(probability or 0.0) * 100)
    return f"{pct:.2f}%"


def _run_scan(title: str, body: str, progress_slot: st.delta_generator.DeltaGenerator) -> None:
    if not title.strip() and not body.strip():
        st.session_state.scan_error = "Enter at least a title or a body before scanning."
        st.session_state.scan_result = None
        return

    st.session_state.scan_error = ""
    with progress_slot.container():
        st.subheader("Scanning in progress")
        bar = st.progress(0)
        label = st.empty()
        for pct in range(0, 101, 10):
            bar.progress(pct)
            label.write(f"Progress: {pct}%")
            time.sleep(0.04)

    try:
        st.session_state.scan_result = predict_email(title=title, body=body)
    except FileNotFoundError as exc:
        st.session_state.scan_error = f"Dataset error: {exc}"
        st.session_state.scan_result = None
    except Exception as exc:  # pragma: no cover
        st.session_state.scan_error = f"Unexpected error: {exc}"
        st.session_state.scan_result = None
    finally:
        progress_slot.empty()


def _render_results(title: str, body: str) -> None:
    st.subheader("Scan results")

    if st.session_state.scan_error:
        st.error(st.session_state.scan_error)
        return

    result = st.session_state.scan_result
    if result is None:
        st.info("Results will appear here after you scan a message.")
        return

    safe_prob = result.get("is_safe", 0.0) or 0.0
    fraud_prob = result.get("is_fraud", 0.0) or 0.0
    injection_prob = result.get("is_injection", 0.0) or 0.0
    ranked = sorted(
        [("SAFE", safe_prob), ("FRAUD", fraud_prob), ("INJECTION", injection_prob)],
        key=lambda item: item[1],
        reverse=True,
    )
    top_label, top_prob = ranked[0]
    runner_label, runner_prob = ranked[1]

    st.markdown(
        f"""
        <div class="summary-box">
            <div class="summary-title big">OUTCOME</div>
            <div class="summary-primary">Main result: {top_label}</div>
            <div class="summary-secondary">Second strongest result: {runner_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    prob_cols = st.columns(3)
    for index, (label, probability) in enumerate(ranked):
        highest_class = "highest" if label == top_label else ""
        with prob_cols[index]:
            st.markdown(
                f"""
                <div class="prob-card {highest_class}">
                    <div class="prob-label">{label}</div>
                    <div class="prob-value">{_format_pct(probability)}</div>
                    <div class="prob-hint">Model confidence</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    fraud_matches = result.get("fraud_matches") or []
    injection_matches = result.get("injection_matches") or []
    flagged_sentences = result.get("flagged_sentences") or []

    st.write("Highlighted evidence")
    if title.strip():
        st.caption("Title")
        st.markdown(
            f'<div class="highlight-box">{_highlight_text(title, fraud_matches, injection_matches)}</div>',
            unsafe_allow_html=True,
        )
    if body.strip():
        st.caption("Body")
        st.markdown(
            f'<div class="highlight-box">{_highlight_text(body, fraud_matches, injection_matches)}</div>',
            unsafe_allow_html=True,
        )

    c1, c2 = st.columns(2)
    with c1:
        st.write("Matched fraud terms")
        st.write("\n".join(f"- {term}" for term in fraud_matches) if fraud_matches else "- none")
    with c2:
        st.write("Matched injection terms")
        st.write("\n".join(f"- {term}" for term in injection_matches) if injection_matches else "- none")

    st.write("Flagged sentences")
    st.write("\n".join(f"- {sentence}" for sentence in flagged_sentences) if flagged_sentences else "- none")

    with st.expander("Raw result / outcome"):
        st.json(result)


def main() -> None:
    _render_styles()
    _init_state()

    st.title("Email Threat Intelligence Scanner")
    st.caption("Hybrid Naive Bayes scoring with explicit fraud and injection signal tracing.")

    left, right = st.columns([1.1, 1])

    with left:
        title = st.text_input("Email title", placeholder="Urgent: Verify account activity")
        body = st.text_area("Email body", height=430, placeholder="Paste the message body here...")
        scan_clicked = st.button("Scan Message", use_container_width=True)

    progress_slot = right.empty()
    with right:
        if scan_clicked:
            _run_scan(title, body, progress_slot)
        _render_results(title, body)


if __name__ == "__main__":
    main()
