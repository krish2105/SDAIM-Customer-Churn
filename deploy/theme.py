"""Design tokens and scoped CSS for the churn intelligence application.

The palette is token-based: every colour used by the interface — including the
matplotlib charts — is resolved from one of the two token sets below, so light
and dark render as one coherent system rather than one theme with the other
patched on afterwards.

No external font, stylesheet or script is referenced. The Hugging Face Docker
Space runs without outbound network access at request time, and a CDN reference
would also be a supply-chain dependency, so the type stack is the platform's own
and all CSS is inlined here.

Contrast was checked against WCAG 2.1 AA (4.5:1 for body text, 3:1 for large
text and UI boundaries) in both token sets.
"""

from __future__ import annotations

from typing import Any

#: Semantic colour tokens. Keys are identical across themes so any consumer can
#: swap the whole set without branching on the theme name.
TOKENS: dict[str, dict[str, str]] = {
    "light": {
        "bg": "#F4F6FA",
        "bg_elevated": "#FFFFFF",
        "surface": "#FFFFFF",
        "surface_muted": "#EDF1F8",
        "text": "#0F172A",
        "text_muted": "#4A5A72",
        "text_subtle": "#64748B",
        "border": "#D7DFEC",
        "border_strong": "#B8C5D9",
        "primary": "#1E40AF",
        "primary_hover": "#1B379B",
        "on_primary": "#FFFFFF",
        "accent": "#B45309",
        "focus": "#1E40AF",
        "risk_low": "#15803D",
        "risk_low_soft": "#E4F5EA",
        "risk_medium": "#B45309",
        "risk_medium_soft": "#FBF0DF",
        "risk_high": "#B4232B",
        "risk_high_soft": "#FBE7E8",
        "chart_grid": "#DDE4EF",
        "chart_series": "#1E40AF",
        "chart_reference": "#64748B",
        "shadow": "0 1px 2px rgba(15,23,42,0.05), 0 8px 24px -12px rgba(15,23,42,0.18)",
    },
    "dark": {
        "bg": "#0A1018",
        "bg_elevated": "#121C29",
        "surface": "#121C29",
        "surface_muted": "#182536",
        "text": "#E9EFF7",
        "text_muted": "#A8B8CC",
        "text_subtle": "#8496AD",
        "border": "#26364C",
        "border_strong": "#3A4E69",
        "primary": "#7EA6F5",
        "primary_hover": "#9BBCFA",
        "on_primary": "#08111D",
        "accent": "#F0B355",
        "focus": "#7EA6F5",
        "risk_low": "#5FD08A",
        "risk_low_soft": "#12281D",
        "risk_medium": "#F0B355",
        "risk_medium_soft": "#2C2113",
        "risk_high": "#F2848B",
        "risk_high_soft": "#2E1518",
        "chart_grid": "#243247",
        "chart_series": "#7EA6F5",
        "chart_reference": "#8496AD",
        "shadow": "0 1px 2px rgba(0,0,0,0.4), 0 12px 28px -14px rgba(0,0,0,0.7)",
    },
}

#: 4 px spacing rhythm, dashboard density.
SPACING: dict[str, str] = {
    "xs": "4px",
    "sm": "8px",
    "md": "12px",
    "lg": "16px",
    "xl": "24px",
    "2xl": "32px",
}

FONT_STACK: str = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", '
    'Arial, "Noto Sans", sans-serif'
)
MONO_STACK: str = 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace'


def tokens_for(theme: str) -> dict[str, str]:
    """Return the token set for ``"light"`` or ``"dark"``."""
    return TOKENS.get(theme, TOKENS["light"])


def risk_colours(theme: str, tier: str) -> tuple[str, str]:
    """``(foreground, soft background)`` for a risk tier."""
    tokens = tokens_for(theme)
    key = {"Low": "risk_low", "Medium": "risk_medium", "High": "risk_high"}[tier]
    return tokens[key], tokens[f"{key}_soft"]


def matplotlib_rc(theme: str) -> dict[str, Any]:
    """Matplotlib rcParams that keep charts inside the same token system."""
    tokens = tokens_for(theme)
    return {
        "figure.facecolor": tokens["surface"],
        "axes.facecolor": tokens["surface"],
        "savefig.facecolor": tokens["surface"],
        "text.color": tokens["text"],
        "axes.labelcolor": tokens["text_muted"],
        "axes.edgecolor": tokens["border"],
        "xtick.color": tokens["text_muted"],
        "ytick.color": tokens["text_muted"],
        "grid.color": tokens["chart_grid"],
        "axes.grid": True,
        "grid.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9.5,
    }


def build_css(theme: str) -> str:
    """Return the complete scoped stylesheet for the active theme.

    Only presentational properties are set. No selector hides focus rings, and
    every interactive surface keeps a visible focus ring at 2 px.
    """
    t = tokens_for(theme)
    scheme = "dark" if theme == "dark" else "light"

    return f"""
<style>
:root {{
  --cci-bg: {t['bg']};
  --cci-surface: {t['surface']};
  --cci-surface-muted: {t['surface_muted']};
  --cci-text: {t['text']};
  --cci-text-muted: {t['text_muted']};
  --cci-text-subtle: {t['text_subtle']};
  --cci-border: {t['border']};
  --cci-border-strong: {t['border_strong']};
  --cci-primary: {t['primary']};
  --cci-primary-hover: {t['primary_hover']};
  --cci-on-primary: {t['on_primary']};
  --cci-accent: {t['accent']};
  --cci-focus: {t['focus']};
  --cci-shadow: {t['shadow']};
  --cci-radius: 10px;
  --cci-space-sm: {SPACING['sm']};
  --cci-space-md: {SPACING['md']};
  --cci-space-lg: {SPACING['lg']};
  --cci-space-xl: {SPACING['xl']};
  color-scheme: {scheme};
}}

/* ---------- Base surfaces ---------- */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
  background: var(--cci-bg);
  color: var(--cci-text);
}}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{
  background: var(--cci-surface);
  border-right: 1px solid var(--cci-border);
}}
/* Set the type stack on the root only. Applying it to span/div as well would
   also hit Streamlit's icon spans, whose glyphs come from an icon font — the
   ligature name would then render as literal text. */
.stApp {{ font-family: {FONT_STACK}; }}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {{
  color: var(--cci-text);
  letter-spacing: -0.015em;
  font-weight: 650;
}}
.stApp p, .stApp li {{ color: var(--cci-text-muted); line-height: 1.55; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1320px; }}

/* ---------- Widget labels and inputs ---------- */
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label {{
  color: var(--cci-text) !important;
  font-size: 0.82rem;
  font-weight: 560;
}}
/* Select control. This Streamlit build exposes no BaseWeb attributes, so the
   control is addressed by the element that owns the combobox input. */
:is(.stSelectbox, .stMultiSelect) :has(> [role="combobox"]) {{
  background: var(--cci-surface) !important;
  border: 1px solid var(--cci-border-strong) !important;
  border-radius: 8px !important;
  min-height: 44px;
}}
[role="combobox"] {{ background: transparent !important; color: var(--cci-text) !important; }}
:is(.stSelectbox, .stMultiSelect) svg {{ fill: var(--cci-text-muted); }}
[role="listbox"], [data-testid="stSelectboxVirtualDropdown"] {{
  background: var(--cci-surface) !important;
  border: 1px solid var(--cci-border) !important;
}}
[role="option"] {{ background: transparent !important; color: var(--cci-text) !important; }}
[role="option"]:hover, [role="option"][aria-selected="true"] {{
  background: var(--cci-surface-muted) !important;
}}

/* Number and text inputs: the visible surface is the container, not the field. */
[data-testid="stNumberInputContainer"], .stTextInput :has(> input) {{
  background: var(--cci-surface) !important;
  border: 1px solid var(--cci-border-strong) !important;
  border-radius: 8px !important;
  min-height: 44px;
}}
[data-testid="stNumberInputField"], .stTextInput input {{
  background: transparent !important;
  color: var(--cci-text) !important;
  border: none !important;
  font-variant-numeric: tabular-nums;
  font-family: {MONO_STACK};
}}
[data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"] {{
  background: var(--cci-surface-muted) !important;
  color: var(--cci-text-muted) !important;
  border-color: var(--cci-border) !important;
}}
:is(input, select, textarea, button, [role="button"], [data-baseweb="select"] > div):focus-visible {{
  outline: 2px solid var(--cci-focus) !important;
  outline-offset: 2px !important;
}}
[data-testid="stWidgetLabel"] + div [disabled],
input:disabled {{ opacity: 0.55 !important; cursor: not-allowed !important; }}

/* ---------- Buttons ---------- */
.stButton > button {{
  border-radius: 8px;
  border: 1px solid var(--cci-border-strong);
  background: var(--cci-surface);
  color: var(--cci-text);
  font-weight: 560;
  min-height: 44px;
  transition: background 160ms ease, border-color 160ms ease, transform 160ms ease;
  cursor: pointer;
}}
.stButton > button:hover {{ border-color: var(--cci-primary); color: var(--cci-primary); }}
.stButton > button * {{ color: inherit !important; }}
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {{
  background: var(--cci-primary) !important;
  border-color: var(--cci-primary) !important;
  color: var(--cci-on-primary) !important;
}}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {{
  background: var(--cci-primary-hover) !important;
  border-color: var(--cci-primary-hover) !important;
  color: var(--cci-on-primary) !important;
}}
@media (prefers-reduced-motion: reduce) {{
  .stButton > button {{ transition: none; }}
}}

/* ---------- Expander / tabs / alerts ---------- */
[data-testid="stExpander"] {{
  background: var(--cci-surface);
  border: 1px solid var(--cci-border);
  border-radius: var(--cci-radius);
}}
[data-testid="stExpander"] summary {{ color: var(--cci-text) !important; font-weight: 560; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid var(--cci-border); }}
.stTabs [data-baseweb="tab"] {{
  color: var(--cci-text-muted);
  border-radius: 8px 8px 0 0;
  padding: 8px 14px;
}}
.stTabs [aria-selected="true"] {{ color: var(--cci-primary) !important; font-weight: 600; }}
[data-testid="stAlert"] {{ border-radius: 8px; border: 1px solid var(--cci-border); }}
hr {{ border-color: var(--cci-border); }}
code {{ font-family: {MONO_STACK}; color: var(--cci-text); background: var(--cci-surface-muted); }}

/* ---------- Application components ---------- */
.cci-header {{
  border: 1px solid var(--cci-border);
  border-left: 4px solid var(--cci-primary);
  background: var(--cci-surface);
  border-radius: var(--cci-radius);
  padding: var(--cci-space-xl);
  box-shadow: var(--cci-shadow);
  margin-bottom: var(--cci-space-lg);
}}
.cci-header h1 {{ font-size: 1.6rem; margin: 0 0 6px 0; line-height: 1.25; }}
.cci-header .cci-purpose {{ color: var(--cci-text-muted); margin: 0; font-size: 0.96rem; }}
.cci-badges {{ display: flex; flex-wrap: wrap; gap: var(--cci-space-sm); margin-top: var(--cci-space-md); }}
.cci-badge {{
  display: inline-flex; align-items: center; gap: 6px;
  border: 1px solid var(--cci-border-strong);
  background: var(--cci-surface-muted);
  color: var(--cci-text-muted);
  border-radius: 999px;
  padding: 4px 11px;
  font-size: 0.74rem;
  font-weight: 560;
  letter-spacing: 0.01em;
}}
.cci-badge strong {{ color: var(--cci-text); font-weight: 640; }}

.cci-notice {{
  border: 1px solid var(--cci-border);
  border-left: 4px solid var(--cci-accent);
  background: var(--cci-surface-muted);
  border-radius: 8px;
  padding: var(--cci-space-md) var(--cci-space-lg);
  color: var(--cci-text-muted);
  font-size: 0.85rem;
  line-height: 1.5;
  margin-bottom: var(--cci-space-lg);
}}
.cci-notice strong {{ color: var(--cci-text); }}

.cci-kpis {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: var(--cci-space-md); }}
@media (max-width: 900px) {{ .cci-kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
@media (max-width: 520px) {{ .cci-kpis {{ grid-template-columns: 1fr; }} }}
.cci-kpi {{
  background: var(--cci-surface);
  border: 1px solid var(--cci-border);
  border-radius: var(--cci-radius);
  padding: var(--cci-space-lg);
  box-shadow: var(--cci-shadow);
}}
.cci-kpi .cci-kpi-label {{
  font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--cci-text-subtle); font-weight: 620; margin-bottom: 6px;
}}
.cci-kpi .cci-kpi-value {{
  font-size: 1.34rem; font-weight: 680; color: var(--cci-text);
  font-variant-numeric: tabular-nums; line-height: 1.2;
}}
.cci-kpi .cci-kpi-note {{ font-size: 0.74rem; color: var(--cci-text-subtle); margin-top: 4px; }}

/* Bordered st.container() is the section card. Styling the native wrapper keeps
   the widgets genuinely inside the card, which a hand-written div cannot do. */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > div > .cci-section-title),
[data-testid="stVerticalBlockBorderWrapper"]:has(.cci-section-title) {{
  background: var(--cci-surface);
  border: 1px solid var(--cci-border) !important;
  border-radius: var(--cci-radius) !important;
  box-shadow: var(--cci-shadow);
  padding: var(--cci-space-lg) !important;
  margin-bottom: var(--cci-space-md);
}}
.cci-section-title {{
  display: flex; align-items: baseline; justify-content: space-between; gap: 12px;
  font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.07em;
  color: var(--cci-primary); font-weight: 660;
  padding-bottom: var(--cci-space-sm); margin-bottom: var(--cci-space-md);
  border-bottom: 1px solid var(--cci-border);
}}
.cci-section-title span.cci-section-hint {{
  text-transform: none; letter-spacing: 0; font-weight: 450;
  color: var(--cci-text-subtle); font-size: 0.74rem;
}}

.cci-result {{
  border-radius: var(--cci-radius);
  border: 1px solid var(--cci-border);
  background: var(--cci-surface);
  box-shadow: var(--cci-shadow);
  overflow: hidden;
}}
.cci-result-head {{ padding: var(--cci-space-lg); border-bottom: 1px solid var(--cci-border); }}
.cci-result-tier {{
  display: inline-flex; align-items: center; gap: 8px;
  border-radius: 999px; padding: 5px 13px;
  font-size: 0.78rem; font-weight: 680; letter-spacing: 0.02em;
}}
.cci-result-prob {{
  font-size: 2.5rem; font-weight: 700; line-height: 1.1;
  font-variant-numeric: tabular-nums; margin: var(--cci-space-md) 0 2px;
}}
.cci-result-caption {{ color: var(--cci-text-subtle); font-size: 0.78rem; margin: 0; }}
.cci-result-body {{ padding: var(--cci-space-lg); }}
.cci-result-body dl {{ margin: 0; }}
.cci-result-row {{
  display: flex; justify-content: space-between; gap: 12px;
  padding: 7px 0; border-bottom: 1px dashed var(--cci-border);
  font-size: 0.85rem;
}}
.cci-result-row:last-child {{ border-bottom: none; }}
.cci-result-row dt {{ color: var(--cci-text-subtle); }}
.cci-result-row dd {{ margin: 0; color: var(--cci-text); font-weight: 580; text-align: right; }}
.cci-action {{
  margin-top: var(--cci-space-md);
  padding: var(--cci-space-md);
  border-radius: 8px;
  border: 1px solid var(--cci-border);
  background: var(--cci-surface-muted);
  font-size: 0.86rem;
  color: var(--cci-text-muted);
  line-height: 1.5;
}}
.cci-action strong {{ color: var(--cci-text); }}

.cci-empty {{
  border: 1px dashed var(--cci-border-strong);
  border-radius: var(--cci-radius);
  background: var(--cci-surface);
  padding: var(--cci-space-xl);
  text-align: center;
  color: var(--cci-text-subtle);
  font-size: 0.88rem;
  line-height: 1.6;
}}
.cci-empty strong {{ display: block; color: var(--cci-text); font-size: 1rem; margin-bottom: 6px; }}

.cci-footnote {{
  color: var(--cci-text-subtle);
  font-size: 0.76rem;
  line-height: 1.55;
  border-top: 1px solid var(--cci-border);
  padding-top: var(--cci-space-md);
  margin-top: var(--cci-space-xl);
}}
.cci-version-caption {{
  display: inline-block;
  font-family: {MONO_STACK};
  font-size: 0.74rem;
  color: var(--cci-text-subtle);
  border: 1px solid var(--cci-border);
  border-radius: 6px;
  padding: 3px 9px;
  background: var(--cci-surface-muted);
}}
</style>
"""
