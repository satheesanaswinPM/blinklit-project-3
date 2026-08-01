from pathlib import Path

p = Path("app.py")
t = p.read_text(encoding="utf-8")

old_css_start = t.find("          .flow-wrap {{")
old_css_end = t.find("          div[data-testid=\"stDataFrame\"] {{")
assert old_css_start > 0 and old_css_end > old_css_start

new_css = '''          .flow-wrap {{
            font-family: "Plus Jakarta Sans", sans-serif;
            color: {tok["ink"]} !important;
            max-width: 640px;
            margin: 0 auto;
          }}
          .simple-flow {{
            display: flex;
            flex-direction: column;
            align-items: stretch;
            gap: 0;
          }}
          .simple-step {{
            border: 1px solid {tok["line"]};
            border-radius: 16px;
            background: {tok["panel"]};
            padding: 1rem 1.15rem;
            box-shadow: var(--shadow);
          }}
          .simple-step .n {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 1.7rem;
            height: 1.7rem;
            border-radius: 999px;
            background: {tok["accent"]};
            color: #fff !important;
            font-family: "Sora", sans-serif;
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.45rem;
          }}
          .simple-step .t {{
            font-family: "Sora", sans-serif;
            font-size: 1.05rem;
            font-weight: 700;
            color: {tok["ink"]} !important;
            margin: 0 0 0.3rem 0;
          }}
          .simple-step .d {{
            color: {tok["muted"]} !important;
            font-size: 0.92rem;
            line-height: 1.45;
            margin: 0;
          }}
          .simple-step .pills {{
            margin-top: 0.55rem;
          }}
          .simple-arrow {{
            text-align: center;
            color: {tok["accent"]} !important;
            font-size: 1.25rem;
            font-weight: 700;
            line-height: 1;
            padding: 0.25rem 0;
          }}
          .flow-note {{
            margin-top: 0.85rem;
            color: {tok["muted"]} !important;
            font-size: 0.82rem;
            text-align: center;
          }}
          .flow-pill {{
            display: inline-block;
            margin: 0.15rem 0.25rem 0.15rem 0;
            padding: 0.18rem 0.55rem;
            border-radius: 999px;
            background: {tok["accent_soft"]};
            color: {tok["accent"]} !important;
            font-size: 0.75rem;
            font-weight: 700;
          }}
'''

t = t[:old_css_start] + new_css + t[old_css_end:]

start = t.find("def render_end_to_end_workflow")
end = t.find("\ndef render_methodology")
assert start > 0 and end > start

new_fn = r'''def render_end_to_end_workflow() -> None:
    """Simple, user-friendly downward flowchart of the research pipeline."""
    c = _workflow_counts()
    sources = (
        f"Play {c['play_raw']:,} · App Store {c['app_store_raw']:,} · "
        f"Reddit {c['reddit_raw']:,} · YouTube {c['youtube_raw']:,}"
    )

    def step(num: str, title: str, detail: str, pills: str = "") -> str:
        pill_html = f'<div class="pills">{pills}</div>' if pills else ""
        return f"""
    <div class="simple-step">
      <div class="n">{html.escape(num)}</div>
      <div class="t">{html.escape(title)}</div>
      <p class="d">{html.escape(detail)}</p>
      {pill_html}
    </div>"""

    arrow = '<div class="simple-arrow">↓</div>'

    html_block = f"""
<div class="flow-wrap">
  <div class="simple-flow">
    {step("1", "Collect feedback", "Gather what people say about Blinkit from app stores and online communities.", f'<span class="flow-pill">{html.escape(sources)}</span>')}
    {arrow}
    {step("2", "Combine into one dataset", "Merge every source into a single clean review list, without duplicates.", f'<span class="flow-pill">{c["merged"]:,} reviews ready</span>')}
    {arrow}
    {step("3", "Find patterns", "Spot recurring themes, sentiment, and shopper groups in the feedback.", f'<span class="flow-pill">{c["themes"]:,} themes</span><span class="flow-pill">{c["sentiment"]:,} sentiment labels</span><span class="flow-pill">{c["segments"]:,} segments</span>')}
    {arrow}
    {step("4", "Tag exploration barriers", "Flag reviews that explain why people stick to familiar categories.", f'<span class="flow-pill">{c["exploration_relevant"]:,} relevant of {c["exploration"]:,}</span>')}
    {arrow}
    {step("5", "Turn findings into actions", "Summarize barriers, jobs-to-be-done, experiments, and category opportunities.", f'<span class="flow-pill">{c["category_ops"]:,} category opportunities</span>')}
    {arrow}
    {step("6", "Explore in the dashboard", "Use Findings, Opportunities, and Validation to review evidence and decide what to test next.", '<span class="flow-pill">Findings Board</span><span class="flow-pill">Category Opportunities</span><span class="flow-pill">Validation Desk</span>')}
  </div>
  <div class="flow-note">Refresh data anytime with <code>python main.py --skip-collect</code>, then click Refresh in the sidebar.</div>
</div>
"""

    panel_start(
        "How the insight engine works",
        "A simple 6-step path from raw feedback to product decisions.",
    )
    st.markdown(html_block, unsafe_allow_html=True)
    panel_end()


'''

p.write_text(t[:start] + new_fn + t[end + 1 :], encoding="utf-8")
print("updated")
