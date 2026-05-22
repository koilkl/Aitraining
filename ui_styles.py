def inject_teachable_style() -> None:
    import streamlit as st

    st.markdown(
        """
<style>
  .tm-title { font-size: 44px; font-weight: 800; margin: 8px 0 16px 0; }
  .tm-sub { color: rgba(0,0,0,.65); margin: -6px 0 24px 0; }
  .tm-card {
    border: 1px solid rgba(0,0,0,.08);
    border-radius: 14px;
    padding: 18px 18px 14px 18px;
    background: white;
    box-shadow: 0 6px 24px rgba(0,0,0,.08);
    height: 100%;
  }
  .tm-card h3 { margin: 0 0 6px 0; font-size: 20px; }
  .tm-card p { margin: 0 0 10px 0; color: rgba(0,0,0,.65); }
  .tm-badge {
    display: inline-block;
    font-size: 12px;
    padding: 3px 8px;
    border-radius: 999px;
    background: rgba(0,0,0,.06);
    color: rgba(0,0,0,.65);
    margin-left: 8px;
  }
  .tm-steps { margin: 8px 0 10px 0; color: rgba(0,0,0,.65); }
  .tm-step-on { font-weight: 700; color: rgba(0,0,0,.85); }
  .tm-kv { border-radius: 12px; padding: 10px 12px; background: rgba(0,0,0,.04); }
  .tm-panel { border: 1px solid rgba(0,0,0,.08); border-radius: 14px; padding: 16px; background: white; box-shadow: 0 8px 26px rgba(0,0,0,.07); }
</style>
        """,
        unsafe_allow_html=True,
    )
