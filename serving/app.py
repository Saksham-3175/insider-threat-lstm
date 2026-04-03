"""
Streamlit SOC dashboard for InsiderThreatLSTM.

Run from project root:
    streamlit run serving/app.py

Pages:
  1 — Overview          : metric cards, risk histogram, top-10 table
  2 — User Drill-Down   : temporal risk evolution, alert badge, threat window
  3 — Model Performance : confusion matrix, PR / ROC curves, key metrics
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE   = "http://localhost:8000"
_ROOT      = Path(__file__).parent.parent
_OUT       = _ROOT / "outputs"
HIGH_THRESH   = 0.7
MEDIUM_THRESH = 0.4

st.set_page_config(
    page_title="Insider Threat SOC",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── API helpers ───────────────────────────────────────────────────────────────
def _api(endpoint: str, timeout: int = 8):
    """GET request to API. Returns parsed JSON or None on failure."""
    try:
        r = requests.get(f"{API_BASE}{endpoint}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _api_online() -> bool:
    h = _api("/health")
    return h is not None and h.get("status") == "ok"


@st.cache_data(ttl=60)
def _fetch_all_scores():
    return _api("/users/scores")


@st.cache_data(ttl=60)
def _fetch_user_detail(user_id: str):
    return _api(f"/users/{user_id}")


@st.cache_data
def _load_eval_results():
    path = _OUT / "evaluation_results.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


# ── Shared components ─────────────────────────────────────────────────────────
def _alert_badge(level: str) -> str:
    colors = {"HIGH": "#d9534f", "MEDIUM": "#f0ad4e", "LOW": "#5cb85c"}
    c = colors.get(level, "#aaa")
    return f'<span style="background:{c};color:white;padding:3px 10px;border-radius:4px;font-weight:bold">{level}</span>'


def _offline_banner():
    st.error(
        "**API offline.** Start the server with:\n\n"
        "```\nuvicorn serving.api:app --reload --port 8000\n```",
        icon="🔴",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Page 1 — Overview
# ─────────────────────────────────────────────────────────────────────────────
def page_overview():
    st.title("🔍 Insider Threat Overview")

    online = _api_online()
    if not online:
        _offline_banner()
        return

    scores = _fetch_all_scores()
    if not scores:
        st.warning("No user scores available.")
        return

    df = pd.DataFrame(scores)

    # ── Metric cards ─────────────────────────────────────────────────────────
    total  = len(df)
    n_high = int((df["risk_score"] > HIGH_THRESH).sum())
    n_med  = int(((df["risk_score"] > MEDIUM_THRESH) & (df["risk_score"] <= HIGH_THRESH)).sum())
    n_low  = total - n_high - n_med
    ev     = _load_eval_results()
    prauc  = ev["checkpoint"]["val_prauc"] if ev else "N/A"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Users Monitored", total)
    c2.metric("🔴 HIGH Risk",   n_high)
    c3.metric("🟡 MEDIUM Risk", n_med)
    c4.metric("🟢 LOW Risk",    n_low)
    c5.metric("Val PR-AUC",     prauc)

    st.divider()

    col_left, col_right = st.columns([2, 1])

    # ── Risk histogram ────────────────────────────────────────────────────────
    with col_left:
        st.subheader("Risk Score Distribution")
        fig = px.histogram(
            df, x="risk_score", nbins=40,
            color="alert_level",
            color_discrete_map={"HIGH": "#d9534f", "MEDIUM": "#f0ad4e", "LOW": "#5cb85c"},
            labels={"risk_score": "Risk Score", "alert_level": "Alert Level"},
            category_orders={"alert_level": ["HIGH", "MEDIUM", "LOW"]},
        )
        fig.add_vline(x=HIGH_THRESH,   line_dash="dash", line_color="#d9534f",
                      annotation_text="HIGH",   annotation_position="top right")
        fig.add_vline(x=MEDIUM_THRESH, line_dash="dash", line_color="#f0ad4e",
                      annotation_text="MEDIUM", annotation_position="top right")
        fig.update_layout(margin=dict(t=20, b=20), height=350, bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)

    # ── Alert level pie ───────────────────────────────────────────────────────
    with col_right:
        st.subheader("Alert Breakdown")
        pie = px.pie(
            values=[n_high, n_med, n_low],
            names=["HIGH", "MEDIUM", "LOW"],
            color=["HIGH", "MEDIUM", "LOW"],
            color_discrete_map={"HIGH": "#d9534f", "MEDIUM": "#f0ad4e", "LOW": "#5cb85c"},
        )
        pie.update_layout(margin=dict(t=20, b=20), height=350)
        st.plotly_chart(pie, use_container_width=True)

    # ── Top 10 table ──────────────────────────────────────────────────────────
    st.subheader("Top 10 Highest-Risk Users")
    top10 = df.head(10)[
        ["user_id", "risk_score", "alert_level", "n_windows",
         "n_alerts", "n_positive_truth", "is_threat_user"]
    ].copy()
    top10.columns = ["User", "Risk Score", "Alert", "Windows",
                     "Alerts Fired", "True Positives", "Known Threat"]

    def _color_row(row):
        c = {"HIGH": "#fde8e8", "MEDIUM": "#fef3e2", "LOW": "#eaf7ea"}.get(row["Alert"], "")
        return [f"background-color: {c}"] * len(row)

    st.dataframe(
        top10.style.apply(_color_row, axis=1).format({"Risk Score": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Page 2 — User Drill-Down
# ─────────────────────────────────────────────────────────────────────────────
def page_drilldown():
    st.title("🔎 User Drill-Down")

    online = _api_online()
    if not online:
        _offline_banner()
        return

    scores = _fetch_all_scores()
    if not scores:
        st.warning("No user scores available.")
        return

    df     = pd.DataFrame(scores)
    users  = df.sort_values("risk_score", ascending=False)["user_id"].tolist()
    labels = {
        r["user_id"]: f"{r['user_id']}  [{r['alert_level']}  {r['risk_score']:.3f}]"
        for r in scores
    }

    selected = st.selectbox(
        "Select user",
        users,
        format_func=lambda u: labels[u],
    )

    if not selected:
        return

    detail = _fetch_user_detail(selected)
    if not detail:
        st.error(f"Could not retrieve details for {selected}.")
        return

    # ── Header ────────────────────────────────────────────────────────────────
    h1, h2, h3, h4 = st.columns(4)
    h1.markdown(f"**Risk Score**<br><span style='font-size:2rem;font-weight:bold'>{detail['risk_score']:.4f}</span>",
                unsafe_allow_html=True)
    h2.markdown(f"**Alert Level**<br>{_alert_badge(detail['alert_level'])}",
                unsafe_allow_html=True)
    h3.metric("Windows Analyzed",   detail["n_windows"])
    h4.metric("Alerts Fired (>0.7)", detail["n_alerts"])

    if detail["is_threat_user"]:
        st.warning(
            f"⚠️ Known insider threat — activity window: "
            f"**{detail['threat_start']}** → **{detail['threat_end']}**"
        )

    st.divider()

    # ── Temporal risk evolution ───────────────────────────────────────────────
    st.subheader("Temporal Risk Evolution")
    ws = pd.DataFrame(detail["window_scores"])
    ws["date"] = pd.to_datetime(ws["date"])
    ws = ws.sort_values("date")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ws["date"], y=ws["score"],
        mode="lines", name="Risk Score",
        line=dict(color="#4a90d9", width=1.5),
        fill="tozeroy", fillcolor="rgba(74,144,217,0.1)",
    ))

    # Threshold lines
    fig.add_hline(y=HIGH_THRESH,   line_dash="dash", line_color="#d9534f",
                  annotation_text="HIGH (0.7)",   annotation_position="top right")
    fig.add_hline(y=MEDIUM_THRESH, line_dash="dot",  line_color="#f0ad4e",
                  annotation_text="MEDIUM (0.4)", annotation_position="top right")

    # Threat period shading
    if detail["is_threat_user"] and detail["threat_start"]:
        fig.add_vrect(
            x0=detail["threat_start"], x1=detail["threat_end"],
            fillcolor="rgba(217,83,79,0.15)", line_width=0,
            annotation_text="Threat window", annotation_position="top left",
        )

    fig.update_layout(
        xaxis_title="Window Start Date",
        yaxis_title="Risk Score",
        yaxis=dict(range=[0, 1]),
        height=420,
        margin=dict(t=30, b=30),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Score summary stats ───────────────────────────────────────────────────
    scores_arr = ws["score"].values
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Max Score",    f"{scores_arr.max():.4f}")
    s2.metric("Mean Score",   f"{scores_arr.mean():.4f}")
    s3.metric("Median Score", f"{pd.Series(scores_arr).median():.4f}")
    s4.metric("Std Dev",      f"{scores_arr.std():.4f}")

    # ── Top 10 highest-risk windows ───────────────────────────────────────────
    st.subheader("Top 10 Highest-Risk Windows")
    top_wins = ws.nlargest(10, "score")[["date", "score"]].copy()
    top_wins["date"] = top_wins["date"].dt.strftime("%Y-%m-%d")
    top_wins.columns = ["Window Start", "Risk Score"]
    st.dataframe(
        top_wins.style.format({"Risk Score": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Page 3 — Model Performance
# ─────────────────────────────────────────────────────────────────────────────
def page_model_performance():
    st.title("📊 Model Performance")

    ev = _load_eval_results()
    if not ev:
        st.error(f"evaluation_results.json not found in {_OUT}. Run `python src/evaluate.py` first.")
        return

    # ── Checkpoint info ───────────────────────────────────────────────────────
    ck = ev["checkpoint"]
    st.caption(
        f"Checkpoint: epoch **{ck['epoch']}** · "
        f"val PR-AUC **{ck['val_prauc']}** · "
        f"val ROC-AUC **{ck.get('val_rocauc', 'N/A')}**"
    )

    # ── Key metrics ───────────────────────────────────────────────────────────
    m = ev["metrics_at_default_threshold"]
    st.subheader(f"Metrics at threshold={m['threshold']}")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("PR-AUC",    m["pr_auc"])
    c2.metric("ROC-AUC",   m["roc_auc"])
    c3.metric("Precision", m["precision"])
    c4.metric("Recall",    m["recall"])
    c5.metric("F1",        m["f1"])
    c6.metric("Accuracy",  m["accuracy"])

    st.divider()

    # ── Confusion matrix numbers ──────────────────────────────────────────────
    st.subheader("Confusion Matrix")
    cm_col, curves_col = st.columns([1, 2])

    with cm_col:
        tp, tn, fp, fn = m["TP"], m["TN"], m["FP"], m["FN"]
        cm_fig = go.Figure(go.Heatmap(
            z=[[tn, fp], [fn, tp]],
            x=["Pred Benign", "Pred Threat"],
            y=["Actual Benign", "Actual Threat"],
            text=[[f"TN\n{tn:,}", f"FP\n{fp:,}"], [f"FN\n{fn:,}", f"TP\n{tp:,}"]],
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
        ))
        cm_fig.update_layout(height=300, margin=dict(t=20, b=20))
        st.plotly_chart(cm_fig, use_container_width=True)

    # ── Threshold sweep chart ─────────────────────────────────────────────────
    with curves_col:
        st.subheader("Threshold Sweep")
        sweep = pd.DataFrame(ev["threshold_sweep"])
        sweep_fig = go.Figure()
        sweep_fig.add_trace(go.Scatter(
            x=sweep["threshold"], y=sweep["precision"],
            name="Precision", mode="lines+markers", line=dict(color="#4a90d9"),
        ))
        sweep_fig.add_trace(go.Scatter(
            x=sweep["threshold"], y=sweep["recall"],
            name="Recall", mode="lines+markers", line=dict(color="#e07b39"),
        ))
        sweep_fig.add_trace(go.Scatter(
            x=sweep["threshold"], y=sweep["f1"],
            name="F1", mode="lines+markers", line=dict(color="#27a744"),
        ))
        sweep_fig.update_layout(
            xaxis_title="Threshold", yaxis_title="Score",
            yaxis=dict(range=[0, 1]), height=300,
            margin=dict(t=20, b=20), legend=dict(x=0.01, y=0.99),
        )
        st.plotly_chart(sweep_fig, use_container_width=True)

    st.divider()

    # ── Saved plot images ─────────────────────────────────────────────────────
    st.subheader("Evaluation Plots")
    img_cols = st.columns(3)
    plots = [
        ("PR Curve",  _OUT / "pr_curve.png"),
        ("ROC Curve", _OUT / "roc_curve.png"),
        ("Time-to-Detection", _OUT / "time_to_detection.png"),
    ]
    for col, (title, path) in zip(img_cols, plots):
        with col:
            st.caption(title)
            if path.exists():
                st.image(str(path), use_container_width=True)
            else:
                st.info(f"{path.name} not found.")

    # ── Per-threat-user detection table ──────────────────────────────────────
    st.subheader("Per-Threat-User Detection Summary")
    threat_data = ev.get("per_user_threat_analysis", {})
    if threat_data:
        rows = []
        for user, r in sorted(threat_data.items()):
            d = r["days_before_threat_start"]
            rows.append({
                "User":           user,
                "Detected":       "✅ YES" if r["detected"] else "❌ NO",
                "Alerts Fired":   r["n_alerts"],
                "First Alert":    r.get("first_alert_date") or "—",
                "Threat Start":   r["threat_start"],
                "Days Early":     f"{d:+d}" if d is not None else "—",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No per-user threat analysis available.")


# ─────────────────────────────────────────────────────────────────────────────
# Navigation
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.sidebar.title("Insider Threat SOC")
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "User Drill-Down", "Model Performance"],
        label_visibility="collapsed",
    )

    health = _api("/health")
    if health:
        st.sidebar.success(
            f"API online · {health['n_users_scored']} users scored"
        )
    else:
        st.sidebar.error("API offline")

    st.sidebar.markdown("---")
    st.sidebar.caption("Thresholds: HIGH >0.7 · MEDIUM >0.4")

    if page == "Overview":
        page_overview()
    elif page == "User Drill-Down":
        page_drilldown()
    else:
        page_model_performance()


if __name__ == "__main__":
    main()
