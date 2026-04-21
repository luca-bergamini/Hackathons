"""Frontend module - Dashboard interattiva per FarmaVita Sales Predictor.

Dashboard Streamlit che integra:
1. Visualizzazione predizioni di vendita per negozio/data
2. Chat con il Sales Advisor (RAG) per domande sui pattern di vendita
3. Grafici e analisi esplorative dei dati

Avvio:
    streamlit run src/frontend/main.py
"""

import os
import datetime
import requests
import streamlit as st
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
#  Configurazione
# ---------------------------------------------------------------------------

API_URL = os.environ.get("API_URL", "http://localhost:8000")
FEATURES_PATH = os.environ.get("FEATURES_PATH", "data/featured/df_final_features.parquet")

# ---------------------------------------------------------------------------
#  Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FarmaVita Sales Predictor",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
#  Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* ===== GLOBAL TEXT — force light text on dark bg ===== */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #e2e8f0 !important;
    }

    /* Force ALL text white/light */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: #ffffff !important;
    }
    .stApp p, .stApp span, .stApp div, .stApp label, .stApp li {
        color: #e2e8f0 !important;
    }
    .stApp .stMarkdown p {
        color: #e2e8f0 !important;
    }
    .stApp strong, .stApp b {
        color: #ffffff !important;
    }

    /* Streamlit widgets labels */
    .stApp .stSelectbox label,
    .stApp .stNumberInput label,
    .stApp .stDateInput label,
    .stApp .stMultiSelect label,
    .stApp .stTextInput label,
    .stApp .stRadio label {
        color: #e2e8f0 !important;
    }

    /* Sidebar — sfondo bianco */
    section[data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid rgba(0,0,0,0.1);
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] .stRadio label {
        color: #1e293b !important;
    }

    /* Metric values (st.metric) */
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    [data-testid="stMetricLabel"] {
        color: #c4b5fd !important;
    }
    [data-testid="stMetricDelta"] {
        color: #34d399 !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        color: #c4b5fd !important;
    }

    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(59, 52, 120, 0.7), rgba(44, 44, 75, 0.9));
        border: 1px solid rgba(165, 180, 252, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.3);
        border-color: rgba(165, 180, 252, 0.4);
    }
    .metric-card h3 {
        color: #c4b5fd !important;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.3rem;
    }
    .metric-card .value {
        color: #ffffff !important;
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-card small {
        color: #a5b4fc !important;
    }

    /* Chat bubbles */
    .chat-user {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: #ffffff !important;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        font-size: 0.95rem;
    }
    .chat-bot {
        background: rgba(255,255,255,0.1);
        color: #f1f5f9 !important;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 80%;
        border: 1px solid rgba(255,255,255,0.15);
        font-size: 0.95rem;
    }

    /* Status badge */
    .status-healthy {
        display: inline-block;
        background: #10b981;
        color: white !important;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-unhealthy {
        display: inline-block;
        background: #ef4444;
        color: white !important;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* Section titles */
    .section-title {
        color: #c4b5fd !important;
        font-size: 1.1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(139, 92, 246, 0.4);
    }

    /* Horizontal rule */
    .stApp hr {
        border-color: rgba(139, 92, 246, 0.25) !important;
    }

    /* Buttons */
    .stApp .stButton > button {
        border: 1px solid rgba(165, 180, 252, 0.3);
        color: #e2e8f0 !important;
    }

    /* Dataframe / tables */
    .stApp .stDataFrame {
        color: #e2e8f0 !important;
    }

    /* Info/warning/error boxes */
    .stApp .stAlert p, .stApp .stAlert span {
        color: #1e293b !important;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
#  Helper functions
# ---------------------------------------------------------------------------

def api_health() -> dict:
    """Check API health."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"status": f"unreachable — {e}"}


def api_predict(store_id: int, date: str, is_open: int) -> dict:
    """Call /predict endpoint."""
    r = requests.post(
        f"{API_URL}/predict",
        json={"store_id": store_id, "date": date, "is_open": is_open},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def api_predict_batch(items: list[dict]) -> list[dict]:
    """Call /predict/batch endpoint."""
    r = requests.post(
        f"{API_URL}/predict/batch",
        json={"requests": items},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["predictions"]


def api_explain(store_id: int, question: str) -> dict:
    """Call /explain endpoint."""
    r = requests.post(
        f"{API_URL}/explain",
        json={"store_id": store_id, "question": question},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=600)
def load_features_data():
    """Carica il dataset feature per i grafici esplorativi."""
    try:
        df = pd.read_parquet(FEATURES_PATH)
        df['date'] = pd.to_datetime(df['date'], format='mixed')
        return df
    except Exception:
        return None


# ---------------------------------------------------------------------------
#  Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 💊 FarmaVita")
    st.markdown("**Sales Predictor Dashboard**")
    st.markdown("---")

    # Health check
    health = api_health()
    is_healthy = health.get("status") == "healthy"
    badge_class = "status-healthy" if is_healthy else "status-unhealthy"
    badge_text = "● API Online" if is_healthy else "● API Offline"
    st.markdown(f'<span class="{badge_class}">{badge_text}</span>', unsafe_allow_html=True)
    st.markdown(f"<small style='color:#94a3b8'>Endpoint: {API_URL}</small>", unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigazione",
        ["🏠 Home", "📊 Predizioni", "🤖 Sales Advisor", "📈 Analisi Dati"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        "<small style='color:#64748b'>FarmaVita Hackathon 2026<br>Team 04</small>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
#  PAGE: Home
# ---------------------------------------------------------------------------

if page == "🏠 Home":
    st.markdown("# 💊 FarmaVita Sales Predictor")
    st.markdown("### Dashboard per la predizione e analisi delle vendite")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Predizioni</h3>
            <div class="value">53.5K</div>
            <small style="color:#94a3b8">Test set requests</small>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🏪 Negozi</h3>
            <div class="value">1.115</div>
            <small style="color:#94a3b8">Store monitorati</small>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 Modello</h3>
            <div class="value">XGBoost</div>
            <small style="color:#94a3b8">Ottimizzato con Optuna</small>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔗 API Status</h3>
            <div class="value">{"✅" if is_healthy else "❌"}</div>
            <small style="color:#94a3b8">{"Online" if is_healthy else "Offline"}</small>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 🚀 Come funziona")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        **📊 Predizioni**

        Inserisci un negozio e una data per ottenere la predizione delle vendite.
        Puoi anche generare predizioni per più giorni e visualizzare il trend.
        """)
    with col_b:
        st.markdown("""
        **🤖 Sales Advisor**

        Chatta con il Sales Advisor AI alimentato da RAG.
        Fai domande sui pattern di vendita, trend stagionali e performance dei negozi.
        """)
    with col_c:
        st.markdown("""
        **📈 Analisi Dati**

        Esplora i dati storici con grafici interattivi.
        Confronta negozi, analizza stagionalità e scopri insight nascosti.
        """)


# ---------------------------------------------------------------------------
#  PAGE: Predizioni
# ---------------------------------------------------------------------------

elif page == "📊 Predizioni":
    st.markdown("# 📊 Predizioni Vendite")
    st.markdown('<div class="section-title">Predizione singola</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        store_id = st.number_input("🏪 Store ID", min_value=1, max_value=1115, value=42, step=1)
    with col2:
        pred_date = st.date_input("📅 Data", value=datetime.date(2015, 9, 15))
    with col3:
        is_open = st.selectbox("🚪 Stato negozio", [1, 0], format_func=lambda x: "Aperto" if x == 1 else "Chiuso")

    if st.button("🔮 Predici Vendite", use_container_width=True, type="primary"):
        with st.spinner("Calcolo predizione..."):
            try:
                result = api_predict(store_id, str(pred_date), is_open)
                sales = result["predicted_sales"]

                st.markdown(f"""
                <div class="metric-card" style="text-align:center; margin-top:1rem;">
                    <h3>Vendite previste per Store {store_id} — {pred_date}</h3>
                    <div class="value" style="font-size:3rem; color:#6366f1">€ {sales:,.0f}</div>
                    <small style="color:#94a3b8">{"Negozio aperto" if is_open else "Negozio chiuso — vendite = 0"}</small>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Errore: {e}")

    # --- Predizione multi-giorno ---
    st.markdown("---")
    st.markdown('<div class="section-title">Predizione multi-giorno</div>', unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        multi_store = st.number_input("🏪 Store ID ", min_value=1, max_value=1115, value=42, step=1, key="multi_store")
    with col_b:
        start_date = st.date_input("📅 Data inizio", value=datetime.date(2015, 8, 1), key="start_date")
    with col_c:
        end_date = st.date_input("📅 Data fine", value=datetime.date(2015, 9, 17), key="end_date")

    if st.button("📈 Genera Trend", use_container_width=True):
        if start_date >= end_date:
            st.warning("La data di inizio deve essere prima della data di fine.")
        else:
            dates = pd.date_range(start_date, end_date)
            items = [
                {"id": i + 1, "store_id": multi_store, "date": str(d.date()), "is_open": 1}
                for i, d in enumerate(dates)
            ]

            with st.spinner(f"Calcolo {len(items)} predizioni..."):
                try:
                    # Batch in blocchi da 500
                    all_preds = []
                    for i in range(0, len(items), 500):
                        batch = items[i:i + 500]
                        preds = api_predict_batch(batch)
                        all_preds.extend(preds)

                    # Costruisci DataFrame
                    pred_df = pd.DataFrame(all_preds)
                    pred_df["date"] = dates[:len(pred_df)]
                    pred_df = pred_df.sort_values("date")

                    # Chart
                    st.markdown(f"#### 📈 Trend vendite — Store {multi_store}")
                    chart_data = pred_df.set_index("date")["predicted_sales"]
                    st.line_chart(chart_data, use_container_width=True)

                    # Statistiche
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Media", f"€ {pred_df['predicted_sales'].mean():,.0f}")
                    with col2:
                        st.metric("Max", f"€ {pred_df['predicted_sales'].max():,.0f}")
                    with col3:
                        st.metric("Min", f"€ {pred_df['predicted_sales'].min():,.0f}")
                    with col4:
                        st.metric("Totale", f"€ {pred_df['predicted_sales'].sum():,.0f}")

                    # Tabella
                    with st.expander("📋 Dettaglio predizioni"):
                        st.dataframe(
                            pred_df[["date", "predicted_sales"]].rename(
                                columns={"date": "Data", "predicted_sales": "Vendite Previste"}
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )

                except Exception as e:
                    st.error(f"Errore: {e}")


# ---------------------------------------------------------------------------
#  PAGE: Sales Advisor (RAG)
# ---------------------------------------------------------------------------

elif page == "🤖 Sales Advisor":
    st.markdown("# 🤖 Sales Advisor")
    st.markdown("Chatta con l'AI per scoprire insights sulle vendite dei negozi FarmaVita.")
    st.markdown("---")

    # Store selector
    advisor_store = st.number_input(
        "🏪 Store ID di riferimento",
        min_value=1, max_value=1115, value=42, step=1,
        help="Il Sales Advisor cercherà informazioni specifiche per questo negozio",
    )

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Suggerimenti rapidi
    st.markdown('<div class="section-title">Domande suggerite</div>', unsafe_allow_html=True)
    suggestions = [
        "Quali sono i giorni con più vendite?",
        "Come vanno le vendite durante le promozioni?",
        "Qual è il trend mensile delle vendite?",
        "Qual è la media di vendita giornaliera?",
    ]
    cols = st.columns(len(suggestions))
    for i, suggestion in enumerate(suggestions):
        with cols[i]:
            if st.button(f"💡 {suggestion}", key=f"suggestion_{i}", use_container_width=True):
                st.session_state.pending_question = suggestion

    st.markdown("---")

    # Chat display
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bot">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

    # Input
    question = st.chat_input("Fai una domanda sulle vendite...")

    # Gestisci domanda da suggerimento
    if "pending_question" in st.session_state:
        question = st.session_state.pending_question
        del st.session_state.pending_question

    if question:
        # Aggiungi messaggio utente
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.markdown(f'<div class="chat-user">🧑 {question}</div>', unsafe_allow_html=True)

        with st.spinner("Il Sales Advisor sta pensando..."):
            try:
                result = api_explain(advisor_store, question)
                answer = result.get("answer", "Mi dispiace, non sono riuscito a rispondere.")
                sources = result.get("sources", [])

                # Aggiungi risposta
                st.session_state.chat_history.append({"role": "bot", "content": answer})
                st.markdown(f'<div class="chat-bot">🤖 {answer}</div>', unsafe_allow_html=True)

                # Sources
                if sources:
                    with st.expander("📚 Fonti utilizzate"):
                        for i, source in enumerate(sources):
                            st.markdown(f"**Fonte {i + 1}:**")
                            st.text(source[:300] + "..." if len(source) > 300 else source)
                            st.markdown("---")

            except Exception as e:
                error_msg = f"Errore nella comunicazione con il Sales Advisor: {e}"
                st.session_state.chat_history.append({"role": "bot", "content": error_msg})
                st.error(error_msg)

    # Pulsante reset chat
    if st.session_state.chat_history:
        if st.button("🗑️ Cancella chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


# ---------------------------------------------------------------------------
#  PAGE: Analisi Dati
# ---------------------------------------------------------------------------

elif page == "📈 Analisi Dati":
    st.markdown("# 📈 Analisi Dati Storici")
    st.markdown("---")

    df = load_features_data()

    if df is None:
        st.error(
            f"Impossibile caricare i dati da `{FEATURES_PATH}`. "
            "Assicurati che la pipeline sia stata eseguita."
        )
    else:
        # Filtro solo training e aperti
        train_df = df[(df.get("is_train", pd.Series([True] * len(df))) == True)]
        if "is_open" in train_df.columns:
            open_df = train_df[train_df["is_open"] == 1]
        else:
            open_df = train_df

        # Store selector
        all_stores = sorted(open_df["store_id"].unique())
        selected_stores = st.multiselect(
            "🏪 Seleziona negozi (max 5)",
            options=all_stores,
            default=[42] if 42 in all_stores else all_stores[:1],
            max_selections=5,
        )

        if not selected_stores:
            st.info("Seleziona almeno un negozio per visualizzare i grafici.")
        else:
            filtered = open_df[open_df["store_id"].isin(selected_stores)]

            # --- Vendite medie per giorno della settimana ---
            st.markdown('<div class="section-title">Vendite medie per giorno della settimana</div>', unsafe_allow_html=True)

            if "day_of_week" in filtered.columns and "sales" in filtered.columns:
                day_names = {0: "Lun", 1: "Mar", 2: "Mer", 3: "Gio", 4: "Ven", 5: "Sab", 6: "Dom"}
                dow_data = (
                    filtered.groupby(["store_id", "day_of_week"])["sales"]
                    .mean()
                    .reset_index()
                )
                dow_data["giorno"] = dow_data["day_of_week"].map(day_names)

                # Pivot per multiline chart
                pivot = dow_data.pivot(index="giorno", columns="store_id", values="sales")
                # Ordina giorni
                day_order = ["Lun", "Mar", "Mer", "Gio", "Ven", "Sab", "Dom"]
                pivot = pivot.reindex([d for d in day_order if d in pivot.index])

                st.bar_chart(pivot, use_container_width=True)
            else:
                st.warning("Colonne `day_of_week` o `sales` non trovate nel dataset.")

            # --- Vendite medie per mese ---
            st.markdown('<div class="section-title">Vendite medie per mese</div>', unsafe_allow_html=True)

            if "month" in filtered.columns and "sales" in filtered.columns:
                month_names = {1: "Gen", 2: "Feb", 3: "Mar", 4: "Apr", 5: "Mag", 6: "Giu",
                               7: "Lug", 8: "Ago", 9: "Set", 10: "Ott", 11: "Nov", 12: "Dic"}
                mo_data = (
                    filtered.groupby(["store_id", "month"])["sales"]
                    .mean()
                    .reset_index()
                )
                mo_data["mese"] = mo_data["month"].map(month_names)

                pivot_mo = mo_data.pivot(index="mese", columns="store_id", values="sales")
                month_order = ["Gen", "Feb", "Mar", "Apr", "Mag", "Giu", "Lug", "Ago", "Set", "Ott", "Nov", "Dic"]
                pivot_mo = pivot_mo.reindex([m for m in month_order if m in pivot_mo.index])

                st.line_chart(pivot_mo, use_container_width=True)
            else:
                st.warning("Colonne `month` o `sales` non trovate nel dataset.")

            # --- Serie storica vendite ---
            st.markdown('<div class="section-title">Serie storica vendite</div>', unsafe_allow_html=True)

            if "date" in filtered.columns and "sales" in filtered.columns:
                # Resample settimanale per leggibilità
                for sid in selected_stores:
                    store_data = filtered[filtered["store_id"] == sid].copy()
                    store_data = store_data.set_index("date")["sales"].resample("W").mean()
                    st.markdown(f"**Store {sid}** — media settimanale")
                    st.line_chart(store_data, use_container_width=True)
            else:
                st.warning("Colonne `date` o `sales` non trovate.")

            # --- Impatto promozioni ---
            if "promo" in filtered.columns and "sales" in filtered.columns:
                st.markdown('<div class="section-title">Impatto promozioni sulle vendite</div>', unsafe_allow_html=True)

                promo_stats = (
                    filtered.groupby(["store_id", "promo"])["sales"]
                    .mean()
                    .reset_index()
                )
                promo_stats["promo_label"] = promo_stats["promo"].map({0: "No Promo", 1: "Promo"})

                for sid in selected_stores:
                    store_promo = promo_stats[promo_stats["store_id"] == sid]
                    if len(store_promo) == 2:
                        no_promo = store_promo[store_promo["promo"] == 0]["sales"].values[0]
                        with_promo = store_promo[store_promo["promo"] == 1]["sales"].values[0]
                        uplift = ((with_promo - no_promo) / no_promo * 100) if no_promo > 0 else 0

                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.metric(f"Store {sid} — Senza promo", f"€ {no_promo:,.0f}")
                        with c2:
                            st.metric(f"Store {sid} — Con promo", f"€ {with_promo:,.0f}")
                        with c3:
                            st.metric(f"Store {sid} — Uplift", f"{uplift:+.1f}%")

            # --- Statistiche generali ---
            st.markdown('<div class="section-title">Statistiche generali</div>', unsafe_allow_html=True)

            for sid in selected_stores:
                store_data = filtered[filtered["store_id"] == sid]
                if "sales" in store_data.columns:
                    st.markdown(f"#### Store {sid}")
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("Media vendite", f"€ {store_data['sales'].mean():,.0f}")
                    with c2:
                        st.metric("Mediana", f"€ {store_data['sales'].median():,.0f}")
                    with c3:
                        st.metric("Max", f"€ {store_data['sales'].max():,.0f}")
                    with c4:
                        st.metric("Giorni osservati", f"{len(store_data):,}")
