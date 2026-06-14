"""
Air Quality Analytics - Urban Environment
Dashboard demo Tugas Besar II4013 Data Analytics (Kelompok 4).

Menjalankan:
    1. python prepare_data.py     (sekali, untuk menyiapkan data + model)
    2. streamlit run app.py
"""
import json
from contextlib import contextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed" / "air_quality_clean.csv"
WEATHER = BASE / "data" / "raw" / "global_air_quality_data_10000.csv"
MODEL_PKL = BASE / "models" / "aqi_model.pkl"
METRICS_JSON = BASE / "models" / "metrics.json"

st.set_page_config(page_title="Air Quality Analytics", layout="wide")

# ----------------------------------------------------------------------------- konstanta
FONT = "'Plus Jakarta Sans', sans-serif"
CAT_ORDER = ["Good", "Moderate", "Unhealthy for Sensitive Groups",
             "Unhealthy", "Very Unhealthy", "Hazardous"]
CAT_COLOR = {
    "Good": "#2ECC71", "Moderate": "#F1C40F",
    "Unhealthy for Sensitive Groups": "#E67E22", "Unhealthy": "#E74C3C",
    "Very Unhealthy": "#9B59B6", "Hazardous": "#7E2946",
}
CAT_SHORT = {"Unhealthy for Sensitive Groups": "USG", "Very Unhealthy": "Very Unhealthy"}
POLLUTANTS = {"CO AQI Value": "CO", "Ozone AQI Value": "Ozone",
              "NO2 AQI Value": "NO₂", "PM2.5 AQI Value": "PM2.5"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
ISO3 = {
    "Australia": "AUS", "Austria": "AUT", "Belgium": "BEL", "Bosnia and Herzegovina": "BIH",
    "Brazil": "BRA", "Canada": "CAN", "Chile": "CHL", "China": "CHN", "Colombia": "COL",
    "Ecuador": "ECU", "France": "FRA", "Germany": "DEU", "Guatemala": "GTM", "Hungary": "HUN",
    "India": "IND", "Ireland": "IRL", "Italy": "ITA", "Latvia": "LVA", "Lithuania": "LTU",
    "Luxembourg": "LUX", "Mexico": "MEX", "Montenegro": "MNE", "Nepal": "NPL",
    "Netherlands": "NLD", "Norway": "NOR", "Peru": "PER", "Poland": "POL", "Portugal": "PRT",
    "Romania": "ROU", "Spain": "ESP", "Switzerland": "CHE", "Thailand": "THA", "Turkey": "TUR",
}

# ----------------------------------------------------------------------------- styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
[data-testid="stAppViewContainer"] *, [data-testid="stSidebar"] * {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
.block-container {padding: 2rem 2.4rem 2.5rem; max-width: 1500px;}
section[data-testid="stSidebar"] {background:#FFFFFF; border-right:1px solid #ECEFF3;}
section[data-testid="stSidebar"] .block-container {padding-top:1.5rem;}
#MainMenu, footer, [data-testid="stHeader"] {visibility:hidden; height:0;}

/* Kartu: HANYA container panel ber-key yang diberi gaya, bukan semua block/kolom */
.stApp [class*="st-key-card"]{
    border:1px solid #ECEFF3 !important; border-radius:16px !important;
    box-shadow:0 1px 3px rgba(16,24,40,.04); background:#FFFFFF;
    padding:20px 22px !important;
}

/* Kartu KPI: kompak, konsisten, teks boleh turun baris */
.kpi{background:#FFFFFF;border:1px solid #ECEFF3;border-radius:16px;padding:18px 20px;
     box-shadow:0 1px 3px rgba(16,24,40,.04);min-height:106px;margin-bottom:6px;}
.kpi-label{color:#8A94A6;font-size:.9rem;font-weight:500;}
.kpi-value{font-size:1.6rem;font-weight:700;color:#1F2933;line-height:1.2;margin:6px 0 2px;
           overflow-wrap:anywhere;}
.kpi-badge{margin-top:4px;}
.badge{display:inline-block;padding:3px 12px;border-radius:20px;font-size:.78rem;font-weight:600;white-space:nowrap;}
.kpi-foot{color:#98A2B3;font-size:.82rem;margin-top:2px;}

/* Legend donut: grid rapi (dot | nama | nilai kanan) */
.legend{display:flex;flex-direction:column;}
.legend-item{display:grid;grid-template-columns:14px 1fr auto;align-items:center;column-gap:9px;
    padding:7px 0;border-bottom:1px solid #F4F6F9;font-size:.85rem;}
.legend-item:last-child{border-bottom:none;}
.legend-item .nm{color:#475467;line-height:1.25;}
.legend-item .vl{color:#1F2933;font-weight:600;text-align:right;white-space:nowrap;}
.dot{height:10px;width:10px;border-radius:50%;display:inline-block;}

.panel-title{font-size:1.02rem;font-weight:600;color:#1F2933;margin:0 0 .6rem;}
.panel-foot{color:#98A2B3;font-size:.8rem;margin-top:.6rem;}
.page-title{font-size:2rem;font-weight:700;color:#1F2933;margin:0;}
.page-sub{color:#8A94A6;font-size:.98rem;margin:2px 0 0;}
.side-title{font-size:1.3rem;font-weight:700;color:#1F2933;}
.side-sub{font-size:.88rem;color:#8A94A6;margin:-4px 0 .6rem;}
.side-foot{color:#98A2B3;font-size:.78rem;line-height:1.6;}
hr{margin:.6rem 0;border-color:#ECEFF3;}

/* Sidebar nav dari st.radio: tanpa lingkaran, shading saat terpilih */
section[data-testid="stSidebar"] div[role="radiogroup"]{gap:6px;}
section[data-testid="stSidebar"] div[role="radiogroup"] > label{
    display:flex;align-items:center;width:100%;padding:11px 14px;border-radius:10px;
    margin:0;cursor:pointer;border:1px solid transparent;transition:background .15s,color .15s;}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover{background:#F4F6F9;}
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child{display:none;}
section[data-testid="stSidebar"] div[role="radiogroup"] > label p{font-size:.96rem;color:#475467;font-weight:500;}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked){
    background:#E8F1FB;border-color:#D3E5F7;}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p{
    color:#2E86AB;font-weight:700;}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- loaders
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df["AQI Category"] = pd.Categorical(df["AQI Category"], categories=CAT_ORDER, ordered=True)
    return df


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PKL)


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    return json.loads(METRICS_JSON.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_weather():
    return pd.read_csv(WEATHER) if WEATHER.exists() else None


# ----------------------------------------------------------------------------- helpers UI
_card_n = [0]


@contextmanager
def panel(title=None, foot=None):
    _card_n[0] += 1
    with st.container(border=True, key=f"card{_card_n[0]}"):
        if title:
            st.markdown(f'<div class="panel-title">{title}</div>', unsafe_allow_html=True)
        yield
        if foot:
            st.markdown(f'<div class="panel-foot">{foot}</div>', unsafe_allow_html=True)


def kpi(label, value, foot="", badge=None):
    badge_html = ""
    if badge:
        c = CAT_COLOR.get(badge, "#2ECC71")
        badge_html = (f'<div class="kpi-badge"><span class="badge" '
                      f'style="background:{c}22;color:{c}">{CAT_SHORT.get(badge, badge)}</span></div>')
    return (f'<div class="kpi"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>{badge_html}'
            f'<div class="kpi-foot">{foot}</div></div>')


def hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def aqi_to_category(value):
    for hi, label in [(50, "Good"), (100, "Moderate"), (150, "Unhealthy for Sensitive Groups"),
                      (200, "Unhealthy"), (300, "Very Unhealthy"), (10_000, "Hazardous")]:
        if value <= hi:
            return label
    return "Hazardous"


def style_fig(fig, height=320):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color="#344054", size=12),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#EEF1F5", zeroline=False)
    return fig


def render(fig, height=320):
    st.plotly_chart(style_fig(fig, height), use_container_width=True,
                    config={"displayModeBar": False})


# ----------------------------------------------------------------------------- pages
def page_overview(df):
    head = st.columns([3, 1])
    with head[0]:
        st.markdown('<div class="page-title">Overview / KPI</div>'
                    '<div class="page-sub">Ringkasan kondisi kualitas udara secara global</div>',
                    unsafe_allow_html=True)
    with head[1]:
        mr = st.select_slider("Rentang bulan", options=list(range(1, 13)), value=(1, 12),
                              format_func=lambda m: MONTHS[m - 1], label_visibility="collapsed")
    d = df[df["month"].between(mr[0], mr[1])]
    st.write("")

    mean_aqi = d["AQI Value"].mean()
    haz_pct = (d["AQI Category"] == "Hazardous").mean() * 100
    k = st.columns(4)
    k[0].markdown(kpi("Total Record", f"{len(d):,}", "Data observasi"), unsafe_allow_html=True)
    k[1].markdown(kpi("Negara / Kota", f"{d['country'].nunique()} / {d['city'].nunique()}",
                      "Cakupan wilayah"), unsafe_allow_html=True)
    k[2].markdown(kpi("Rata-rata AQI", f"{mean_aqi:.1f}", badge=aqi_to_category(mean_aqi)),
                  unsafe_allow_html=True)
    k[3].markdown(kpi("% Hazardous", f"{haz_pct:.1f}%", "Dari seluruh data"), unsafe_allow_html=True)
    st.write("")

    c = st.columns(2)
    with c[0]:
        with panel("Distribusi Kategori AQI (Global)", "Berdasarkan AQI Value (US EPA)"):
            counts = d["AQI Category"].value_counts().reindex(CAT_ORDER).fillna(0)
            sub = st.columns([1, 1.2])
            with sub[0]:
                fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=.62,
                                       marker=dict(colors=[CAT_COLOR[c] for c in counts.index]),
                                       textinfo="none", sort=False))
                fig.add_annotation(text=f"<b>{len(d):,}</b><br>Total", showarrow=False, font_size=15)
                fig.update_layout(showlegend=False)
                render(fig, 280)
            with sub[1]:
                total = counts.sum()
                rows = "".join(
                    f'<div class="legend-item"><span class="dot" style="background:{CAT_COLOR[cat]}">'
                    f'</span><span class="nm">{cat}</span><span class="vl">'
                    f'{counts[cat] / total * 100 if total else 0:.1f}% · {int(counts[cat]):,}</span></div>'
                    for cat in CAT_ORDER)
                st.markdown(f'<div class="legend">{rows}</div>', unsafe_allow_html=True)
    with c[1]:
        with panel("Tren AQI Global (Rata-rata Bulanan)", "Bulan diturunkan dari timestamp OpenAQ"):
            m = d.groupby("month")["AQI Value"].mean().reindex(range(1, 13))
            fig = go.Figure(go.Scatter(x=MONTHS, y=m.values, mode="lines+markers",
                                       line=dict(color="#2E86AB", width=3), marker=dict(size=8),
                                       fill="tozeroy", fillcolor="rgba(46,134,171,.10)"))
            render(fig, 280)
    st.write("")

    top_country = d.groupby("country")["AQI Value"].mean().idxmax()
    with panel("Ringkasan Insight"):
        cols = st.columns(3)
        cols[0].markdown(f"**Kondisi Umum**\n\nRata-rata AQI global berada pada kategori "
                         f"**{aqi_to_category(mean_aqi)}** ({mean_aqi:.1f}). Sebagian besar observasi "
                         f"berada pada kategori Moderate dan Good.")
        cols[1].markdown(f"**Kategori Berisiko**\n\n{haz_pct:.1f}% data berada pada kategori Hazardous. "
                         f"Wilayah dengan rata-rata AQI tertinggi adalah **{top_country}**.")
        cols[2].markdown("**Rekomendasi Umum**\n\nPrioritaskan pengendalian PM2.5 dari sumber industri "
                         "dan pembakaran, perkuat pemantauan musiman, dan sebarkan peringatan dini "
                         "pada bulan puncak polusi.")


def page_trend(df):
    st.markdown('<div class="page-title">Eksplorasi Tren</div>'
                '<div class="page-sub">Pola temporal kualitas udara per bulan dan per jam</div>',
                unsafe_allow_html=True)
    st.write("")
    f = st.columns(2)
    sel = f[0].multiselect("Negara", sorted(df["country"].unique()), placeholder="Semua negara")
    mr = f[1].select_slider("Rentang bulan", options=list(range(1, 13)), value=(1, 12),
                            format_func=lambda m: MONTHS[m - 1])
    d = df[df["country"].isin(sel)] if sel else df
    d = d[d["month"].between(mr[0], mr[1])]
    st.write("")

    c = st.columns(2)
    with c[0]:
        with panel("Rata-rata AQI per Bulan (± 1 Std Dev)"):
            agg = d.groupby("month")["AQI Value"].agg(["mean", "std"]).reindex(range(1, 13))
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=MONTHS + MONTHS[::-1],
                y=list(agg["mean"] + agg["std"]) + list((agg["mean"] - agg["std"])[::-1]),
                fill="toself", fillcolor="rgba(46,134,171,.12)", line=dict(color="rgba(0,0,0,0)"),
                showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=MONTHS, y=agg["mean"], mode="lines+markers",
                                     line=dict(color="#2E86AB", width=3), name="Rata-rata"))
            render(fig)
    with c[1]:
        with panel("Rata-rata AQI per Jam"):
            h = d.groupby("hour")["AQI Value"].mean()
            fig = go.Figure(go.Scatter(x=h.index, y=h.values, mode="lines+markers",
                                       line=dict(color="#A23B72", width=3),
                                       fill="tozeroy", fillcolor="rgba(162,59,114,.10)"))
            fig.update_xaxes(title="Jam (0-23)")
            render(fig)

    with panel("Jumlah Observasi: Bulan × Kategori AQI", "Distribusi kategori sepanjang tahun"):
        piv = (d.pivot_table(index="AQI Category", columns="month", values="AQI Value",
                             aggfunc="count", observed=False)
               .reindex(CAT_ORDER).reindex(columns=range(1, 13)))
        fig = px.imshow(piv.values, x=MONTHS, y=CAT_ORDER, color_continuous_scale="Blues",
                        aspect="auto", labels=dict(color="Jumlah"))
        render(fig, 300)


def page_region(df):
    st.markdown('<div class="page-title">Perbandingan Wilayah</div>'
                '<div class="page-sub">Sebaran spasial dan peringkat kualitas udara antarnegara</div>',
                unsafe_allow_html=True)
    st.write("")
    metric = st.radio("Metrik", ["Rata-rata AQI", "% Hazardous"], horizontal=True)
    st.write("")

    g = df.groupby("country").agg(
        mean_aqi=("AQI Value", "mean"),
        haz=("AQI Category", lambda s: (s == "Hazardous").mean() * 100),
        n=("AQI Value", "size")).reset_index()
    g = g[g["n"] >= 30].copy()
    g["iso"] = g["country"].map(ISO3)
    value_col = "mean_aqi" if metric == "Rata-rata AQI" else "haz"

    with panel(f"Peta Sebaran {metric} per Negara"):
        fig = px.choropleth(g.dropna(subset=["iso"]), locations="iso", locationmode="ISO-3",
                            color=value_col, hover_name="country",
                            hover_data={"iso": False, value_col: ":.1f", "n": True},
                            color_continuous_scale="RdYlGn_r")
        fig.update_geos(projection_type="natural earth", showframe=False, showcoastlines=False,
                        showcountries=True, countrycolor="#E5E8EC", showland=True,
                        landcolor="#F4F6F9", showocean=True, oceancolor="#EAF1F6",
                        lataxis_range=[-58, 84])
        fig.update_coloraxes(colorbar_title=metric)
        render(fig, 430)
    st.write("")

    c = st.columns(2)
    with c[0]:
        with panel(f"10 Negara Paling Tercemar ({metric})"):
            top = g.sort_values(value_col, ascending=False).head(10)
            fig = px.bar(top, x=value_col, y="country", orientation="h",
                         color=value_col, color_continuous_scale="Reds")
            fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
            render(fig, 360)
    with c[1]:
        with panel(f"10 Negara Paling Bersih ({metric})"):
            bot = g.sort_values(value_col, ascending=True).head(10)
            fig = px.bar(bot, x=value_col, y="country", orientation="h",
                         color=value_col, color_continuous_scale="Greens_r")
            fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
            render(fig, 360)


def page_pollutant(df):
    st.markdown('<div class="page-title">Analisis Polutan</div>'
                '<div class="page-sub">Kontribusi tiap polutan dan hubungan antarvariabel</div>',
                unsafe_allow_html=True)
    st.write("")
    c = st.columns(2)
    with c[0]:
        with panel("Rata-rata AQI per Polutan"):
            means = df[list(POLLUTANTS)].mean().rename(index=POLLUTANTS)
            fig = px.bar(x=means.index, y=means.values, color=means.index,
                         color_discrete_sequence=["#2E86AB", "#F18F01", "#44BBA4", "#C73E1D"])
            fig.update_layout(showlegend=False)
            fig.update_traces(texttemplate="%{y:.1f}", textposition="outside")
            render(fig)
    with c[1]:
        with panel("Heatmap Korelasi Antarvariabel"):
            cols = list(POLLUTANTS) + ["traffic_pollution_proxy", "AQI Value"]
            labels = list(POLLUTANTS.values()) + ["Traffic", "AQI"]
            fig = px.imshow(df[cols].corr().values, x=labels, y=labels, text_auto=".2f",
                            color_continuous_scale="RdYlGn", zmin=-1, zmax=1, aspect="auto")
            render(fig)

    with panel("Sebaran AQI Tiap Polutan per Kategori (Box Plot)"):
        pol = st.selectbox("Polutan", list(POLLUTANTS), format_func=lambda c: POLLUTANTS[c])
        fig = px.box(df, x="AQI Category", y=pol, color="AQI Category",
                     category_orders={"AQI Category": CAT_ORDER}, color_discrete_map=CAT_COLOR)
        fig.update_layout(showlegend=False)
        render(fig, 340)

    weather = load_weather()
    if weather is not None:
        with panel("Pengaruh Faktor Cuaca terhadap Polutan (Dataset Pendukung)",
                   "Global Air Quality Dataset (10.000 baris: suhu, kelembapan, kecepatan angin)"):
            env = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
            wx = ["Temperature", "Humidity", "Wind Speed"]
            cc = st.columns(2)
            with cc[0]:
                corr = weather[env + wx].corr().loc[env, wx]
                fig = px.imshow(corr.values, x=wx, y=env, text_auto=".2f",
                                color_continuous_scale="RdBu", zmin=-.3, zmax=.3, aspect="auto")
                render(fig, 300)
            with cc[1]:
                wsel = st.selectbox("Faktor cuaca", wx)
                fig = px.scatter(weather.sample(min(2000, len(weather)), random_state=42),
                                 x=wsel, y="PM2.5", color="PM2.5",
                                 color_continuous_scale="YlOrRd", opacity=.55)
                fig.update_layout(coloraxis_showscale=False)
                render(fig, 300)


def page_model(df):
    st.markdown('<div class="page-title">Prediksi / Model</div>'
                '<div class="page-sub">Model regresi AQI dan simulasi prediksi interaktif</div>',
                unsafe_allow_html=True)
    st.write("")
    metrics = load_metrics()
    bundle = load_model()
    res = pd.DataFrame(metrics["results"])
    best = res.iloc[0]

    k = st.columns(4)
    k[0].markdown(kpi("Model Terbaik", metrics["best_model"], "RMSE terkecil"), unsafe_allow_html=True)
    k[1].markdown(kpi("R² (Test)", f"{best['R2']:.3f}", "Variasi AQI terjelaskan"), unsafe_allow_html=True)
    k[2].markdown(kpi("MAE", f"{best['MAE']:.2f}", "Rata-rata error absolut"), unsafe_allow_html=True)
    k[3].markdown(kpi("RMSE", f"{best['RMSE']:.2f}", "Penalti error besar"), unsafe_allow_html=True)
    st.write("")

    c = st.columns(2)
    with c[0]:
        with panel("Perbandingan Performa Model"):
            fig = px.bar(res.sort_values("R2"), x="R2", y="Model", orientation="h",
                         color="R2", color_continuous_scale="Blues")
            fig.update_layout(coloraxis_showscale=False)
            render(fig, 320)
    with c[1]:
        with panel("Feature Importance (Top 8)"):
            imp = pd.Series(metrics["feature_importance"]).head(8).sort_values()
            fig = px.bar(x=imp.values, y=imp.index, orientation="h",
                         color=imp.values, color_continuous_scale="Teal")
            fig.update_layout(coloraxis_showscale=False)
            render(fig, 320)

    with panel("Simulasi Prediksi AQI"):
        st.caption("Masukkan nilai sub-indeks polutan untuk memperkirakan AQI keseluruhan. "
                   "Model bersifat explanatory: sub-indeks polutan berhubungan langsung dengan AQI, "
                   "sehingga hasil mengonfirmasi kontributor dominan, bukan prediksi operasional independen.")
        i = st.columns(4)
        pm = i[0].slider("PM2.5 AQI", 0, 500, 120)
        oz = i[1].slider("Ozone AQI", 0, 300, 40)
        co = i[2].slider("CO AQI", 0, 60, 1)
        no2 = i[3].slider("NO₂ AQI", 0, 100, 2)

        feats = bundle["features"]
        row = df[feats].median(numeric_only=True).copy()
        row["PM2.5 AQI Value"], row["Ozone AQI Value"] = pm, oz
        row["CO AQI Value"], row["NO2 AQI Value"] = co, no2
        row["traffic_pollution_proxy"] = co + no2
        pred = float(bundle["model"].predict(pd.DataFrame([row])[feats])[0])
        cat = aqi_to_category(pred)

        r = st.columns([1, 2])
        r[0].markdown(kpi("Prediksi AQI Value", f"{pred:.0f}", badge=cat), unsafe_allow_html=True)
        with r[1]:
            bands = [(0, 50, "Good"), (50, 100, "Moderate"), (100, 150, "Unhealthy for Sensitive Groups"),
                     (150, 200, "Unhealthy"), (200, 300, "Very Unhealthy"), (300, 500, "Hazardous")]
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=pred,
                gauge={"axis": {"range": [0, 500]}, "bar": {"color": CAT_COLOR[cat]},
                       "steps": [{"range": [lo, hi], "color": hex_to_rgba(CAT_COLOR[cc], .22)}
                                 for lo, hi, cc in bands]}))
            render(fig, 220)


def page_insight(df):
    st.markdown('<div class="page-title">Insight & Rekomendasi</div>'
                '<div class="page-sub">Temuan utama berbasis data dan rekomendasi tindakan</div>',
                unsafe_allow_html=True)
    st.write("")
    insights = [
        ("PM2.5 adalah penentu dominan AQI",
         "Korelasi PM2.5 terhadap AQI mencapai r = 0.994 dan menjadi fitur paling penting pada model. "
         "Polutan lain (Ozone, CO, NO₂) berkontribusi minor.",
         "Intervensi dan pemantauan kualitas udara harus memprioritaskan PM2.5."),
        ("Polusi bersifat musiman, bukan harian",
         "AQI memuncak pada Januari–Februari (≈202–215) dan September–November (≈186–198), "
         "sedangkan tren per jam relatif datar.",
         "Peringatan dini dan kebijakan dijadwalkan mengikuti musim, bukan jam sibuk."),
        ("Lalu lintas bukan sumber utama PM2.5",
         "Proksi lalu lintas hanya berkorelasi r = 0.14 dengan AQI dan seragam di semua kategori.",
         "Fokus regulasi pada industri dan pembakaran, tidak hanya emisi kendaraan."),
        ("Beban polusi terkonsentrasi di wilayah tertentu",
         "India memiliki rata-rata AQI ≈ 242 (Very Unhealthy) dan mendominasi data, "
         "kontras dengan Australia/Kanada yang rata-rata di bawah 26.",
         "Sumber daya pengendalian perlu diarahkan ke wilayah dengan beban tertinggi."),
    ]
    st.markdown("### Insight Utama")
    cols = st.columns(2)
    for idx, (t, ev, im) in enumerate(insights):
        with cols[idx % 2]:
            with panel():
                st.markdown(f"**Insight {idx + 1} — {t}**")
                st.markdown(f'<span style="color:#475467">{ev}</span>', unsafe_allow_html=True)
                st.markdown(f"**Implikasi:** {im}")

    st.markdown("### Rekomendasi")
    recs = pd.DataFrame([
        ["Jadikan PM2.5 indikator prioritas pemantauan & baku mutu", "KLHK, DLH, BMKG", "r = 0.994; importance tertinggi", "Tinggi"],
        ["Aktifkan peringatan dini musiman (Jan–Feb, Sep–Nov)", "BMKG, Dinas Kesehatan", "Tren bulanan AQI", "Tinggi"],
        ["Arahkan pengendalian ke sumber non-kendaraan (industri, pembakaran)", "Pemda, KLHK", "Traffic proxy r = 0.14", "Sedang"],
        ["Prioritaskan wilayah beban tinggi (kategori Very Unhealthy)", "Pemerintah Pusat & Daerah", "India ≈ 242; 12.1% Hazardous", "Tinggi"],
    ], columns=["Rekomendasi", "Pihak Sasaran", "Dasar Data", "Prioritas"])
    with panel():
        st.dataframe(recs, use_container_width=True, hide_index=True)
        st.caption("Keterbatasan: AQI sangat ditentukan PM2.5 (potensi data leakage konseptual pada model), "
                   "integrasi inner join membatasi cakupan ke 33 negara/386 kota, dan resolusi temporal "
                   "terbatas pada timestamp OpenAQ.")


# ----------------------------------------------------------------------------- sidebar + router
PAGES = {
    "1  Overview / KPI": page_overview,
    "2  Eksplorasi Tren": page_trend,
    "3  Perbandingan Wilayah": page_region,
    "4  Analisis Polutan": page_pollutant,
    "5  Prediksi / Model": page_model,
    "6  Insight & Rekomendasi": page_insight,
}

with st.sidebar:
    st.markdown('<div class="side-title">Air Quality Analytics</div>'
                '<div class="side-sub">Urban Environment</div>', unsafe_allow_html=True)
    choice = st.radio("Navigasi", list(PAGES), label_visibility="collapsed")
    st.markdown("<br>" * 4, unsafe_allow_html=True)
    st.markdown('<hr><div class="side-foot"><b>Sumber Data</b><br>'
                'OpenAQ + Kaggle Global Air Pollution<br><br>'
                'Dashboard dibangun dengan Streamlit<br>Kelompok 4 · II4013 Data Analytics</div>',
                unsafe_allow_html=True)

if not DATA.exists():
    st.error("Data belum disiapkan. Jalankan `python prepare_data.py` terlebih dahulu.")
    st.stop()

_card_n[0] = 0
PAGES[choice](load_data())
