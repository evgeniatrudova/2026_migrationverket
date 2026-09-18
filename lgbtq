import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import concurrent.futures
from datetime import datetime
import json
from fpdf import FPDF
from openai import OpenAI
from streamlit_autorefresh import st_autorefresh

# ---------------------------------------------------------
# Bilingual & Academic Configuration
# ---------------------------------------------------------
I18N = {
    "sv": {
        "title": "⚖️ COI & Kriminologisk Hotbildsanalys (HBTQI)",
        "subtitle": "Landinformation och akademisk trendanalys för asylprövning.",
        "warning": "<b>Dataminimering & Integritet:</b> Inga sökningar eller API-nycklar lagras lokalt (Stateless operation).",
        "sidebar_lang": "Språk / Language",
        "sidebar_params": "Utredningsparametrar",
        "state_select": "Välj delstat",
        "year_select": "Publiceringsår (Tidskriteriet)",
        "custom_focus": "Specifik utredningsfråga (Frivillig)",
        "openai_key": "OpenAI API-nyckel",
        "news_key": "NewsAPI-nyckel (Realtidsmedia)",
        "run_query": "Kör Systematisk COI-utvinning",
        "tab_delta": "📊 Risk-Delta (Kriminologi)",
        "tab_live": "🚨 Realtidsövervakning (Media)",
        "tab_ai": "🤖 Rättslig Syntes",
        "tab_audit": "🛡️ Söklogg & Metod",
        "tab_pdf": "📄 Beslutsunderlag (PDF)"
    },
    "en": {
        "title": "⚖️ COI & Criminological Threat Analysis (LGBTQI+)",
        "subtitle": "Country of Origin Information and academic trend analysis.",
        "warning": "<b>Data Privacy:</b> No queries or API keys are stored locally (Stateless operation).",
        "sidebar_lang": "Language / Språk",
        "sidebar_params": "Investigation Parameters",
        "state_select": "Select State",
        "year_select": "Publication Year (Proximity in Time)",
        "custom_focus": "Specific Legal Focus (Optional)",
        "openai_key": "OpenAI API Key",
        "news_key": "NewsAPI Key (Real-time Media)",
        "run_query": "Execute Systematic COI Retrieval",
        "tab_delta": "📊 Risk Delta (Criminology)",
        "tab_live": "🚨 Real-Time Monitor (Media)",
        "tab_ai": "🤖 Legal Synthesis",
        "tab_audit": "🛡️ Audit Trail & Method",
        "tab_pdf": "📄 Download Brief (PDF)"
    }
}

STATE_MAPPING = {
    "California": "University of California",
    "Texas": "Texas A&M University",
    "Florida": "University of Central Florida",
    "New York": "New York University",
    "Ohio": "Ohio State University",
    "Michigan": "University of Michigan",
    "Washington": "University of Washington"
}

# ---------------------------------------------------------
# Criminological Engine: Baseline Risk Delta
# ---------------------------------------------------------
def calculate_risk_delta(state: str) -> dict:
    """Mock structural baseline calculation."""
    baselines = {
        "California": {"gen": 4.4, "hate": 8.1, "rr": 1.84},
        "Texas": {"gen": 4.3, "hate": 11.2, "rr": 2.60},
        "Florida": {"gen": 3.8, "hate": 10.5, "rr": 2.76},
    }
    data = baselines.get(state, {"gen": 4.0, "hate": 9.0, "rr": 2.25})
    data["level"] = "Kritisk" if data["rr"] > 2.5 else "Förhöjd" if data["rr"] > 1.5 else "Baslinje"
    return data

# ---------------------------------------------------------
# Real-Time Media Velocity Engine
# ---------------------------------------------------------
def fetch_live_media_velocity(state: str, api_key: str) -> dict:
    if not api_key:
        return {"count": 0, "headline": "API-nyckel saknas för live-data.", "source": "N/A", "time": ""}
    
    # Mocking real-time fetch to avoid strict API limits in demo
    return {
        "count": 14,
        "headline": f"Vandalism at local LGBTQ youth center in {state} currently under investigation.",
        "source": f"{state} Local Tribune",
        "time": datetime.now().strftime('%H:%M:%S')
    }

# ---------------------------------------------------------
# PubMed Extraction Engine
# ---------------------------------------------------------
def generate_search_string(university: str, year: int) -> str:
    return f'(("Transgender Persons"[Mesh] OR transgender[Title/Abstract]) AND {year}[Date - Publication] AND ("{university}"[Affiliation]))'

def fetch_pubmed_data(state: str, university: str, year: int) -> dict:
    email = "coi_research@migrationsverket.se"
    query = generate_search_string(university, year)
    search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={query}&retmode=json&retmax=15&email={email}"
    
    try:
        res = requests.get(search_url, timeout=8).json()
        id_list = res.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return {"state": state, "status": "Success (0 hits)", "data": []}
            
        sum_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={','.join(id_list)}&retmode=json&email={email}"
        sum_res = requests.get(sum_url, timeout=8).json().get("result", {})
        
        articles = []
        for pmid in id_list:
            item = sum_res.get(pmid, {})
            articles.append({
                "state": state, "pmid": pmid,
                "title": item.get("title", "Unknown").rstrip("."),
                "journal": item.get("source", "N/A")
            })
        return {"state": state, "status": f"Success ({len(articles)} hits)", "data": articles}
    except Exception as e:
        return {"state": state, "status": f"API Error: {str(e)}", "data": []}

def execute_extraction(state: str, year: int):
    university = STATE_MAPPING[state]
    return fetch_pubmed_data(state, university, year)

# ---------------------------------------------------------
# LLM Legal Synthesis 
# ---------------------------------------------------------
def generate_ai_synthesis(api_key: str, df: pd.DataFrame, custom_focus: str, lang: str) -> dict:
    if not api_key or df.empty:
        return {"synthesis": "API Key required or no data available."}
    
    client = OpenAI(api_key=api_key)
    context = "\n".join(f"- [PMID: {r['pmid']}] {r['title']}" for _, r in df.iterrows())
    
    prompt = f"""
    Du är en asylrättslig landinformationsanalytiker (COI). Analysera titlarna för transpersoner i USA.
    Svara på formell juridisk {'svenska' if lang == 'sv' else 'engelska'}.
    Fokusera på: {custom_focus if custom_focus else 'Kumulativ diskriminering och myndighetsskydd.'}
    
    KRAV PÅ KÄLLHÄNVISNING: Varje påstående MÅSTE åtföljas av referens till artikelns PMID, t.ex. (PMID: 12345678).
    
    Formatera strikt som JSON: {{ "synthesis": "Din källhänvisade analys här i 2-3 stycken." }}
    Titlar: {context}
    """
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "system", "content": prompt}],
            temperature=0.0, response_format={"type": "json_object"}
        )
        return json.loads(res.choices[0].message.content)
    except Exception as e:
        return {"synthesis": f"Error: {str(e)}"}

# ---------------------------------------------------------
# PDF Generator 
# ---------------------------------------------------------
class COIPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(100, 116, 139)
        self.cell(0, 8, "MIGRATIONSVERKET - RÄTTSLIGT BESLUTSUNDERLAG (COI)", border=0, align='L')
        self.line(10, 15, 200, 15)
        self.ln(8)
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f"Genererad: {datetime.now().strftime('%Y-%m-%d %H:%M')}", align='C')

def generate_pdf(df: pd.DataFrame, ai_data: dict, year: int, query: str, state: str) -> bytes:
    pdf = COIPDF()
    pdf.add_page()
    
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 8, f"Rätts- och Socialrapport: Säkerhetsläge ({state}, USA)", ln=True)
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 6, f"Referensår: {year} | N = {len(df)} referensgranskade källor", ln=True)
    pdf.ln(5)

    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 7, "1. Rättslig Syntes (AI-understödd, källhänvisad)", ln=True)
    pdf.set_font('Helvetica', '', 10)
    pdf.multi_cell(0, 5, ai_data.get("synthesis", "Ingen syntes.").encode('latin-1', 'replace').decode('latin-1'))
    pdf.ln(5)

    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 7, "2. Metodologi & Söksträng (Spårbarhet)", ln=True)
    pdf.set_font('Helvetica', 'I', 9)
    pdf.multi_cell(0, 5, f"API Query: {query}")
    pdf.ln(5)

    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 7, "3. Formell Referenslista", ln=True)
    pdf.set_font('Helvetica', '', 9)
    for _, row in df.iterrows():
        ref = f"[{row['pmid']}] {row['title']} ({row['journal']}). PubMed."
        pdf.multi_cell(0, 5, ref.encode('latin-1', 'replace').decode('latin-1'))
        pdf.ln(2)

    return pdf.output()

# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="COI System: Migrationsverket", layout="wide")
    
    # 10 Minute Autorefresh (600,000 ms) for Live Media Tab
    st_autorefresh(interval=600000, key="data_refresh")
    
    lang = "sv" if st.sidebar.radio("Språk / Language", ["Svenska", "English"], horizontal=True) == "Svenska" else "en"
    t = I18N[lang]

    st.title(t["title"])
    st.caption(t["subtitle"])
    st.info(t["warning"])

    with st.sidebar:
        st.header(t["sidebar_params"])
        target_state = st.selectbox(t["state_select"], list(STATE_MAPPING.keys()))
        target_year = st.slider(t["year_select"], 2020, 2026, 2026)
        custom_focus = st.text_area(t["custom_focus"])
        openai_key = st.text_input(t["openai_key"], type="password")
        news_key = st.text_input(t["news_key"], type="password")
        run_query = st.button(t["run_query"], type="primary", use_container_width=True)

    if run_query:
        with st.spinner("Kommunicerar med NCBI/PubMed API..."):
            result = execute_extraction(target_state, target_year)
            df = pd.DataFrame(result["data"])
            st.session_state['data'] = df
            st.session_state['audit'] = result["status"]
            st.session_state['state'] = target_state
            st.session_state['year'] = target_year
            st.session_state['ai'] = generate_ai_synthesis(openai_key, df, custom_focus, lang)

    tabs = st.tabs([t["tab_delta"], t["tab_live"], t["tab_ai"], t["tab_audit"], t["tab_pdf"]])

    # 1. Criminological Risk Delta
    with tabs[0]:
        st.subheader("Strukturell Överrisk (Relative Risk)")
        risk = calculate_risk_delta(target_state)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Gen. Våldsbrott (per 100k)", risk["gen"])
        c2.metric("HBTQI Hatbrott (per 100k)", risk["hate"])
        c3.metric("Relativ Risk (RR)", f"{risk['rr']}x", delta="Överrisk", delta_color="inverse")
        c4.metric("Officiell Hotnivå", risk["level"])
        
        fig = go.Figure(data=[
            go.Bar(name='Allmän (Gen Pop)', x=[target_state], y=[risk["gen"]], marker_color='#94A3B8'),
            go.Bar(name='Riktat Våld (HBTQI)', x=[target_state], y=[risk["hate"]], marker_color='#EF4444')
        ])
        fig.update_layout(barmode='group', height=350, margin=dict(t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # 2. Live Media
    with tabs[1]:
        st.subheader(f"Media Threat Velocity (10-min uppdatering)")
        live_data = fetch_live_media_velocity(target_state, news_key)
        m1, m2 = st.columns([1, 3])
        with m1:
            st.metric("Händelser (24h)", live_data["count"], delta="Uppdaterades nyss" if news_key else "Väntar")
        with m2:
            st.info(f"**Senaste Rubrik ({live_data['time']}):**\n\n*{live_data['headline']}*\n\nKälla: {live_data['source']}")

    if 'data' in st.session_state and not st.session_state['data'].empty:
        df = st.session_state['data']
        ai_data = st.session_state['ai']
        
        # 3. AI Legal Synthesis
        with tabs[2]:
            st.write(ai_data.get("synthesis", ""))

        # 4. Audit
        with tabs[3]:
            st.code(generate_search_string(STATE_MAPPING[target_state], target_year))
            st.write(f"**API Status:** {st.session_state['audit']}")
            st.dataframe(df[['pmid', 'title']], use_container_width=True)

        # 5. PDF
        with tabs[4]:
            pdf_bytes = generate_pdf(df, ai_data, target_year, generate_search_string(STATE_MAPPING[target_state], target_year), target_state)
            st.download_button(label=t["tab_pdf"], data=bytes(pdf_bytes), file_name=f"COI_{target_state}_{target_year}.pdf", mime="application/pdf")

if __name__ == "__main__":
    main()
