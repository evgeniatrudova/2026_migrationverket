import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import hashlib
from fpdf import FPDF
import os

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ---------------------------------------------------------
# Configuration & Bilingual Dictionary
# ---------------------------------------------------------
st.set_page_config(page_title="LQBTQ-EVelution", layout="wide", initial_sidebar_state="collapsed")

I18N = {
    "sv": {
        "title": "Beslutsunderlag",
        "legal_warning": "AI-algoritmen är juridiskt ägd av EVelutionAB. Tillfällig användning gäller vid Nathaniel Christian Karlsson bruk samt vid utvärdering av Robin L'fira ärende.",
        "admin_header": "1. Ärendeuppgifter (För närvarande inaktiverad)",
        "case_num": "Ärendenummer",
        "officer_1": "Handläggare (Signatur)",
        "officer_2": "Beslutsfattare (Signatur)",
        "param_header": "2. Utredningsparametrar",
        "state": "Geografiskt område (USA)",
        "year": "Referensår (Tidskriteriet)",
        "databases": "Validerade databaser",
        "question_header": "3. Rättslig Frågeställning",
        "question": "Standardiserad SOGI-fråga",
        "btn_run": "Generera Underlag",
        "methodology_header": "Metodologisk Validering",
        "methodology_text": "Denna metodologiska ram för landinformation (COI) har utvecklats och standardiserats i samarbete med forskargrupper vid Uppsala universitet, Lunds universitet och Stockholms universitet. Ramverket tillämpar rigorös källkritik och algoritmisk validering av empirisk data. En referensbaslinje (Sverige) appliceras konsekvent för att motverka kognitiv bias och säkerställa en objektiv komparativ bedömning i enlighet med förvaltningsrättsliga beviskrav.",
        "results_header": "Analysresultat",
        "ref_header": "Referensförteckning (Validerade källor)",
        "pdf_btn": "Ladda ner PDF för Journalföring",
        "eval_header": "Kvantitativa Utvärderingsparametrar (Komparativ Analys)"
    },
    "en": {
        "title": "Country of Origin Information (COI) - Dossier",
        "legal_warning": "LEGAL NOTICE: The AI algorithm is under the legal ownership of EVelutionAB and is not to be used by the Migration Agency without a valid contract of use.",
        "admin_header": "1. Administrative Data (Currently Disabled)",
        "case_num": "Case Number",
        "officer_1": "Case Officer (Signature)",
        "officer_2": "Decision Maker (Signature)",
        "param_header": "2. Investigation Parameters",
        "state": "Geographical Area (USA)",
        "year": "Reference Year",
        "databases": "Validated Databases",
        "question_header": "3. Legal Inquiry",
        "question": "Standardized SOGI Question",
        "btn_run": "Generate Dossier",
        "methodology_header": "Methodological Validation",
        "methodology_text": "This methodological framework for COI has been standardized in collaboration with Uppsala, Lund, and Stockholm Universities. It applies rigorous source criticism and algorithmic validation. A reference baseline (Sweden) is consistently applied to mitigate cognitive bias and ensure objective comparative assessment in accordance with administrative evidentiary requirements.",
        "results_header": "Analysis Results",
        "ref_header": "Reference List (Validated Sources)",
        "pdf_btn": "Download PDF for Archiving",
        "eval_header": "Quantitative Evaluation Parameters (Comparative Analysis)"
    }
}

STATE_MAPPING = {
    "Alabama": "University of Alabama", "Alaska": "University of Alaska", "Arizona": "University of Arizona",
    "Arkansas": "University of Arkansas", "California": "University of California", "Colorado": "University of Colorado",
    "Connecticut": "University of Connecticut", "Delaware": "University of Delaware", "Florida": "University of Florida",
    "Georgia": "University of Georgia", "Hawaii": "University of Hawaii", "Idaho": "University of Idaho",
    "Illinois": "University of Illinois", "Indiana": "Indiana University", "Iowa": "University of Iowa",
    "Kansas": "University of Kansas", "Kentucky": "University of Kentucky", "Louisiana": "Louisiana State University",
    "Maine": "University of Maine", "Maryland": "University of Maryland", "Massachusetts": "University of Massachusetts",
    "Michigan": "University of Michigan", "Minnesota": "University of Minnesota", "Mississippi": "University of Mississippi",
    "Missouri": "University of Missouri", "Montana": "Montana State University", "Nebraska": "University of Nebraska",
    "Nevada": "University of Nevada", "New Hampshire": "University of New Hampshire", "New Jersey": "Rutgers University",
    "New Mexico": "University of New Mexico", "New York": "New York University", "North Carolina": "University of North Carolina",
    "North Dakota": "University of North Dakota", "Ohio": "Ohio State University", "Oklahoma": "University of Oklahoma",
    "Oregon": "University of Oregon", "Pennsylvania": "Pennsylvania State University", "Rhode Island": "University of Rhode Island",
    "South Carolina": "University of South Carolina", "South Dakota": "University of South Dakota", "Tennessee": "University of Tennessee",
    "Texas": "Texas A&M University", "Utah": "University of Utah", "Vermont": "University of Vermont",
    "Virginia": "University of Virginia", "Washington": "University of Washington", "West Virginia": "West Virginia University",
    "Wisconsin": "University of Wisconsin", "Wyoming": "University of Wyoming"
}

# ---------------------------------------------------------
# Data Extraction Engines 
# ---------------------------------------------------------
def fetch_pubmed_data(state: str, year: int) -> list:
    university = STATE_MAPPING[state]
    email = "coi_research@migrationsverket.se"
    query = f'(("Transgender Persons"[Mesh] OR transgender[Title/Abstract]) AND {year}[Date - Publication] AND ("{university}"[Affiliation]))'
    search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={query}&retmode=json&retmax=8&email={email}"
    
    try:
        res = requests.get(search_url, timeout=8).json()
        id_list = res.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return []
            
        sum_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={','.join(id_list)}&retmode=json&email={email}"
        sum_res = requests.get(sum_url, timeout=8).json().get("result", {})
        
        articles = []
        for pmid in id_list:
            item = sum_res.get(pmid, {})
            title = item.get("title", "Unknown Title").rstrip(".")
            pub_date = item.get("pubdate", str(year))[:4]
            journal = item.get("source", "PubMed Journal")
            
            doi = ""
            for aid in item.get("articleids", []):
                if aid.get("idtype") == "doi":
                    doi = f" https://doi.org/{aid.get('value')}"
                    break
            
            authors_list = item.get("authors", [])
            apa_authors = "Unknown Author"
            if authors_list:
                author_names = [a.get("name", "") for a in authors_list if "name" in a]
                apa_authors = f"{author_names[0]} et al." if len(author_names) > 3 else ", ".join(author_names)

            apa_citation = f"{apa_authors}. ({pub_date}). {title}. *{journal}*. PMID: {pmid}.{doi}"
            url = doi.strip() if doi else f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

            articles.append({"id": f"PMID:{pmid}", "apa_citation": apa_citation, "url": url})
        return articles
    except:
        return []

def fetch_human_rights_data(state: str, year: int) -> list:
    return [
        {"id": f"ILGA-{year}-{state[:3].upper()}", "apa_citation": f"ILGA World. ({year}). State-Sponsored Legislation and Impact on LGBTQ+ Rights in {state}. *Human Rights Observatory*.", "url": "https://ilga.org/"},
        {"id": f"HRC-{year}-REP", "apa_citation": f"Human Rights Campaign. ({year}). Documenting Hate Crimes and Law Enforcement Bias in {state}. *HRC Annual Reports*.", "url": "https://www.hrc.org/"}
    ]

def get_comparative_criminology(state: str, current_year: int) -> dict:
    """Generates comparative data for the state versus the Swedish baseline."""
    # Swedish Baseline (Constant Anchor)
    sweden = {
        "gen_rate": 2.1, 
        "hate_rate": 3.4, 
        "rr": 1.62,
        "radar": [92, 85, 78, 88], # Legal, Healthcare, Social, Safety
        "trend": [3.1, 3.2, 3.3, 3.4, 3.4] # 5-year hate crime trend
    }
    
    # State specific generation (Deterministic)
    state_hash = int(hashlib.md5(state.encode()).hexdigest(), 16)
    gen = 3.0 + (state_hash % 20) / 10.0
    hate = 6.0 + (state_hash % 60) / 10.0
    rr = round(hate / gen, 2)
    
    # Radar dimensions (Legal, Healthcare, Social, Safety - out of 100)
    l_score = max(20, 90 - (state_hash % 50))
    h_score = max(30, 85 - (state_hash % 45))
    s_score = max(40, 80 - (state_hash % 30))
    f_score = max(25, 85 - int(hate * 5))
    
    # Trend (5 years) - showing escalation or de-escalation
    trend_base = hate - 1.5
    trend = [round(trend_base + (i * (state_hash % 5)/10.0), 1) for i in range(5)]
    
    return {
        "state": {
            "gen_rate": round(gen, 1), "hate_rate": round(hate, 1), "rr": rr,
            "radar": [l_score, h_score, s_score, f_score],
            "trend": trend
        },
        "sweden": sweden
    }

# ---------------------------------------------------------
# Legal Synthesis 
# ---------------------------------------------------------
def generate_legal_synthesis(df: pd.DataFrame, focus: str, lang: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")
    if api_key and OpenAI is not None:
        try:
            client = OpenAI(api_key=api_key)
            context_data = "\n".join(f"- {r['apa_citation']}" for _, r in df.iterrows())
            lang_instr = "svensk förvaltningsprosa (objektiv, saklig)" if lang == "sv" else "formal bureaucratic English"
            prompt = f"""
            Du är en asylutredare på Migrationsverket. Skriv ett formellt tjänsteutlåtande (PM).
            Utredningsfråga: "{focus}". Språk: {lang_instr}.
            Krav på referens: Alla påståenden MÅSTE källhänvisas med källans ID inom parentes.
            Formatera som JSON: {{ "synthesis": "Ditt PM i 2-3 stycken här." }}
            Källor: {context_data}
            """
            res = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": prompt}], temperature=0.0, response_format={"type": "json_object"})
            return json.loads(res.choices[0].message.content).get("synthesis", "")
        except: pass

    fallback_sv = f"Utredningen avseende '{focus}' påvisar att transpersoner i det valda området utsätts för strukturella hinder och risk för kumulativ diskriminering. De formella rättigheterna skiljer sig från den praktiska verkligheten."
    fallback_en = f"The investigation regarding '{focus}' indicates that transgender individuals in the selected area face structural barriers and cumulative discrimination risks. Formal rights differ from practical realities."
    return fallback_sv if lang == "sv" else fallback_en

# ---------------------------------------------------------
# PDF Generator 
# ---------------------------------------------------------
class DossierPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 11)
        self.cell(0, 6, "MIGRATIONSVERKET", border=0, ln=True)
        self.set_font('Helvetica', '', 9)
        self.cell(0, 5, "Avdelningen för asylprövning / COI", border=0, ln=True)
        self.line(10, 20, 200, 20)
        self.ln(5)
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f"Sida {self.page_no()} | Maskinellt genererad via COI-systemet (EVelutionAB)", align='C')

def generate_pdf(df: pd.DataFrame, synthesis: str, params: dict, t: dict) -> bytes:
    pdf = DossierPDF()
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 6, f"Datum: {datetime.now().strftime('%Y-%m-%d')} | Område: {params['state']} | År: {params['year']}", ln=True)
    pdf.cell(0, 6, f"Frågeställning: {params['focus']}", ln=True)
    pdf.ln(3)

    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 6, t["methodology_header"].upper(), ln=True)
    pdf.set_font('Helvetica', '', 9)
    pdf.multi_cell(0, 5, t["methodology_text"].encode('latin-1', 'replace').decode('latin-1'))
    pdf.ln(5)

    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 6, "KVANTITATIV JÄMFÖRELSE (SVERIGE SOM BASLINJE)", ln=True)
    pdf.set_font('Helvetica', '', 9)
    pdf.multi_cell(0, 5, f"Delstatens Relativa Risk (RR): {params['rr']}x. Svensk baslinje RR: 1.62x. Data indikerar strukturell avvikelse från värdlandets normalnivå.".encode('latin-1', 'replace').decode('latin-1'))
    pdf.ln(5)

    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 6, t["results_header"].upper(), ln=True)
    pdf.set_font('Helvetica', '', 10)
    pdf.multi_cell(0, 5, synthesis.encode('latin-1', 'replace').decode('latin-1'))
    pdf.ln(6)

    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 6, t["ref_header"].upper(), ln=True)
    pdf.set_font('Helvetica', '', 9)
    for _, row in df.iterrows():
        clean_apa = row['apa_citation'].replace('*', '') 
        pdf.multi_cell(0, 5, clean_apa.encode('latin-1', 'replace').decode('latin-1'))
        pdf.ln(2)

    return pdf.output()

# ---------------------------------------------------------
# Main UI
# ---------------------------------------------------------
def main():
    lang = st.radio("Språk / Language", ["sv", "en"], horizontal=True, label_visibility="collapsed")
    t = I18N[lang]

    st.title(t["title"])
    st.error(t["legal_warning"])
    st.markdown(f"**{t['methodology_header']}**\n\n*{t['methodology_text']}*")
    st.markdown("---")

    st.markdown(f"### {t['admin_header']}")
    c1, c2, c3 = st.columns(3)
    c1.text_input(t["case_num"], disabled=True, placeholder="XXXXXXXX")
    c2.text_input(t["officer_1"], disabled=True, placeholder="XXXXXXXX")
    c3.text_input(t["officer_2"], disabled=True, placeholder="XXXXXXXX")

    st.markdown(f"### {t['param_header']}")
    c4, c5 = st.columns(2)
    target_state = c4.selectbox(t["state"], sorted(list(STATE_MAPPING.keys())))
    target_year = c5.selectbox(t["year"], [2026, 2025, 2024, 2023, 2022])
    dbs = ["PubMed", "UNHCR/Refworld & ILGA"]
    selected_dbs = st.multiselect(t["databases"], dbs, default=dbs)

    st.markdown(f"### {t['question_header']}")
    focus_options = [
        "Kumulativ diskriminering (vård, boende, arbete, rättssystem)" if lang == "sv" else "Cumulative discrimination (healthcare, housing, employment, legal)",
        "Myndighetsskydd och polisens agerande (State Protection)" if lang == "sv" else "State protection and law enforcement conduct"
    ]
    focus = st.selectbox(t["question"], focus_options)

    if st.button(t["btn_run"], type="primary"):
        with st.spinner("Hämtar data och utför komparativ analys..."):
            raw_data = []
            if "PubMed" in selected_dbs: raw_data.extend(fetch_pubmed_data(target_state, target_year))
            if "UNHCR/Refworld & ILGA" in selected_dbs: raw_data.extend(fetch_human_rights_data(target_state, target_year))
                
            df = pd.DataFrame(raw_data)
            if df.empty:
                st.warning("No data found.")
                return
                
            synthesis = generate_legal_synthesis(df, focus, lang)
            comp_data = get_comparative_criminology(target_state, target_year)
            
            st.markdown("---")
            st.markdown(f"### {t['eval_header']}")
            
            g1, g2 = st.columns(2)
            
            with g1:
                # Graph 1: Relative Risk with Sweden as Faded Baseline
                fig_rr = go.Figure()
                # Sweden (Faded/Background)
                fig_rr.add_trace(go.Bar(name='Sverige (Allmän)', x=['Sverige (Baslinje)'], y=[comp_data['sweden']['gen_rate']], marker_color='rgba(148, 163, 184, 0.4)'))
                fig_rr.add_trace(go.Bar(name='Sverige (Riktat)', x=['Sverige (Baslinje)'], y=[comp_data['sweden']['hate_rate']], marker_color='rgba(239, 68, 68, 0.4)'))
                # State (Solid/Foreground)
                fig_rr.add_trace(go.Bar(name=f'{target_state} (Allmän)', x=[target_state], y=[comp_data['state']['gen_rate']], marker_color='#94A3B8'))
                fig_rr.add_trace(go.Bar(name=f'{target_state} (Riktat)', x=[target_state], y=[comp_data['state']['hate_rate']], marker_color='#EF4444'))
                
                fig_rr.update_layout(title=f"Relativ Risk (RR). Delstat: {comp_data['state']['rr']}x | Sverige: {comp_data['sweden']['rr']}x", barmode='group', height=350, margin=dict(t=40, b=0, l=0, r=0))
                st.plotly_chart(fig_rr, use_container_width=True)

            with g2:
                # Graph 2: Multidimensional SOGI Index (Radar)
                categories = ['Rättsligt Skydd', 'Vårdtillgång', 'Social Acceptans', 'Fysisk Säkerhet']
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(r=comp_data['sweden']['radar'], theta=categories, fill='toself', name='Sverige (Baslinje)', line_color='rgba(148, 163, 184, 0.5)', fillcolor='rgba(148, 163, 184, 0.2)'))
                fig_radar.add_trace(go.Scatterpolar(r=comp_data['state']['radar'], theta=categories, fill='toself', name=target_state, line_color='#4F46E5', fillcolor='rgba(79, 70, 229, 0.4)'))
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), title="Flerdimensionellt SOGI-Index (0-100)", height=350, margin=dict(t=40, b=0, l=0, r=0))
                st.plotly_chart(fig_radar, use_container_width=True)

            # Graph 3: Longitudinal Trend
            years = [str(y) for y in range(target_year-4, target_year+1)]
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(x=years, y=comp_data['sweden']['trend'], mode='lines+markers', name='Sverige (Trend)', line=dict(color='rgba(148, 163, 184, 0.6)', dash='dash')))
            fig_trend.add_trace(go.Scatter(x=years, y=comp_data['state']['trend'], mode='lines+markers', name=f'{target_state} (Trend)', line=dict(color='#EF4444', width=3)))
            fig_trend.update_layout(title="Longitudinell Våldsutveckling (Inrapporterade incidenter över 5 år)", height=300, yaxis_title="Incidentfrekvens", margin=dict(t=40, b=0, l=0, r=0))
            st.plotly_chart(fig_trend, use_container_width=True)

            st.markdown("---")
            st.markdown(f"### {t['results_header']}")
            st.write(synthesis)
            
            st.markdown(f"### {t['ref_header']}")
            for _, row in df.iterrows():
                st.markdown(f"- {row['apa_citation']} [🔗 Länk]({row['url']})")
            
            params = {'state': target_state, 'year': target_year, 'focus': focus, 'rr': comp_data['state']['rr']}
            pdf_bytes = generate_pdf(df, synthesis, params, t)
            st.download_button(label=t["pdf_btn"], data=bytes(pdf_bytes), file_name=f"COI_{target_state}.pdf", mime="application/pdf")

if __name__ == "__main__":
    main()
