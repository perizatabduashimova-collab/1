from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Оқу нәтижесін болжау", layout="centered")
st.title("1-Зерттеу сұрағы: 8-ші аптадағы тест балын болжау")
st.write("**Зерттеу сұрағы:** Алғашқы төрт аптадағы оқу көрсеткіштері бойынша сегізінші аптадағы пәндік тест балын болжау[cite: 1].")
st.info("Бұл модель Ridge регрессиясы арқылы 300 жасанды дерек негізінде оқытылған.")

@st.cache_resource
def load_model():
    path = Path(__file__).parent / "research_model.joblib"
    return joblib.load(path)

bundle = load_model()
model = bundle["model"]
features = bundle["features"]

with st.form("prediction_form"):
    st.subheader("Алғашқы 4 апта көрсеткіштерін енгізіңіз[cite: 1]:")
    pretest_score = st.slider("Бастапқы тест балы (0-100)", 30, 100, 60)
    attendance_4weeks = st.slider("Алғашқы 4 аптадағы сабаққа қатысу үлесі (%)[cite: 1]", 50, 100, 80)
    submitted_4weeks = st.number_input("Алғашқы 4 аптада орындалған жұмыстар саны[cite: 1]", min_value=3, max_value=10, value=7, step=1)
    late_4weeks = st.number_input("Оның ішінде кеш тапсырылған жұмыстар саны[cite: 1]", min_value=0, max_value=3, value=1, step=1)
    
    submitted = st.form_submit_button("Болжам жасау")

if submitted:
    if late_4weeks > submitted_4weeks:
        st.error("Қате: Кеш тапсырылған жұмыстар саны орындалған жұмыстардан артық болмауы тиіс!")
    else:
        row = pd.DataFrame([{
            "pretest_score": pretest_score,
            "attendance_4weeks": attendance_4weeks,
            "submitted_4weeks": submitted_4weeks,
            "late_4weeks": late_4weeks
        }])
        
        prediction = float(model.predict(row[features])[0])
        st.metric("Болжамды тест балы (8-ші апта)[cite: 1]", f"{prediction:.1f} / 100")
        
        # Қосымша көрсеткіштер мен пайыздарды шығару
        completion_pct = (submitted_4weeks / 10) * 100
        late_pct = (late_4weeks / submitted_4weeks * 100) if submitted_4weeks > 0 else 0
        
        col1, col2 = st.columns(2)
        col1.metric("Тапсырмаларды орындау үлесі", f"{completion_pct:.0f}%")
        col2.metric("Кеш тапсыру үлесі", f"{late_pct:.1f}%")
        
        st.caption("Бұл болжам тек оқу үдерісін жоспарлауға көмектеседі және түпкілікті баға болып саналмайды[cite: 1].")
