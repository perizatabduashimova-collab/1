from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="8-апта тест балын болжау", layout="centered")
st.title("1-Зерттеу сұрағы: Сегізінші аптадағы тест балын болжау")
st.write("**Педагогикалық мақсат:** Алғашқы 4 аптаның көрсеткіштері негізінде студенттің 8-аптадағы оқу нәтижесін алдын ала болжап, қолдау шараларын жоспарлау.")

@st.cache_resource
def load_model():
    path = Path(__file__).parent / "research_model.joblib"
    return joblib.load(path)

bundle = load_model()
model = bundle["model"]
features = bundle["features"]

with st.form("prediction_form"):
    st.subheader("1. Бастапқы деңгей көрсеткіші")
    pretest_score = st.slider("Бастапқы (Pretest) тест балы (0-100)", 30, 100, 60)
    
    st.subheader("2. Алғашқы 4 аптадағы жиынтық оқу динамикасы")
    col_a, col_b = st.columns(2)
    with col_a:
        attendance_4weeks = st.slider("4 аптадағы орташа қатысу үлесі (%)", 50, 100, 85)
        submitted_4weeks = st.number_input("4 аптада орындалған жұмыстар саны (барлығы 10)", min_value=3, max_value=10, value=8)
    with col_b:
        late_4weeks = st.number_input("Оның ішінде кеш тапсырылған тапсырмалар", min_value=0, max_value=5, value=1)
    
    submitted = st.form_submit_button("8-ші аптадағы тест балын болжау")

if submitted:
    if late_4weeks > submitted_4weeks:
        st.error("Қате: Кешіктірілген тапсырмалар саны барлық тапсырылған жұмыстардан аспауы тиіс!")
    else:
        # Модельге енгізілетін деректер
        row = pd.DataFrame([{
            "pretest_score": pretest_score,
            "attendance_4weeks": attendance_4weeks,
            "submitted_4weeks": submitted_4weeks,
            "late_4weeks": late_4weeks
        }])
        
        # 8-ші аптаның болжамды балын есептеу
        prediction = float(model.predict(row[features])[0])
        
        st.markdown("---")
        st.subheader("Болжам нәтижесі:")
        st.metric("8-ші аптадағы болжамды пәндік тест балы", f"{prediction:.1f} / 100 балл")
        
        # Педагогикалық кеңес/талдау
        if prediction < 60:
            st.warning("⚠️ **Ескерту:** Студенттің 8-ші аптадағы нәтижесі төмен болуы мүмкін. Қосымша кеңес беру (консультация) ұсынылады.")
        elif prediction >= 85:
            st.success("✅ **Өте жақсы:** Студенттің 4 апталық оқу қарқыны жоғары, оқу үдерісі сәтті өтуде[cite: 1].")
        else:
            st.info("ℹ️ **Қалыпты:** Студенттің оқу үлгерімі орташа деңгейде[cite: 1].")
            
        st.caption("Шектеу: Бұл модель баға қоюға негіз емес, тек педагогикалық қолдауды жоспарлауға арналған[cite: 1].")
