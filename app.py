import streamlit as st
import openai
import re
import math
from io import BytesIO

# Try importing document readers
try:
    import docx
except ImportError:
    docx = None

try:
    import pypdf
except ImportError:
    pypdf = None

# Page configuration
st.set_page_config(
    page_title="AI & Paraphrase Detector", 
    page_icon="🛡️", 
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    .stAlert {
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ Академический детектор ИИ-текста и рерайтинга")
st.markdown("Интеллектуальная система анализа курсовых, дипломов и эссе на наличие следов генерации нейросетями и глубокого рерайтинга.")

# Sidebar configuration
st.sidebar.header("Настройки анализа")
api_key_input = st.sidebar.text_input("OpenRouter / OpenAI API Key", type="password", help="Введите ваш API ключ для доступа к языковой модели")
selected_model = st.sidebar.selectbox(
    "Модель для аудита",
    ["openai/gpt-4o-mini", "anthropic/claude-3-haiku", "google/gemini-flash-1.5"],
    help="Выберите модель для глубокого семантического анализа"
)

# Helper functions for text metrics
def calculate_text_metrics(text):
    if not text.strip():
        return 0, 0, 0, 0
    
    # Split text into sentences and words
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    words = re.findall(r'\b\w+\b', text.lower())
    
    total_words = len(words)
    total_sentences = len(sentences)
    
    if total_sentences == 0 or total_words == 0:
        return 0, 0, 0, 0
        
    avg_sentence_length = total_words / total_sentences
    
    # Calculate lexical diversity (Type-Token Ratio)
    unique_words = set(words)
    ttr = len(unique_words) / total_words
    
    # Calculate burstiness (standard deviation of sentence lengths in words)
    sentence_lengths = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
    mean_len = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
    variance = sum((l - mean_len) ** 2 for l in sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
    burstiness = math.sqrt(variance)
    
    return total_words, round(avg_sentence_length, 1), round(ttr * 100, 1), round(burstiness, 1)

def extract_text_from_file(uploaded_file):
    text = ""
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension == 'txt':
        text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    elif file_extension == 'docx':
        if docx is not None:
            doc = docx.Document(BytesIO(uploaded_file.getvalue()))
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        else:
            st.error("Библиотека python-docx не установлена.")
    elif file_extension == 'pdf':
        if pypdf is not None:
            reader = pypdf.PdfReader(BytesIO(uploaded_file.getvalue()))
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        else:
            st.error("Библиотека pypdf не установлена.")
    return text

# Input method selection
tab1, tab2 = st.tabs(["📝 Ввести текст вручную", "📂 Загрузить документ (PDF / Word / TXT)"])

text_to_analyze = ""

with tab1:
    manual_text = st.text_area("Вставьте текст работы:", height=250, placeholder="Скопируйте сюда текст курсовой или эссе...")
    if manual_text:
        text_to_analyze = manual_text

with tab2:
    uploaded_file = st.file_uploader("Загрузить файл", type=["txt", "docx", "pdf"])
    if uploaded_file is not None:
        with st.spinner("Извлечение текста из файла..."):
            extracted_text = extract_text_from_file(uploaded_file)
            if extracted_text:
                text_to_analyze = extracted_text
                st.success(f"Файл успешно загружен! Распознано символов: {len(text_to_analyze)}")
                with st.expander("Посмотреть извлеченный текст"):
                    st.write(text_to_analyze[:1500] + ("..." if len(text_to_analyze) > 1500 else ""))

# Main analysis trigger
if st.button("🔍 Запустить глубокий анализ", type="primary"):
    if not text_to_analyze.strip():
        st.warning("Пожалуйста, введите текст или загрузите файл для проверки.")
    elif not api_key_input.strip():
        st.error("Пожалуйста, укажите ваш API ключ (OpenRouter / OpenAI) в боковой панели слева.")
    else:
        # Calculate local metrics
        words_count, avg_len, ttr, burstiness = calculate_text_metrics(text_to_analyze)
        
        st.markdown("---")
        st.subheader("📊 Первичные метрики текста")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Всего слов", words_count)
        with col2:
            st.metric("Ср. длина предложения", f"{avg_len} слов")
        with col3:
            st.metric("Лексическое разнообразие", f"{ttr}%")
        with col4:
            st.metric("Скачкообразность (Burstiness)", burstiness, help="Низкое значение часто характерно для ИИ-текстов с монотонной длиной предложений")
        
        with st.spinner("Проводится глубокий ИИ-аудит структуры, лексики и маркеров рерайтинга..."):
            try:
                client = openai.OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key_input.strip(),
                )
                
                system_prompt = (
                    "Ты профессиональный академический эксперт по детекции текстов, созданных искусственным интеллектом, "
                    "и систем глубокого рерайтинга. Проведи строгий анализ представленного текста."
                )
                
                user_prompt = f"""
                Проанализируй следующий текст на наличие следов генерации нейросетями (ChatGPT, Claude, YandexGPT и т.д.) или глубокого рерайта.
                Учитывай метрики: лексическое разнообразие={ttr}%, скачкообразность (burstiness)={burstiness}.
                
                Выдай ответ в структурированном виде на русском языке по пунктам:
                1. **Вероятность ИИ-происхождения:** [X]%
                2. **Итоговый вердикт:** (Написано человеком / Смешанный текст / Вероятно, сгенерировано ИИ / Глубокий рерайтинг)
                3. **Анализ стиля и лексики:** (Опиши клише, предсказуемость фраз, монотонность предложений)
                4. **Подозрительные фрагменты:** (Выдели цитаты из текста, которые вызывают наибольшие подозрения)
                5. **Рекомендации для преподавателя:** (Стоит ли отправлять работу на доработку)

                Текст для анализа:
                {text_to_analyze[:4000]}  # Ограничение длины для токенов
                """

                response = client.chat.completions.create(
                    model=selected_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2
                )
                
                ai_report = response.choices[0].message.content
                
                st.markdown("---")
                st.subheader("📑 Экспертный отчет системы аудита")
                st.markdown(ai_report)
                
            except Exception as e:
                st.error(f"Произошла ошибка при обращении к API: {e}")
