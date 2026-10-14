import streamlit as st
from google import genai
import docx
import pypdf
import io

# Настройка страницы
st.set_page_config(
    page_title="VeritasAI - Детектор текста", 
    page_icon="🛡️", 
    layout="wide"
)

# Пытаемся получить ключ из секретов Streamlit Cloud
# Если запутаете локально, можно подставить запасной вариант или оставить поле
server_api_key = st.secrets.get("GEMINI_API_KEY", "")

# Боковая панель для настроек (теперь чистая и красивая)
with st.sidebar:
    st.header("⚙️ О проекте")
    st.markdown("**VeritasAI** — интеллектуальная система анализа студенческих работ на базе Google Gemini.")
    st.markdown("---")
    st.markdown("### Инструкция:")
    st.markdown("1. Вставьте текст или загрузите документ.")
    st.markdown("2. Нажмите кнопку запуска проверки.")
    
    # Если ключ на сервере не задан, оставляем поле для ввода (удобно для локального теста)
    if not server_api_key:
        api_key_input = st.text_input("Gemini API Key", type="password")
    else:
        api_key_input = server_api_key

st.title("🛡️ VeritasAI: Академический антиплагиат и детектор ИИ")
st.markdown("Инструмент анализа курсовых, дипломов и эссе на наличие маркеров генерации искусственным интеллектом.")

# Ввод текста
tab1, tab2 = st.tabs(["📝 Вставить текст вручную", "📁 Загрузить документ (PDF / Word / TXT)"])

extracted_text = ""

with tab1:
    manual_text = st.text_area("Введите текст работы для анализа:", height=200, placeholder="Скопируйте сюда текст...")
    if manual_text:
        extracted_text = manual_text

with tab2:
    uploaded_file = st.file_uploader("Выберите файл", type=["txt", "docx", "pdf"])
    if uploaded_file is not None:
        ext = uploaded_file.name.split(".")[-1].lower()
        try:
            if ext == "txt":
                extracted_text = uploaded_file.getvalue().decode("utf-8")
            elif ext == "docx":
                doc = docx.Document(io.BytesIO(uploaded_file.getvalue()))
                extracted_text = "\n".join([p.text for p in doc.paragraphs])
            elif ext == "pdf":
                reader = pypdf.PdfReader(io.BytesIO(uploaded_file.getvalue()))
                extracted_text = "".join([page.extract_text() + "\n" for page in reader.pages if page.extract_text()])
            
            if extracted_text.strip():
                st.success(f"Файл успешно прочитан! Символов: {len(extracted_text)}")
        except Exception as e:
            st.error(f"Ошибка при чтении файла: {e}")

st.markdown("---")

# Кнопка проверки
if st.button("🔍 Запустить глубокий анализ", type="primary", use_container_width=True):
    if not api_key_input.strip():
        st.error("⚠️ API-ключ не настроен на сервере. Обратитесь к администратору проекта.")
    elif not extracted_text.strip():
        st.warning("⚠️ Текст для анализа пустой. Введите текст или загрузите документ.")
    else:
        with st.spinner("Модель Gemini анализирует структуру текста и маркеры ИИ..."):
            try:
                # Инициализация клиента Gemini
                client = genai.Client(api_key=api_key_input)
                
                prompt = f"""
                Ты строгий академический эксперт по информационной безопасности текстов и детекции следов искусственного интеллекта или глубокого рерайтинга.
                Проанализируй предоставленный текст. Оцени вероятность (в процентах от 0 до 100%), что текст написан ИИ или подвергся машинному рерайту.
                
                Выдай ответ на русском языке в строгом структурированном виде (используй Markdown):
                1. **Вероятность ИИ-происхождения:** [X]%
                2. **Итоговый вердикт:** [Написано человеком / Смешанный текст / Сгенерировано ИИ / Глубокий рерайт]
                3. **Анализ стиля и лексики:** (Опиши предсказуемость фраз, монотонность, наличие шаблонных конструкций).
                4. **Подозрительные фрагменты:** (Выдели цитаты из текста, которые вызывают больше всего сомнений).

                Текст для анализа:
                {extracted_text}
                """

                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt,
                )
                
                st.success("Анализ успешно завершен!")
                st.markdown("### 📊 Отчет об академической проверке:")
                st.markdown(response.text)
                
            except Exception as e:
                st.error(f"Произошла ошибка при обращении к API Gemini: {e}")
