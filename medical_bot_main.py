import os
import asyncio
import logging
import sys
from io import BytesIO
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from zoneinfo import ZoneInfo

import uvicorn
from fastapi import FastAPI
import aiohttp
from PIL import Image

from aiogram import Bot, Dispatcher, types
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.types import Message, InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery
from aiogram.client.default import DefaultBotProperties

import google.generativeai as genai

# ═══════════════════════════════════════════════════════════════
# ⚙️ КОНФИГУРАЦИЯ
# ═══════════════════════════════════════════════════════════════

TOKEN = os.getenv("TELEGRAM_TOKEN")
GOOGLE_KEYS = [
    os.getenv("GOOGLE_API_KEY"),
    os.getenv("GOOGLE_API_KEY_2"),
    os.getenv("GOOGLE_API_KEY_3"),
    os.getenv("GOOGLE_API_KEY_4"),
    os.getenv("GOOGLE_API_KEY_5"),
    os.getenv("GOOGLE_API_KEY_6"),
]
RENDER_URL = os.getenv("RENDER_EXTERNAL_URL")

GOOGLE_KEYS = [k for k in GOOGLE_KEYS if k]

generation_config = {
    "temperature": 0.2,
    "top_p": 0.8,
    "top_k": 40,
    "max_output_tokens": 4096,
}

MSK_TZ = ZoneInfo("Europe/Moscow")

# ═══════════════════════════════════════════════════════════════
# 📚 СИСТЕМНЫЕ ПРОМТЫ
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT_GENERAL_MEDICINE = """Ты — исследователь-аналитик и академический ассистент для общей медицины.

⚠️ ДИСКЛЕЙМЕР:
Все клинические решения требуют консультации с лицензированным врачом.

ПРИНЦИП РАБОТЫ:
├─ Поиск достоверных источников (PubMed, Cochrane, официальные гайдлайны)
├─ Анализ эффективности методов через GRADE оценку
├─ Указание авторов, лет публикации и уровня доказательности
├─ Отказ от выдуманных данных и источников
└─ Честное указание пробелов в знаниях

📚 ОФИЦИАЛЬНЫЕ ИСТОЧНИКИ:
PubMed/PMC, Cochrane Library, Web of Science, Scopus (peer-review)
Гайдлайны: WHO, CDC, ESC, ADA, GOLD, EASL, Минздрав РФ, NICE

🎯 СТРУКТУРА ОТВЕТА:
Используй простой текстовый формат. Пример:

📌 **Исследование:** (название)
   Год: 2023 | Авторы: (имена)
   Метод: (описание)
   Результат: (описание)
   Эффективность: (данные с 95% CI)
   GRADE: High/Moderate/Low
   PMID: 12345678

🛡️ КОНТРОЛЬ КАЧЕСТВА:
- Если данных нет → "Данные отсутствуют в открытых источниках"
- НЕ выдумывай PMID, авторов, цифры
- ВСЕГДА указывай источник (PMID/DOI)
- При расхождениях → объясни почему
- Данные старше 5 лет → отметь как "историческая справка"

⚠️ ЛОКАЛЬНАЯ РЕЛЕВАНТНОСТЬ:
Указывай, где подход РФ отличается от международных стандартов

📝 СТИЛЬ:
- Ясно, академично, без просторечий
- Факты впереди мнений
- Цифры: OR, RR, 95% CI
- МАКСИМУМ 3000 символов!"""

SYSTEM_PROMPT_GYNECOLOGY = """Ты — клинический ассистент-аналитик по гинекологии и акушерству.

⚠️ ДИСКЛЕЙМЕР:
Все клинические решения требуют консультации с лицензированным врачом.

ПРИНЦИП РАБОТЫ:
├─ Анализ клинических рекомендаций от ACOG, RCOG, ESHRE, Минздрава РФ
├─ Оценка эффективности методов лечения и диагностики
├─ Указание уровня доказательности (GRADE, Evidence Level)
├─ Честное указание авторов и их вклада
└─ Отказ от выдуманных источников и данных

📚 ПРИОРИТЕТНЫЕ ГАЙДЛАЙНЫ:
1. RCOG (Royal College)
2. ACOG (American College)
3. ESHRE (European Society)
4. DGG/DGGG (Германия)
5. Минздрав РФ

🎯 СТРУКТУРА ОТВЕТА:
Используй простой текстовый формат. Пример:

📌 **Guideline:** (название)
   Организация: RCOG/ACOG/ESHRE
   Год: 2023
   Авторы: (список)
   Метод: (описание)
   Результат: (описание)
   Эффективность: (данные)
   GRADE: High/Moderate/Low
   PMID/DOI: (ссылка)

🛡️ КОНТРОЛЬ КАЧЕСТВА:
- Если данных нет → явно отмечай "Данные отсутствуют"
- НЕ создавай вымышленные PMID или авторов
- ВСЕГДА ссылка на источник для каждого факта
- При расхождении RCOG vs ACOG → объясни разницу методологий
- Рекомендации старше 5-7 лет → отметь как требующие проверки

⚠️ СПЕЦИФИКА ГИНЕКОЛОГИИ:
- RCOG часто консервативнее ACOG
- ESHRE специализируется на ВРТ
- Минздрав РФ может рекомендовать другие препараты
- ВСЕГДА отмечай: "В [стране] подход иной из-за [причина]"

📝 СТИЛЬ:
- Ясно, логично, академично
- Не копируй абстракты - переформулируй
- Цифры с доверительными интервалами (95% CI)
- МАКСИМУМ 3000 символов!"""

# ═══════════════════════════════════════════════════════════════
# 🤖 СИСТЕМА УПРАВЛЕНИЯ МОДЕЛЯМИ
# ═══════════════════════════════════════════════════════════════

class ModelManager:
    """Управляет доступными моделями и их лимитами."""
    
    def __init__(self):
        self.api_key_index = 0
        self.current_model = None
        self.current_model_name = "Searching..."
        self.model_limits = {}
    
    def get_models(self):
        """Получает список доступных моделей."""
        models = []
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    name = m.name.replace("models/", "")
                    if "gemini" in name:
                        models.append(name)
        except:
            pass
        
        fallback = ["gemini-2.0-flash-exp", "gemini-1.5-flash", "gemini-1.5-flash-8b"]
        for m in fallback:
            if m not in models:
                models.append(m)
        
        return models
    
    async def find_working_model(self):
        """Находит рабочую модель на текущем API ключе."""
        models = self.get_models()
        
        for model_name in models:
            if self.model_limits.get(model_name, {}).get(self.api_key_index, False):
                continue
            
            try:
                test_model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=generation_config,
                    system_instruction=SYSTEM_PROMPT_GENERAL_MEDICINE
                )
                response = await test_model.generate_content_async("test")
                
                if response and response.text:
                    self.current_model = test_model
                    self.current_model_name = model_name
                    print(f"✅ Модель: {model_name}")
                    return True
            except Exception as e:
                if "429" in str(e):
                    if model_name not in self.model_limits:
                        self.model_limits[model_name] = {}
                    self.model_limits[model_name][self.api_key_index] = True
        
        return False
    
    async def switch_api(self):
        """Переключает на следующий API ключ."""
        old_index = self.api_key_index
        
        for i in range(len(GOOGLE_KEYS)):
            next_index = (self.api_key_index + 1) % len(GOOGLE_KEYS)
            if next_index == old_index:
                return False
            
            self.api_key_index = next_index
            try:
                genai.configure(api_key=GOOGLE_KEYS[self.api_key_index])
                print(f"🔄 API #{self.api_key_index + 1}")
                
                if await self.find_working_model():
                    return True
            except:
                pass
        
        return False

model_manager = ModelManager()

# ═══════════════════════════════════════════════════════════════
# 📋 ИНИЦИАЛИЗАЦИЯ
# ═══════════════════════════════════════════════════════════════

bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN))
dp = Dispatcher()
app = FastAPI()

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

USER_STATES = {}

def get_user_state(user_id: int) -> Dict:
    """Получает или создаёт состояние пользователя."""
    if user_id not in USER_STATES:
        USER_STATES[user_id] = {
            "mode": "medicine_general",
            "conversation_history": [],
            "last_activity": datetime.now(MSK_TZ)
        }
    
    USER_STATES[user_id]["last_activity"] = datetime.now(MSK_TZ)
    return USER_STATES[user_id]

# ═══════════════════════════════════════════════════════════════
# 🎯 РАСШИРЕННЫЕ ТРИГГЕРЫ (ТОЧНОЕ СОВПАДЕНИЕ)
# ═══════════════════════════════════════════════════════════════

TRIGGER_WORDS_MAPPING = {
    # Гинекология
    "!ген": "gynecology",
    "!гениколог": "gynecology",
    "!гинеколог": "gynecology",
    "!гин": "gynecology",
    
    # Общая медицина
    "!док": "doctor",
    "!доктор": "doctor",
    "!врач": "doctor",
    "!медицина": "doctor",
    "!med": "doctor",
    
    # Информация
    "!инфо": "info",
    "!информация": "info",
    "!помощь": "info",
    "!help": "info",
    "!справка": "info",
    
    # Старт
    "!старт": "start",
    "!start": "start",
    "!начать": "start",
    
    # Очистка памяти
    "!очистить": "refresh",
    "!обнови": "refresh",
    "!очисти": "refresh",
    "!забудь": "refresh",
    "!refresh": "refresh",
}

def check_for_triggers(text: str) -> Optional[str]:
    """
    Проверяет ТОЧНОЕ совпадение триггеров.
    Триггер должен быть отдельным словом, не частью другого слова.
    """
    if not text:
        return None
    
    text_lower = text.strip().lower()
    
    # Разбиваем текст на слова (по пробелам)
    words = text_lower.split()
    
    # Проверяем каждое слово на точное совпадение с триггером
    for word in words:
        if word in TRIGGER_WORDS_MAPPING:
            action = TRIGGER_WORDS_MAPPING[word]
            print(f"🔴 ТОЧНЫЙ ТРИГГЕР ОБНАРУЖЕН: '{word}' → Действие: {action}")
            return action
    
    # Если триггер не найден - возвращаем None
    return None

# ═══════════════════════════════════════════════════════════════
# 🔧 ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ═══════════════════════════════════════════════════════════════

def get_mode_buttons() -> InlineKeyboardMarkup:
    """Клавиатура для выбора режима."""
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="🏥 Общая медицина", callback_data="mode_general"),
            InlineKeyboardButton(text="👶 Гинекология", callback_data="mode_gyn"),
        ]
    ])
    return keyboard

async def is_addressed_to_bot(message: Message, bot_user: types.User) -> bool:
    """Проверяет, адресовано ли сообщение боту."""
    if message.chat.type == "private":
        return True
    if message.reply_to_message and message.reply_to_message.from_user.id == bot_user.id:
        return True
    if message.text and f"@{bot_user.username}" in message.text:
        return True
    if message.caption and f"@{bot_user.username}" in message.caption:
        return True
    return False

async def prepare_prompt_parts(message: Message, bot_user: types.User) -> Tuple[List, List]:
    """Подготавливает части промта."""
    prompt_parts = []
    temp_files_to_delete = []
    
    text_content = ""
    if message.text:
        text_content = message.text.replace(f"@{bot_user.username}", "").strip()
    elif message.caption:
        text_content = message.caption.replace(f"@{bot_user.username}", "").strip()
    
    if text_content:
        prompt_parts.append(text_content)
    
    if message.photo:
        try:
            print(f"📸 Загружаю фото...")
            photo_id = message.photo[-1].file_id
            file_info = await bot.get_file(photo_id)
            img_data = BytesIO()
            await bot.download_file(file_info.file_path, img_data)
            img_data.seek(0)
            image = Image.open(img_data)
            
            prompt_parts.append(image)
            print(f"✅ Фото добавлено")
        except Exception as e:
            print(f"❌ Ошибка фото: {e}")
    
    return prompt_parts, temp_files_to_delete

async def send_long_message(message: Message, text: str, max_length: int = 4096):
    """Отправляет длинное сообщение, разбивая его на части."""
    if len(text) <= max_length:
        await message.reply(text, parse_mode=ParseMode.MARKDOWN)
        return
    
    # Разбиваем по абзацам
    parts = []
    current_part = ""
    
    paragraphs = text.split("\n")
    for paragraph in paragraphs:
        if len(current_part) + len(paragraph) + 1 <= max_length:
            current_part += paragraph + "\n"
        else:
            if current_part:
                parts.append(current_part.strip())
            current_part = paragraph + "\n"
    
    if current_part:
        parts.append(current_part.strip())
    
    # Отправляем части
    for i, part in enumerate(parts):
        if part:
            if i < len(parts) - 1:
                await message.reply(part + "\n\n_[часть " + str(i+1) + "/" + str(len(parts)) + "]_", parse_mode=ParseMode.MARKDOWN)
            else:
                await message.reply(part, parse_mode=ParseMode.MARKDOWN)
            
            # Небольшая задержка между сообщениями
            await asyncio.sleep(0.5)

async def process_message(message: Message, bot_user: types.User, text_content: str, 
                          prompt_parts: List, user_state: Dict):
    """Обработка сообщения."""
    try:
        if user_state["mode"] == "medicine_general":
            system_prompt = SYSTEM_PROMPT_GENERAL_MEDICINE
            mode_name = "🏥 Общая медицина"
        else:
            system_prompt = SYSTEM_PROMPT_GYNECOLOGY
            mode_name = "👶 Гинекология"
        
        print(f"\n📨 Запрос от {message.from_user.id} [{mode_name}]")
        print(f"   Модель: {model_manager.current_model_name}")
        
        conversation_history = user_state["conversation_history"]
        
        current_model = genai.GenerativeModel(
            model_name=model_manager.current_model_name,
            generation_config=generation_config,
            system_instruction=system_prompt
        )
        
        if conversation_history:
            full_prompt = conversation_history + [{"role": "user", "parts": prompt_parts}]
        else:
            full_prompt = [{"role": "user", "parts": prompt_parts}]
        
        response = await current_model.generate_content_async(full_prompt)
        
        if response.text:
            print(f"✅ Ответ получен ({len(response.text)} символов)")
            
            user_state["conversation_history"].append({
                "role": "user",
                "parts": [text_content]
            })
            user_state["conversation_history"].append({
                "role": "model",
                "parts": [response.text]
            })
            
            if len(user_state["conversation_history"]) > 20:
                user_state["conversation_history"] = user_state["conversation_history"][-20:]
            
            answer_text = response.text
            
            # Отправляем длинное сообщение
            await send_long_message(message, answer_text)
            print(f"✅ Ответ отправлен")
            return True
        
        else:
            await message.reply("⚠️ Пустой ответ от модели")
            return False
    
    except Exception as e:
        error_str = str(e)
        print(f"❌ Ошибка: {error_str[:100]}")
        
        if "429" in error_str or "quota" in error_str or "RESOURCE_EXHAUSTED" in error_str:
            print(f"⚠️ Лимит")
            
            if await model_manager.switch_api():
                await model_manager.find_working_model()
                return await process_message(message, bot_user, text_content, prompt_parts, user_state)
            
            await message.reply(
                "❌ Все лимиты исчерпаны на данный момент.\n"
                "Лимиты обновляются каждые 24 часа.\n"
                "Попробуйте позже! 🕐"
            )
            return False
        
        else:
            await message.reply(f"❌ Ошибка: {error_str[:100]}")
            return False

async def handle_trigger_action(message: Message, action: str, bot_user: types.User):
    """Обрабатывает триггер-действие."""
    user_id = message.from_user.id
    user_state = get_user_state(user_id)
    
    if action == "doctor":
        user_state["mode"] = "medicine_general"
        await message.answer(
            "🏥 **РЕЖИМ: Общая медицина** ✅\n\n"
            "Готов анализировать гайдлайны по кардиологии, инфекциям, пульмологии и др.\n"
            "Используй рекомендации: WHO, CDC, ESC, ADA, GOLD и другие\n\n"
            "_Триггеры: !док, !доктор, !врач, !медицина_\n\n"
            "📝 Задай свой вопрос! 👇"
        )
        print(f"✅ {message.from_user.first_name} переключился на режим ОБЩЕЙ МЕДИЦИНЫ")
    
    elif action == "gynecology":
        user_state["mode"] = "medicine_gynecology"
        await message.answer(
            "👶 **РЕЖИМ: Гинекология и акушерство** ✅\n\n"
            "Готов анализировать рекомендации ACOG, RCOG, ESHRE и Минздрава РФ\n"
            "Репродуктивная медицина, менструальные расстройства, ВРТ, беременность\n\n"
            "_Триггеры: !ген, !гениколог, !гинеколог, !гин_\n\n"
            "📝 Задай свой вопрос! 👇"
        )
        print(f"✅ {message.from_user.first_name} переключился на режим ГИНЕКОЛОГИИ")
    
    elif action == "info":
        print(f"ℹ️ {message.from_user.first_name} запросил информацию")
        await command_info_handler(message)
    
    elif action == "start":
        print(f"🔄 {message.from_user.first_name} запросил /start")
        await command_start_handler(message)
    
    elif action == "refresh":
        user_state["conversation_history"] = []
        await message.answer(
            "🗑️ **История диалога очищена!** ✅\n\n"
            "Бот больше не помнит предыдущих сообщений.\n"
            "Начинаем диалог с чистого листа! 📄\n\n"
            "_Триггеры: !очистить, !обнови, !очисти, !забудь_"
        )
        print(f"✅ {message.from_user.first_name} очистил историю диалога")

# ═══════════════════════════════════════════════════════════════
# 📝 CALLBACK ХЕНДЛЕРЫ
# ═══════════════════════════════════════════════════════════════

@dp.callback_query()
async def handle_mode_callback(query: CallbackQuery):
    """Обработка переключения режимов."""
    user_id = query.from_user.id
    user_state = get_user_state(user_id)
    
    callback_data = query.data
    
    if callback_data == "mode_general":
        user_state["mode"] = "medicine_general"
        message_text = (
            "🏥 **Режим: Общая медицина**\n\n"
            "**Специализация:** Кардиология, инфекции, пульмология, гастроэнтерология, эндокринология и др.\n\n"
            "**Принцип работы:**\n"
            "• Ищу официальные гайдлайны и исследования (PubMed, Cochrane, CDC, WHO и т.д.)\n"
            "• Оцениваю эффективность методов через GRADE систему\n"
            "• Указываю авторов, годы и уровень доказательности\n"
            "• Объясняю расхождения между странами (РФ vs Запад)\n"
            "• Честно говорю, где данных нет\n\n"
            "⚠️ Все клинические решения требуют консультации с лицензированным врачом."
        )
        
    elif callback_data == "mode_gyn":
        user_state["mode"] = "medicine_gynecology"
        message_text = (
            "👶 **Режим: Гинекология и акушерство**\n\n"
            "**Специализация:** Репродуктивная медицина, менструальные расстройства, ВРТ, беременность и т.д.\n\n"
            "**Принцип работы:**\n"
            "• Ищу рекомендации ACOG, RCOG, ESHRE, Минздрава РФ\n"
            "• Анализирую клинические исследования с высоким уровнем доказательности\n"
            "• Указываю авторов и их вклад в науку\n"
            "• Объясняю различия в подходах между странами\n"
            "• Отмечаю пробелы в знаниях\n\n"
            "⚠️ Все клинические решения требуют консультации с лицензированным врачом."
        )
    else:
        return
    
    try:
        await query.message.edit_text(
            message_text,
            reply_markup=get_mode_buttons(),
            parse_mode=ParseMode.MARKDOWN
        )
        await query.answer(f"✅ Режим переключен", show_alert=False)
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        await query.answer("❌ Ошибка", show_alert=True)

# ═══════════════════════════════════════════════════════════════
# 🎮 ИНФОРМАЦИЯ
# ═══════════════════════════════════════════════════════════════

INFO_TEXT = """🏥 **МЕДИЦИНСКИЙ АССИСТЕНТ V4.0 - ИНСТРУКЦИЯ**

**ЧТО УМЕЕТ БОТ:**
Анализирует медицинские гайдлайны и исследования для:
• 🏥 Общей медицины (кардиология, инфекции, пульмология и др.)
• 👶 Гинекологии и акушерства
• 📸 Анализирует картинки через Google AI

**ВАЖНО:** Все клинические решения требуют консультации с лицензированным врачом!

═══════════════════════════════════════════════════════════════

**КОМАНДЫ:**

/start - Главное меню и список команд
/info - Эта инструкция
/medic - Режим "Общая медицина"
/gen - Режим "Гинекология"
/refresh - Очистить историю диалога

═══════════════════════════════════════════════════════════════

**ТРИГГЕР-СЛОВА (вызывают режимы автоматически):**

*Гинекология:*
!ген, !гениколог, !гинеколог, !гин

*Общая медицина:*
!док, !доктор, !врач, !медицина, !med

*Информация/Помощь:*
!инфо, !информация, !помощь, !help, !справка

*Главное меню:*
!старт, !start, !начать

*Очистить память:*
!очистить, !обнови, !очисти, !забудь, !refresh

═══════════════════════════════════════════════════════════════

**КАК ИСПОЛЬЗОВАТЬ:**

1️⃣ Выбери режим:
   • /medic или !док для общей медицины
   • /gen или !ген для гинекологии

2️⃣ Напиши свой вопрос:
   "Какая эффективность метформина при СПКЯ?"
   "Какие гайдлайны по лечению пневмонии?"

3️⃣ Или отправь картинку:
   Бот проанализирует её через Google AI

4️⃣ Бот ответит с:
   • Официальными гайдлайнами (RCOG, ACOG, WHO и т.д.)
   • Уровнем доказательности (GRADE)
   • PMID исследований
   • Различиями между странами

5️⃣ Бот запомнит контекст диалога:
   Можешь задать уточняющие вопросы, он будет помнить предыдущие

6️⃣ Очистить память:
   /refresh или !обнови

═══════════════════════════════════════════════════════════════

**ПРИМЕРЫ ВОПРОСОВ:**

🏥 Общая медицина:
"Актуальные гайдлайны по лечению гипертензии у беременных"
"Эффективность антибиотиков при бактериальной пневмонии"
"GRADE оценка методов диагностики сахарного диабета"

👶 Гинекология:
"Рекомендации ACOG и RCOG по ведению СПКЯ"
"Эффективность ВРТ при трубном факторе бесплодия"
"Протоколы лечения эндометриоза согласно ESHRE"

═══════════════════════════════════════════════════════════════

**ДИСКЛЕЙМЕР:**
⚠️ Все клинические решения требуют консультации с лицензированным врачом
⚠️ Информация может быть неполной - всегда проверяй источники

═══════════════════════════════════════════════════════════════

Вопросы? Команда /start в любой момент!"""

# ═══════════════════════════════════════════════════════════════
# 🎮 КОМАНДЫ
# ═══════════════════════════════════════════════════════════════

@dp.message(CommandStart())
async def command_start_handler(message: Message):
    """Стартовое сообщение."""
    user_id = message.from_user.id
    user_state = get_user_state(user_id)
    
    api_info = f" (API #{model_manager.api_key_index + 1}/{len(GOOGLE_KEYS)})"
    status = f"✅ `{model_manager.current_model_name}`{api_info}" if model_manager.current_model_name != "Searching..." else "💀 Модель загружается..."
    
    commands_info = (
        "\n\n📋 **БЫСТРЫЕ КОМАНДЫ:**\n"
        "/medic - Общая медицина\n"
        "/gen - Гинекология\n"
        "/info - Полная инструкция\n"
        "/refresh - Очистить историю\n\n"
        "Или используй триггеры:\n"
        "!врач, !ген, !инфо, !обнови\n\n"
        "👉 /info - для полной инструкции"
    )
    
    await message.answer(
        f"🏥 **Медицинский Ассистент V4.0**\n{status}{commands_info}",
        reply_markup=get_mode_buttons()
    )

@dp.message(Command("info"))
async def command_info_handler(message: Message):
    """Полная инструкция."""
    await message.answer(INFO_TEXT, reply_markup=get_mode_buttons())

@dp.message(Command("medic"))
async def command_medic_handler(message: Message):
    """Включить режим общей медицины."""
    user_id = message.from_user.id
    user_state = get_user_state(user_id)
    user_state["mode"] = "medicine_general"
    
    await message.answer(
        "🏥 **Режим: Общая медицина** ✅\n\n"
        "Готов анализировать гайдлайны по кардиологии, инфекциям, пульмологии и др.\n\n"
        "_Триггеры: !док, !доктор, !врач, !медицина_\n\n"
        "📝 Задай свой вопрос! 👇"
    )

@dp.message(Command("gen"))
async def command_gen_handler(message: Message):
    """Включить режим гинекологии."""
    user_id = message.from_user.id
    user_state = get_user_state(user_id)
    user_state["mode"] = "medicine_gynecology"
    
    await message.answer(
        "👶 **Режим: Гинекология** ✅\n\n"
        "Готов анализировать клинические рекомендации ACOG, RCOG, ESHRE и Минздрава РФ.\n\n"
        "_Триггеры: !ген, !гениколог, !гинеколог, !гин_\n\n"
        "📝 Задай свой вопрос! 👇"
    )

@dp.message(Command("refresh"))
async def command_refresh_handler(message: Message):
    """Очистить память диалога."""
    user_id = message.from_user.id
    user_state = get_user_state(user_id)
    user_state["conversation_history"] = []
    
    await message.answer(
        "🗑️ **История диалога очищена**\n\n"
        "Бот больше не помнит предыдущих сообщений. Начинаем с чистого листа!\n\n"
        "_Триггеры: !очистить, !обнови, !очисти, !забудь_"
    )

# ═══════════════════════════════════════════════════════════════
# 🔥 ГЛАВНЫЙ ХЕНДЛЕР
# ═══════════════════════════════════════════════════════════════

@dp.message()
async def main_handler(message: Message):
    """Главный обработчик сообщений."""
    user_id = message.from_user.id
    user_state = get_user_state(user_id)
    
    text_to_check = message.text or message.caption or ""
    trigger_result = check_for_triggers(text_to_check)
    
    # ✅ ЕСЛИ НАЙДЕН ТРИГГЕР - ВЫПОЛНЯЕМ ДЕЙСТВИЕ
    if trigger_result:
        bot_user = await bot.get_me()
        await handle_trigger_action(message, trigger_result, bot_user)
        return
    
    # Если нет триггера - проверяем адресацию к боту
    if not model_manager.current_model:
        status_msg = await message.answer("⏳ Загрузка модели...")
        if not await model_manager.find_working_model():
            await status_msg.edit_text("❌ Не удалось загрузить модель. Проверьте API ключи.")
            return
        try:
            await status_msg.delete()
        except:
            pass
    
    bot_user = await bot.get_me()
    is_addressed = await is_addressed_to_bot(message, bot_user)
    
    if not is_addressed:
        return
    
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    
    try:
        text_content = ""
        if message.text:
            text_content = message.text.replace(f"@{bot_user.username}", "").strip()
        elif message.caption:
            text_content = message.caption.replace(f"@{bot_user.username}", "").strip()
        
        print(f"\n📨 Новый запрос от {user_id}: {text_content[:60]}...")
        
        prompt_parts, temp_files_to_delete = await prepare_prompt_parts(message, bot_user)
        
        if not prompt_parts:
            await message.reply("⚠️ Не найден текст или изображение")
            return
        
        await process_message(message, bot_user, text_content, prompt_parts, user_state)
    
    except Exception as e:
        logging.error(f"Main Handler Error: {e}")
        await message.reply(f"❌ Ошибка: {str(e)[:100]}")

# ═══════════════════════════════════════════════════════════════
# 🌐 WEB SERVER
# ═══════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {
        "status": "Alive",
        "bot_type": "Medical Assistant V4.0",
        "model": model_manager.current_model_name,
        "api_key": f"#{model_manager.api_key_index + 1}/{len(GOOGLE_KEYS)}",
        "active_users": len(USER_STATES),
    }

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": model_manager.current_model is not None,
        "model_name": model_manager.current_model_name,
    }

async def keep_alive_ping():
    """Пингует сервер для keep-alive."""
    if not RENDER_URL:
        return
    while True:
        await asyncio.sleep(300)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{RENDER_URL}/health") as resp:
                    pass
        except:
            pass

async def start_bot():
    """Запуск бота в polling режиме."""
    print(f"✅ API #{model_manager.api_key_index + 1} готов к использованию")
    
    print(f"🔍 Инициализирую модель...")
    if not await model_manager.find_working_model():
        print(f"⚠️ Не удалось загрузить модель, но продолжаю работу...")
    
    print(f"🤖 Запуск бота в polling режиме...")
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())

async def start_server():
    """Запуск FastAPI сервера."""
    config = uvicorn.Config(app, host="0.0.0.0", port=10000, log_level="error")
    server = uvicorn.Server(config)
    await server.serve()

async def main():
    """Главная точка входа."""
    print("=" * 50)
    print("🚀 ЗАПУСК МЕДИЦИНСКОГО АССИСТЕНТА V4.0")
    print("=" * 50)
    
    if not GOOGLE_KEYS:
        print("❌ ОШИБКА: Google API ключи не установлены!")
        sys.exit(1)
    
    print(f"✅ Найдено {len(GOOGLE_KEYS)} API ключей")
    
    try:
        genai.configure(api_key=GOOGLE_KEYS[model_manager.api_key_index])
        print(f"✅ API #{model_manager.api_key_index + 1} сконфигурирован")
    except Exception as e:
        print(f"❌ Ошибка конфигурации API: {e}")
        sys.exit(1)
    
    await asyncio.gather(
        start_server(),
        start_bot(),
        keep_alive_ping(),
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Завершение работы...")
