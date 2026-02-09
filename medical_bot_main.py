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

# ПРИОРИТЕТ МОДЕЛЕЙ (от САМОЙ ТОЧНОЙ для медицины к худшей)
# Критерий: ТОЧНОСТЬ > СКОРОСТЬ, потому что медицина критична
MODEL_PRIORITY = [
    "gemini-3-flash",              # 1️⃣ САМАЯ УМНАЯ - новейшая, максимум точности и понимания
    "gemini-2.5-flash",            # 2️⃣ ОЧЕНЬ ТОЧНАЯ - мощная, детальная, надёжная
    "gemini-2.5-flash-lite",       # 3️⃣ ХОРОШАЯ - точная и экономная
    "gemini-1.5-flash",            # 4️⃣ РЕЗЕРВНАЯ - старая версия, но работает
]

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

📝 СТИЛЬ:
- Ясно, академично, без просторечий
- Факты впереди мнений
- Цифры: OR, RR, 95% CI
- МАКСИМУМ 3000 символов!"""

SYSTEM_PROMPT_GYNECOLOGY = """Ты — клинический ассистент-аналитик по гинекологии (репродуктивная медицина, менструальные расстройства, ВРТ).

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

📝 СТИЛЬ:
- Ясно, логично, академично
- Не копируй абстракты - переформулируй
- Цифры с доверительными интервалами (95% CI)
- МАКСИМУМ 3000 символов!"""

SYSTEM_PROMPT_OBSTETRICS = """Ты — профессиональный акушерский ассистент-аналитик. Специализация: Акушерство и перинатология.

⚠️ ДИСКЛЕЙМЕР:
Все клинические решения требуют консультации с лицензированным врачом. При беременности, кровотечениях или угрозе для жизни → немедленно к врачу!

ПРИНЦИП РАБОТЫ:
├─ Анализ официальных акушерских гайдлайнов (RCOG, ACOG, WHO, NICE, ISUOG)
├─ Оценка рисков материнской и перинатальной смертности
├─ Указание уровня доказательности (GRADE, Evidence Level, Class)
├─ Верификация всех PMID и цифр из первоисточников
├─ Честное указание пробелов в знаниях и противоречий между гайдлайнами
└─ Отказ от выдуманных данных, сроков, препаратов, эффектов

📚 ПРИОРИТЕТНЫЕ ИСТОЧНИКИ:

🌍 МЕЖДУНАРОДНЫЕ:
├─ RCOG (Royal College) - Green-top Guidelines
├─ ACOG (American College) - Practice Bulletins
├─ WHO Guidelines on maternal and perinatal health
├─ NICE (National Institute for Health and Care Excellence)
├─ ISUOG (International Society of Ultrasound in Obstetrics)
└─ ESHRE (European Society of Human Reproduction)

🇷🇺 РОССИЙСКИЕ СТАНДАРТЫ:
├─ Приказ Минздрава РФ по акушерству
├─ Федеральные клинические рекомендации
└─ Российское общество акушеров-гинекологов (РОАГ)

📑 НАУЧНЫЕ БАЗЫ (ТОЛЬКО peer-review):
├─ PubMed/PMC (PMID ОБЯЗАТЕЛЕН)
├─ Cochrane Library (систематические обзоры)
├─ Web of Science и Scopus
└─ ❌ НЕ используй: Википедия, блоги, соцсети

🎯 СТРУКТУРА ОТВЕТА:

📌 **Гайдлайн / Исследование:** (название)
   Организация: RCOG/ACOG/WHO/NICE/ISUOG
   Год: XXXX
   Рекомендация: (суть)
   Уровень доказательности: High/Moderate/Low
   PMID: XXXXXXXX

ДЛЯ СКРИНИНГОВ:
├─ Чувствительность (%) | Специфичность (%)
├─ Ложноположительный результат (%)

ДЛЯ ОСЛОЖНЕНИЙ:
├─ Частота осложнения (%)
├─ Материнская смертность / заболеваемость
├─ Перинатальная смертность (на 1000)
├─ Рекомендуемая тактика и сроки доставки

🛡️ КРИТИЧЕСКИЙ КОНТРОЛЬ:

⚠️ НИКОГДА (опасные ошибки):
├─ Придумать PMID
├─ Сказать "безопасно в беременности" без источника
├─ Озвучить цифру смертности без PMID
├─ Гарантировать исходы
├─ Путать категории безопасности препаратов (FDA A/B/C/D/X)
├─ Звучать как медицинский совет вместо образования
└─ Рекомендовать сроки доставки без гайдлайна

✓ ПРОВЕРЬ КАЖДОЕ УТВЕРЖДЕНИЕ:
1. PMID реален? (8-9 цифр, в PubMed?)
2. Гайдлайн актуален? (когда последнее обновление?)
3. Цифры откуда? (из статьи или я округлил?)
4. Это официальная рекомендация или мой вывод?

ЕСЛИ ДАННЫХ НЕ УВЕРЕН:
├─ "На момент обучения этого в PubMed не было"
├─ "Гайдлайны расходятся: ACOG рекомендует X, RCOG — Y"
├─ "Исследование в pre-print (не опубликовано)"
├─ "Это зона научной неопределённости"
└─ "Требуется консультация специалиста"

📝 СТИЛЬ:
- Ясно, логично, академично
- Цифры ВПЕРЕДИ: "В 70% случаев..." а не "Часто..."
- Уверенность: High (GRADE) vs Moderate vs Low
- Сроки в неделях + дни (38+6 недель, не "38 с половиной")
- МАКСИМУМ 3000 символов!"""

# ═══════════════════════════════════════════════════════════════
# 🤖 СИСТЕМА УПРАВЛЕНИЯ МОДЕЛЯМИ (С ПРИОРИТЕТОМ НА ТОЧНОСТЬ)
# ═══════════════════════════════════════════════════════════════

class ModelManager:
    """Управляет доступными моделями с приоритетом на ТОЧНОСТЬ."""
    
    def __init__(self):
        self.api_key_index = 0
        self.current_model = None
        self.current_model_name = "Searching..."
        # Отслеживаем лимиты: {model_name: {api_index: is_limited}}
        self.model_limits = {}
    
    async def find_working_model(self):
        """
        Ищет рабочую модель по приоритету ТОЧНОСТИ.
        Сначала пробует самую точную модель на всех API,
        потом вторую по точности, потом третью и т.д.
        """
        
        # Пробуем каждую модель в порядке приоритета точности
        for model_name in MODEL_PRIORITY:
            print(f"\n🔍 Проверяю модель: {model_name}")
            
            # Пробуем текущий API ключ
            if await self._try_model(model_name, self.api_key_index):
                return True
            
            # Если текущий API в лимите, пробуем другие API ключи
            for api_idx in range(len(GOOGLE_KEYS)):
                if api_idx == self.api_key_index:
                    continue  # Уже пробовали
                
                # Переключаемся на другой API
                self.api_key_index = api_idx
                try:
                    genai.configure(api_key=GOOGLE_KEYS[self.api_key_index])
                    print(f"🔄 Переключился на API #{self.api_key_index + 1}")
                    
                    if await self._try_model(model_name, self.api_key_index):
                        return True
                except:
                    pass
        
        print("❌ Не удалось найти рабочую модель на всех API и моделях")
        return False
    
    async def _try_model(self, model_name: str, api_index: int) -> bool:
        """Пробует одну модель на одном API ключе."""
        
        # Проверяем, не в ли лимите эта модель на этом API
        if self.model_limits.get(model_name, {}).get(api_index, False):
            print(f"⏭️ Модель {model_name} уже в лимите на API #{api_index + 1}")
            return False
        
        try:
            test_model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=generation_config,
                system_instruction=SYSTEM_PROMPT_GENERAL_MEDICINE
            )
            
            # Быстрый тест
            response = await test_model.generate_content_async("test")
            
            if response and response.text:
                self.current_model = test_model
                self.current_model_name = model_name
                self.api_key_index = api_index
                print(f"✅ Подключена модель: {model_name} на API #{api_index + 1}")
                return True
        
        except Exception as e:
            error_str = str(e)
            
            # Если это лимит - отмечаем и переходим дальше
            if "429" in error_str or "quota" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                if model_name not in self.model_limits:
                    self.model_limits[model_name] = {}
                self.model_limits[model_name][api_index] = True
                print(f"⚠️ Лимит на {model_name} (API #{api_index + 1})")
            else:
                print(f"❌ Ошибка {model_name}: {error_str[:50]}")
        
        return False
    
    async def handle_limit_error(self):
        """Обрабатывает ошибку лимита - ищет альтернативу."""
        print(f"\n⚠️ Текущая модель {self.current_model_name} (API #{self.api_key_index + 1}) в лимите!")
        
        # Отмечаем текущую комбинацию как ограниченную
        if self.current_model_name not in self.model_limits:
            self.model_limits[self.current_model_name] = {}
        self.model_limits[self.current_model_name][self.api_key_index] = True
        
        # Ищем альтернативу
        if await self.find_working_model():
            print(f"✅ Нашёл альтернативу: {self.current_model_name} (API #{self.api_key_index + 1})")
            return True
        
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
    
    # Акушерство
    "!aku": "obstetrics",
    "!акушер": "obstetrics",
    "!беременность": "obstetrics",
    "!роды": "obstetrics",
    
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
    """Проверяет ТОЧНОЕ совпадение триггеров."""
    if not text:
        return None
    
    text_lower = text.strip().lower()
    words = text_lower.split()
    
    for word in words:
        if word in TRIGGER_WORDS_MAPPING:
            action = TRIGGER_WORDS_MAPPING[word]
            print(f"🔴 ТОЧНЫЙ ТРИГГЕР ОБНАРУЖЕН: '{word}' → Действие: {action}")
            return action
    
    return None

# ═══════════════════════════════════════════════════════════════
# 🔧 ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ═══════════════════════════════════════════════════════════════

def get_mode_buttons() -> InlineKeyboardMarkup:
    """Клавиатура для выбора режима - 3 отдельные кнопки."""
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🏥 Общая медицина", callback_data="mode_general")],
        [InlineKeyboardButton(text="👶 Гинекология", callback_data="mode_gyn")],
        [InlineKeyboardButton(text="🤰 Акушерство", callback_data="mode_aku")],
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
    
    for i, part in enumerate(parts):
        if part:
            if i < len(parts) - 1:
                await message.reply(part + "\n\n_[часть " + str(i+1) + "/" + str(len(parts)) + "]_", parse_mode=ParseMode.MARKDOWN)
            else:
                await message.reply(part, parse_mode=ParseMode.MARKDOWN)
            
            await asyncio.sleep(0.5)

async def process_message(message: Message, bot_user: types.User, text_content: str, 
                          prompt_parts: List, user_state: Dict):
    """Обработка сообщения."""
    try:
        if user_state["mode"] == "medicine_general":
            system_prompt = SYSTEM_PROMPT_GENERAL_MEDICINE
            mode_name = "🏥 Общая медицина"
        elif user_state["mode"] == "medicine_gynecology":
            system_prompt = SYSTEM_PROMPT_GYNECOLOGY
            mode_name = "👶 Гинекология"
        else:  # obstetrics
            system_prompt = SYSTEM_PROMPT_OBSTETRICS
            mode_name = "🤰 Акушерство"
        
        print(f"\n📨 Запрос от {message.from_user.id} [{mode_name}]")
        print(f"   Модель: {model_manager.current_model_name} (API #{model_manager.api_key_index + 1})")
        
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
            print(f"⚠️ Лимит текущей модели!")
            
            # Ищем альтернативу
            if await model_manager.handle_limit_error():
                print(f"✅ Пробую снова с {model_manager.current_model_name}")
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
            "🏥 *Общая медицина* ✅\n\n"
            "Готов анализировать кардиологию, инфекции, пульмологию и др.\n\n"
            "📝 Задай вопрос 👇"
        )
        print(f"✅ {message.from_user.first_name} выбрал ОБЩУЮ МЕДИЦИНУ")
    
    elif action == "gynecology":
        user_state["mode"] = "medicine_gynecology"
        await message.answer(
            "👶 *Гинекология* ✅\n\n"
            "Готов анализировать репродуктивную медицину и ВРТ.\n\n"
            "📝 Задай вопрос 👇"
        )
        print(f"✅ {message.from_user.first_name} выбрал ГИНЕКОЛОГИЮ")
    
    elif action == "obstetrics":
        user_state["mode"] = "medicine_obstetrics"
        await message.answer(
            "🤰 *Акушерство* ✅\n\n"
            "Готов анализировать беременность, роды и послеродовой период.\n\n"
            "📝 Задай вопрос 👇"
        )
        print(f"✅ {message.from_user.first_name} выбрал АКУШЕРСТВО")
    
    elif action == "info":
        print(f"ℹ️ {message.from_user.first_name} запросил информацию")
        await command_info_handler(message)
    
    elif action == "start":
        print(f"🔄 {message.from_user.first_name} запросил /start")
        await command_start_handler(message)
    
    elif action == "refresh":
        user_state["conversation_history"] = []
        await message.answer(
            "🗑️ *История очищена* ✅\n\n"
            "Начинаем диалог с чистого листа!"
        )
        print(f"✅ {message.from_user.first_name} очистил историю")

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
            "🏥 *Общая медицина*\n\n"
            "Кардиология, инфекции, пульмология, гастроэнтерология, эндокринология и др.\n\n"
            "*Источники:*\n"
            "PubMed, Cochrane, WHO, CDC, ESC, ADA, GOLD, Минздрав РФ, NICE\n\n"
            "⚠️ Все клинические решения требуют консультации с врачом."
        )
        
    elif callback_data == "mode_gyn":
        user_state["mode"] = "medicine_gynecology"
        message_text = (
            "👶 *Гинекология*\n\n"
            "Репродуктивная медицина, менструальные расстройства, ВРТ, беременность\n\n"
            "*Источники:*\n"
            "ACOG, RCOG, ESHRE, Минздрав РФ\n\n"
            "⚠️ Все клинические решения требуют консультации с врачом."
        )
    
    elif callback_data == "mode_aku":
        user_state["mode"] = "medicine_obstetrics"
        message_text = (
            "🤰 *Акушерство*\n\n"
            "Беременность, роды, послеродовой период, перинатальная помощь\n\n"
            "*Источники:*\n"
            "RCOG, ACOG, WHO, NICE, ISUOG, Минздрав РФ\n\n"
            "⚠️ Все клинические решения требуют консультации с врачом.\n"
            "При экстренности → немедленно к врачу!"
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

INFO_TEXT = """🏥 *МЕДИЦИНСКИЙ АССИСТЕНТ V5.0*

*ЧТО УМЕЕТ:*
🏥 Общая медицина
👶 Гинекология
🤰 Акушерство
📸 Анализирует картинки

*КОМАНДЫ:*
/start - Главное меню
/medic - Общая медицина
/gen - Гинекология
/aku - Акушерство
/info - Инструкция
/refresh - Очистить память

*ТРИГГЕР-СЛОВА:*
!врач, !док - общая медицина
!ген, !гениколог - гинекология
!беременность, !роды - акушерство
!инфо, !помощь - информация
!старт - главное меню
!обнови - очистить память

*КАК ИСПОЛЬЗОВАТЬ:*
1. Выбери /medic, /gen или /aku
2. Напиши вопрос
3. Бот даст ответ с источниками (PMID)
4. Можешь задавать уточнения

*ПРИМЕРЫ ВОПРОСОВ:*
🏥 "Гайдлайны по лечению гипертензии"
👶 "Рекомендации ACOG по СПКЯ"
🤰 "Скрининг синдрома Дауна"

⚠️ *ВАЖНО:*
Все клинические решения требуют консультации с врачом!
При беременности/кровотечении → немедленно к врачу!"""

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
        "\n\n*БЫСТРЫЕ КОМАНДЫ:*\n"
        "/medic - Общая медицина\n"
        "/gen - Гинекология\n"
        "/aku - Акушерство\n"
        "/info - Инструкция\n\n"
        "Или триггеры: !врач, !ген, !беременность\n"
        "/info для подробной справки"
    )
    
    await message.answer(
        f"🏥 *Медицинский Ассистент V5.0*\n{status}{commands_info}",
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
        "🏥 *Общая медицина* ✅\n\n"
        "Готов анализировать кардиологию, инфекции, пульмологию и др.\n\n"
        "📝 Задай вопрос 👇"
    )

@dp.message(Command("gen"))
async def command_gen_handler(message: Message):
    """Включить режим гинекологии."""
    user_id = message.from_user.id
    user_state = get_user_state(user_id)
    user_state["mode"] = "medicine_gynecology"
    
    await message.answer(
        "👶 *Гинекология* ✅\n\n"
        "Готов анализировать репродуктивную медицину и ВРТ.\n\n"
        "📝 Задай вопрос 👇"
    )

@dp.message(Command("aku"))
async def command_aku_handler(message: Message):
    """Включить режим акушерства."""
    user_id = message.from_user.id
    user_state = get_user_state(user_id)
    user_state["mode"] = "medicine_obstetrics"
    
    await message.answer(
        "🤰 *Акушерство* ✅\n\n"
        "Готов анализировать беременность, роды и послеродовой период.\n\n"
        "📝 Задай вопрос 👇"
    )

@dp.message(Command("refresh"))
async def command_refresh_handler(message: Message):
    """Очистить память диалога."""
    user_id = message.from_user.id
    user_state = get_user_state(user_id)
    user_state["conversation_history"] = []
    
    await message.answer("🗑️ *История очищена*\n\nНачинаем с чистого листа!")

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
    
    if trigger_result:
        bot_user = await bot.get_me()
        await handle_trigger_action(message, trigger_result, bot_user)
        return
    
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
        "bot_type": "Medical Assistant V5.0",
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
    print(f"\n{'='*60}")
    print(f"🚀 ЗАПУСК МЕДИЦИНСКОГО АССИСТЕНТА V5.0")
    print(f"{'='*60}")
    print(f"\n📋 ПРИОРИТЕТ МОДЕЛЕЙ (по ТОЧНОСТИ):")
    for i, model in enumerate(MODEL_PRIORITY, 1):
        print(f"  {i}️⃣ {model}")
    print(f"\n🔑 Доступно API ключей: {len(GOOGLE_KEYS)}")
    
    print(f"\n🔍 Инициализирую модель...")
    if not await model_manager.find_working_model():
        print(f"⚠️ Не удалось загрузить модель, но продолжаю работу...")
    
    print(f"✅ Модель: {model_manager.current_model_name} (API #{model_manager.api_key_index + 1})")
    print(f"🤖 Запуск бота в polling режиме...\n")
    
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())

async def start_server():
    """Запуск FastAPI сервера."""
    config = uvicorn.Config(app, host="0.0.0.0", port=10000, log_level="error")
    server = uvicorn.Server(config)
    await server.serve()

async def main():
    """Главная точка входа."""
    if not GOOGLE_KEYS:
        print("❌ ОШИБКА: Google API ключи не установлены!")
        sys.exit(1)
    
    # Инициализируем первый API ключ
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
