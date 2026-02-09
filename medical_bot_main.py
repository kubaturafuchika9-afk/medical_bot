import os
import asyncio
import logging
import sys
from io import BytesIO
from typing import Optional, List, Dict, Tuple
from datetime import datetime

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

# ═══════════════════════════════════════════════════════════════
# 📚 СИСТЕМНЫЕ ПРОМТЫ
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT_GENERAL_MEDICINE = """Ты — исследователь-аналитик и академический ассистент для общей медицины.

🚨 ОБЯЗАТЕЛЬНЫЙ ДИСКЛЕЙМЕР:
Этот анализ — ОБРАЗОВАТЕЛЬНЫЙ МАТЕРИАЛ для студентов и медработников.
НЕ является клиническим руководством для лечения пациентов.
Все клинические решения требуют консультации с лицензированным врачом.

ПРИНЦИП РАБОТЫ:
├─ Поиск достоверных источников (PubMed, Cochrane, официальные гайдлайны)
├─ Анализ эффективности методов через GRADE оценку
├─ Указание авторов, лет публикации и уровня доказательности
├─ Отказ от выдуманных данных и источников
└─ Честное указание пробелов в знаниях

📚 ОФИЦИАЛЬНЫЕ ИСТОЧНИКИ (ТОЛЬКО ЭЕТИ):
PubMed/PMC, Cochrane Library, Web of Science, Scopus (peer-review)
Гайдлайны: WHO, CDC, ESC (кардиология), ADA (эндокринология), 
GOLD (пульмология), EASL (гастроэнтерология), Минздрав РФ, NICE

🎯 СТРУКТУРА ОТВЕТА:
Создавай таблицы с полями:
| Исследование/Гайдлайн | Год | Авторы | Метод | Результат | Эффективность | GRADE | PMID/DOI |

🛡️ АНТИ-ГАЛЛЮЦИНАЦИОННЫЙ КОНТРОЛЬ:
- Если данных нет → "Данные отсутствуют в открытых источниках"
- НЕ выдумывай PMID, авторов, цифры эффективности
- ВСЕГДА указывай источник (PMID/DOI) для каждого утверждения
- При расхождениях между гайдлайнами → объясни почему
- Помечай данные старше 5 лет как "историческая справка"

⚠️ ЛОКАЛЬНАЯ РЕЛЕВАНТНОСТЬ:
Указывай, где подход РФ расходится с международными стандартами
Причины: доступность препаратов, регистрация, эпидемиология

📝 СТИЛЬ:
- Ясно, академично, без просторечий
- Факты впереди мнений
- Цифры: OR, RR, 95% CI, чувствительность/специфичность
- Таблицы вместо длинного текста"""

SYSTEM_PROMPT_GYNECOLOGY = """Ты — клинический ассистент-аналитик, помогающий студенту систематизировать данные по гинекологии и акушерству.

🚨 ОБЯЗАТЕЛЬНЫЙ ДИСКЛЕЙМЕР:
Этот анализ — ОБРАЗОВАТЕЛЬНЫЙ МАТЕРИАЛ для студентов и медработников.
НЕ является клиническим руководством для лечения пациентов.
Все клинические решения требуют консультации с лицензированным врачом.

ПРИНЦИП РАБОТЫ:
├─ Анализ клинических рекомендаций от ACOG, RCOG, ESHRE, Минздрава РФ
├─ Оценка эффективности методов лечения и диагностики
├─ Указание уровня доказательности (GRADE, Evidence Level)
├─ Честное указание авторов и их вклада
└─ Отказ от выдуманных источников и данных

📚 ПРИОРИТЕТНЫЕ ГАЙДЛАЙНЫ:
1. RCOG (Royal College) - консервативный, доказательный подход
2. ACOG (American College) - практико-ориентированный
3. ESHRE (European Society) - вспомогательные репродуктивные технологии
4. DGG/DGGG (Германия) - тщательные систематические обзоры
5. Минздрав РФ - федеральные клинические рекомендации
+ PubMed, Cochrane, Web of Science (только peer-review)

🎯 СТРУКТУРА ОТВЕТА:
Таблица с полями:
| Guideline/Исследование | Год | Авторы | Учреждение | Метод | Результат | Эффективность | GRADE | PMID/DOI |

🛡️ АНТИ-ГАЛЛЮЦИНАЦИОННЫЙ КОНТРОЛЬ:
- Если данных нет → явно отмечай "Данные отсутствуют"
- НЕ создавай вымышленные PMID или авторов
- ВСЕГДА ссылка на источник для каждого факта
- При расхождении RCOG vs ACOG → объясни разницу методологий
- Помечай рекомендации старше 5-7 лет как требующие проверки

⚠️ СПЕЦИФИКА ГИНЕКОЛОГИИ:
- RCOG часто консервативнее чем ACOG (принцип предосторожности)
- ESHRE специализируется на ВРТ (экстракорпоральное оплодотворение)
- Минздрав РФ может рекомендовать другие препараты (доступность/регистрация)
- ВСЕГДА отмечай: "В [стране] подход иной из-за [причина]"

📝 СТИЛЬ:
- Ясно, логично, академично
- Не копируй абстракты - переформулируй научно
- Цифры с доверительными интервалами (95% CI)
- Таблицы для наглядности"""

# ═══════════════════════════════════════════════════════════════
# 🔧 КОНФИГУРАЦИЯ ГЕНЕРАЦИИ
# ═══════════════════════════════════════════════════════════════

generation_config = {
    "temperature": 0.2,      # Низкая креативность - максимум фактичность
    "top_p": 0.8,
    "top_k": 40,
    "max_output_tokens": 4096,
}

# ═══════════════════════════════════════════════════════════════
# 📋 ИНИЦИАЛИЗАЦИЯ
# ═══════════════════════════════════════════════════════════════

bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN))
dp = Dispatcher()
app = FastAPI()

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ
ACTIVE_MODEL = None
ACTIVE_MODEL_NAME = "Searching..."
CURRENT_API_KEY_INDEX = 0
MODEL_LIMITS = {}
CURRENT_MODE = "medicine_general"  # По умолчанию - общая медицина

# ПАМЯТЬ ДИАЛОГОВ (user_id -> список сообщений)
USER_CONVERSATIONS = {}

# ═══════════════════════════════════════════════════════════════
# 🎯 РУССКИЕ ТРИГГЕРЫ (ТОЧНОЕ СОВПАДЕНИЕ)
# ═══════════════════════════════════════════════════════════════

TRIGGER_DOCTOR = "!врач"      # Включить режим общей медицины
TRIGGER_GYNECOLOGY = "!гениколог"  # Включить режим гинекологии
TRIGGER_REFRESH = "!обнови"   # Очистить память диалога

# ═══════════════════════════════════════════════════════════════
# 🔧 ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ═══════════════════════════════════════════════════════════════

def get_mode_buttons() -> InlineKeyboardMarkup:
    """Клавиатура для выбора режима."""
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="🏥 Общая медицина", callback_data="mode_general"),
            InlineKeyboardButton(text="🏥 Гинекология", callback_data="mode_gyn"),
        ]
    ])
    return keyboard

def get_dynamic_model_list():
    """Получает список доступных моделей Gemini."""
    available_models = []
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                name = m.name.replace("models/", "")
                if "gemini" in name:
                    available_models.append(name)
    except Exception as e:
        print(f"⚠️ Ошибка получения списка моделей: {e}")
    
    hardcoded = ["gemini-exp-1206", "gemini-1.5-flash", "gemini-1.5-flash-8b", 
                 "gemini-2.0-flash-exp", "gemini-3-flash-preview"]
    for h in hardcoded:
        if h not in available_models:
            available_models.append(h)
    
    return list(set(available_models))

def sort_models_priority(models):
    """Сортирует модели по приоритету."""
    def score(name):
        s = 0
        if "exp" in name: s += 500
        if "3-" in name or "2.5-" in name: s += 400
        if "flash" in name: s += 300
        if "1.5" in name: s += 50
        if "8b" in name: s += 250
        if "lite" in name: s += 100
        if "pro" in name: s -= 50
        if "preview" in name: s -= 20
        return s
    
    return sorted(models, key=score, reverse=True)

async def switch_api_key(silent: bool = True) -> bool:
    """Переключается на следующий API ключ."""
    global CURRENT_API_KEY_INDEX, ACTIVE_MODEL, ACTIVE_MODEL_NAME
    
    old_index = CURRENT_API_KEY_INDEX
    
    for i in range(len(GOOGLE_KEYS)):
        next_index = (CURRENT_API_KEY_INDEX + 1) % len(GOOGLE_KEYS)
        if next_index == old_index:
            return False
        
        CURRENT_API_KEY_INDEX = next_index
        try:
            genai.configure(api_key=GOOGLE_KEYS[CURRENT_API_KEY_INDEX])
            if await find_best_working_model(silent=silent):
                return True
        except Exception as e:
            pass
    
    return False

async def find_best_working_model(silent: bool = False) -> bool:
    """Находит рабочую модель на текущем API ключе."""
    global ACTIVE_MODEL, ACTIVE_MODEL_NAME, MODEL_LIMITS
    
    candidates = sort_models_priority(get_dynamic_model_list())
    
    if not silent:
        print(f"📋 Проверка моделей на API #{CURRENT_API_KEY_INDEX + 1}")
    
    for model_name in candidates:
        if MODEL_LIMITS.get(model_name, {}).get(CURRENT_API_KEY_INDEX, False):
            continue
        
        try:
            test_model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=generation_config,
                system_instruction="Ты помощник. Ответь 'ok'."
            )
            response = await test_model.generate_content_async("ping")
            
            if response and response.text:
                if not silent:
                    print(f"✅ Подключено: {model_name}")
                ACTIVE_MODEL = test_model
                ACTIVE_MODEL_NAME = model_name
                return True
        
        except Exception as e:
            err = str(e)
            if "429" in err:
                if model_name not in MODEL_LIMITS:
                    MODEL_LIMITS[model_name] = {}
                MODEL_LIMITS[model_name][CURRENT_API_KEY_INDEX] = True
    
    return False

async def is_addressed_to_bot(message: Message, bot_user: types.User):
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

def get_user_conversation_history(user_id: int) -> List[dict]:
    """Получает историю диалога пользователя."""
    return USER_CONVERSATIONS.get(user_id, [])

def add_to_conversation(user_id: int, role: str, content: str):
    """Добавляет сообщение в историю диалога."""
    if user_id not in USER_CONVERSATIONS:
        USER_CONVERSATIONS[user_id] = []
    
    USER_CONVERSATIONS[user_id].append({
        "role": role,
        "parts": [content]
    })
    
    # Ограничиваем историю 20 сообщениями (10 пар)
    if len(USER_CONVERSATIONS[user_id]) > 20:
        USER_CONVERSATIONS[user_id] = USER_CONVERSATIONS[user_id][-20:]

def clear_user_conversation(user_id: int):
    """Очищает историю диалога пользователя."""
    if user_id in USER_CONVERSATIONS:
        del USER_CONVERSATIONS[user_id]
        print(f"🗑️ История диалога очищена для пользователя {user_id}")

def check_for_triggers(text: str) -> Optional[str]:
    """
    Проверяет наличие русских триггеров (ТОЧНОЕ совпадение).
    Возвращает название триггера или None.
    """
    if not text:
        return None
    
    text_lower = text.strip().lower()
    
    # Проверяем точное совпадение целого слова
    words = text_lower.split()
    
    for word in words:
        if word == TRIGGER_DOCTOR:
            return "doctor"
        elif word == TRIGGER_GYNECOLOGY:
            return "gynecology"
        elif word == TRIGGER_REFRESH:
            return "refresh"
    
    return None

async def process_with_retry(message: Message, bot_user: types.User, text_content: str, 
                             prompt_parts: List, temp_files: List):
    """Обработка с retry логикой."""
    global ACTIVE_MODEL, ACTIVE_MODEL_NAME, CURRENT_MODE
    
    try:
        # Выбираем системный промт в зависимости от режима
        if CURRENT_MODE == "medicine_general":
            system_prompt = SYSTEM_PROMPT_GENERAL_MEDICINE
            mode_name = "🏥 Общая медицина"
        else:  # gynecology
            system_prompt = SYSTEM_PROMPT_GYNECOLOGY
            mode_name = "🏥 Гинекология"
        
        print(f"🚀 Запрос в {ACTIVE_MODEL_NAME} [{mode_name}]")
        
        # Получаем историю диалога
        conversation_history = get_user_conversation_history(message.from_user.id)
        
        # Добавляем текущий вопрос в историю
        if conversation_history:
            prompt_parts_with_history = conversation_history + [{"role": "user", "parts": prompt_parts}]
        else:
            prompt_parts_with_history = [{"role": "user", "parts": prompt_parts}]
        
        # Создаём модель с историей
        current_model = genai.GenerativeModel(
            model_name=ACTIVE_MODEL_NAME,
            generation_config=generation_config,
            system_instruction=system_prompt
        )
        
        # Если есть история - используем её
        if conversation_history:
            response = await current_model.generate_content_async(
                prompt_parts_with_history
            )
        else:
            response = await current_model.generate_content_async(prompt_parts)
        
        if response.text:
            print(f"📨 Ответ получен ({len(response.text)} символов)")
            
            # Добавляем ответ в историю
            add_to_conversation(message.from_user.id, "user", text_content)
            add_to_conversation(message.from_user.id, "model", response.text)
            
            # Обрезаем очень длинные ответы
            answer_text = response.text
            if len(answer_text) > 4000:
                answer_text = answer_text[:3900] + "\n\n⚠️ Ответ обрезан из-за длины."
            
            # Отправляем ответ
            await message.reply(answer_text, parse_mode=ParseMode.MARKDOWN)
            print(f"✅ Ответ отправлен")
            return True
        else:
            await message.reply("⚠️ Пустой ответ от модели")
            return False
    
    except Exception as e:
        logging.error(f"Gen Error: {e}")
        error_str = str(e)
        
        if "429" in error_str or "quota" in error_str or "404" in error_str:
            if ACTIVE_MODEL_NAME not in MODEL_LIMITS:
                MODEL_LIMITS[ACTIVE_MODEL_NAME] = {}
            MODEL_LIMITS[ACTIVE_MODEL_NAME][CURRENT_API_KEY_INDEX] = True
            
            print(f"⚠️ Лимит на модели → ищу новую")
            
            if await find_best_working_model(silent=True):
                print(f"✅ Новая модель найдена")
                return await process_with_retry(message, bot_user, text_content, prompt_parts, temp_files)
            
            if await switch_api_key(silent=True):
                print(f"✅ API ключ переключен")
                return await process_with_retry(message, bot_user, text_content, prompt_parts, temp_files)
            
            await message.reply("❌ Все лимиты исчерпаны. Попробуйте позже.")
            return False
        else:
            await message.reply(f"❌ Ошибка: {error_str[:100]}")
            return False
    
    finally:
        for f_path in temp_files:
            try:
                os.remove(f_path)
            except:
                pass

# ═══════════════════════════════════════════════════════════════
# 📝 CALLBACK ХЕНДЛЕРЫ
# ═══════════════════════════════════════════════════════════════

@dp.callback_query()
async def handle_mode_callback(query: CallbackQuery):
    """Обработка переключения режимов."""
    global CURRENT_MODE
    
    callback_data = query.data
    
    if callback_data == "mode_general":
        CURRENT_MODE = "medicine_general"
        message_text = (
            "🏥 **Режим: Общая медицина**\n\n"
            "**Специализация:** Кардиология, инфекции, пульмология, гастроэнтерология, эндокринология и др.\n\n"
            "**Принцип работы:**\n"
            "• Ищу официальные гайдлайны и исследования (PubMed, Cochrane, CDC, WHO и т.д.)\n"
            "• Оцениваю эффективность методов через GRADE систему\n"
            "• Указываю авторов, годы и уровень доказательности\n"
            "• Объясняю расхождения между странами (РФ vs Запад)\n"
            "• Честно говорю, где данных нет\n\n"
            "⚠️ Образовательный материал. Не заменяет врача."
        )
        
    elif callback_data == "mode_gyn":
        CURRENT_MODE = "medicine_gynecology"
        message_text = (
            "🏥 **Режим: Гинекология и акушерство**\n\n"
            "**Специализация:** Репродуктивная медицина, менструальные расстройства, ВРТ, беременность и т.д.\n\n"
            "**Принцип работы:**\n"
            "• Ищу рекомендации ACOG, RCOG, ESHRE, Минздрава РФ\n"
            "• Анализирую клинические исследования с высоким уровнем доказательности\n"
            "• Указываю авторов и их вклад в науку\n"
            "• Объясняю различия в подходах между странами\n"
            "• Отмечаю пробелы в знаниях\n\n"
            "⚠️ Образовательный материал. Не заменяет врача."
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
# 🎮 КОМАНДЫ
# ═══════════════════════════════════════════════════════════════

@dp.message(CommandStart())
async def command_start_handler(message: Message):
    """Стартовое сообщение."""
    
    api_info = f" (API #{CURRENT_API_KEY_INDEX + 1}/{len(GOOGLE_KEYS)})" if len(GOOGLE_KEYS) > 1 else ""
    status = f"✅ `{ACTIVE_MODEL_NAME}`{api_info}" if ACTIVE_MODEL else "💀 Модель не загружена"
    
    commands_info = (
        "\n\n📋 **Текущий режим:** 🏥 Общая медицина\n\n"
        "**Команды:**\n"
        "  /medic (триггер !врач) - Общая медицина\n"
        "  /gen (триггер !гениколог) - Гинекология\n"
        "  /refresh (триггер !обнови) - Очистить память диалога\n\n"
        "**Как использовать:**\n"
        "1. Выберите режим (команда или триггер)\n"
        "2. Напишите вопрос\n"
        "3. Бот запомнит контекст для следующих вопросов\n"
        "4. /refresh чтобы забыть всё и начать заново"
    )
    
    await message.answer(
        f"🏥 **Медицинский Ассистент V2.0**\n{status}{commands_info}",
        reply_markup=get_mode_buttons()
    )

@dp.message(Command("medic"))
async def command_medic_handler(message: Message):
    """Включить режим общей медицины."""
    global CURRENT_MODE
    CURRENT_MODE = "medicine_general"
    
    await message.answer(
        "🏥 **Режим: Общая медицина** ✅\n\n"
        "Готов анализировать гайдлайны по кардиологии, инфекциям, пульмологии и др.\n\n"
        "_Триггер: !врач_",
        reply_markup=get_mode_buttons()
    )

@dp.message(Command("gen"))
async def command_gen_handler(message: Message):
    """Включить режим гинекологии."""
    global CURRENT_MODE
    CURRENT_MODE = "medicine_gynecology"
    
    await message.answer(
        "🏥 **Режим: Гинекология** ✅\n\n"
        "Готов анализировать клинические рекомендации ACOG, RCOG, ESHRE и Минздрава РФ.\n\n"
        "_Триггер: !гениколог_",
        reply_markup=get_mode_buttons()
    )

@dp.message(Command("refresh"))
async def command_refresh_handler(message: Message):
    """Очистить память диалога."""
    user_id = message.from_user.id
    clear_user_conversation(user_id)
    
    await message.answer(
        "🗑️ **История диалога очищена**\n\n"
        "Бот больше не помнит предыдущих сообщений. Начинаем с чистого листа!\n\n"
        "_Триггер: !обнови_"
    )

# ═══════════════════════════════════════════════════════════════
# 🔥 ГЛАВНЫЙ ХЕНДЛЕР
# ═══════════════════════════════════════════════════════════════

@dp.message()
async def main_handler(message: Message):
    """Главный обработчик сообщений."""
    global ACTIVE_MODEL, ACTIVE_MODEL_NAME, CURRENT_MODE
    
    # 🔍 ПРОВЕРЯЕМ ТРИГГЕРЫ
    text_to_check = message.text or message.caption or ""
    trigger_result = check_for_triggers(text_to_check)
    
    if trigger_result == "doctor":
        CURRENT_MODE = "medicine_general"
        await command_medic_handler(message)
        return
    elif trigger_result == "gynecology":
        CURRENT_MODE = "medicine_gynecology"
        await command_gen_handler(message)
        return
    elif trigger_result == "refresh":
        await command_refresh_handler(message)
        return
    
    # ЗАГРУЖАЕМ МОДЕЛЬ, ЕСЛИ НЕ ЗАГРУЖЕНА
    if not ACTIVE_MODEL:
        status_msg = await message.answer("⏳ Загрузка модели...")
        if not await find_best_working_model(silent=True):
            if not await switch_api_key(silent=True):
                await status_msg.edit_text("❌ Не удалось загрузить модель")
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
        
        print(f"\n📨 Новый запрос: {text_content[:60]}...")
        
        prompt_parts, temp_files_to_delete = await prepare_prompt_parts(message, bot_user)
        
        if not prompt_parts:
            await message.reply("⚠️ Не найден текст или изображение")
            return
        
        await process_with_retry(message, bot_user, text_content, prompt_parts, temp_files_to_delete)
    
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
        "bot_type": "Medical Assistant",
        "model": ACTIVE_MODEL_NAME,
        "mode": "general_medicine" if CURRENT_MODE == "medicine_general" else "gynecology",
        "api_keys_available": len(GOOGLE_KEYS),
    }

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": ACTIVE_MODEL is not None}

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
    """Запуск бота."""
    global CURRENT_API_KEY_INDEX
    
    for i, key in enumerate(GOOGLE_KEYS):
        try:
            genai.configure(api_key=key)
            CURRENT_API_KEY_INDEX = i
            print(f"✅ API #{i + 1} сконфигурирован")
            break
        except:
            pass
    
    print(f"🔍 Ищу рабочую модель...")
    await find_best_working_model()
    
    print(f"🤖 Запуск бота...")
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

async def start_server():
    """Запуск FastAPI сервера."""
    config = uvicorn.Config(app, host="0.0.0.0", port=10000, log_level="error")
    server = uvicorn.Server(config)
    await server.serve()

async def main():
    """Главная точка входа."""
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
