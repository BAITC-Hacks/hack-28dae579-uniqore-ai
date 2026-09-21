"""Классификация обращений и черновики ответов.

Два пути получения результата:

1. LLM — один вызов на обращение к OpenAI-совместимому эндпоинту. По умолчанию
   OpenRouter; тот же код работает с OpenAI и NVIDIA NIM — меняется только
   LLM_BASE_URL и LLM_MODEL. Используется, только если задан ключ.
2. Правила — детерминированный лексический классификатор плюс шаблон ответа.
   Работает всегда: без ключа, без сети, без зависимостей.

Правила намеренно общие: они опираются на лексические признаки запроса информации,
проблемы и запроса действия, а не на текст конкретных тестовых обращений.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field

CATEGORY_HELP = "справка"
CATEGORY_COMPLAINT = "жалоба"
CATEGORY_OTHER = "другое"
CATEGORIES = (CATEGORY_HELP, CATEGORY_COMPLAINT, CATEGORY_OTHER)

# При равном счёте выигрывает категория, стоящая раньше в этом списке:
# пропустить жалобу дороже, чем ошибиться в сторону справки.
TIE_BREAK_ORDER = (CATEGORY_COMPLAINT, CATEGORY_HELP, CATEGORY_OTHER)

MAX_MESSAGE_LEN = 2000
MAX_REPLY_LEN = 1200

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_TIMEOUT = 20.0

# OpenRouter просит указывать источник запроса. Заголовки необязательные,
# другим провайдерам не отправляются.
APP_URL = "https://github.com/BAITC-Hacks/hack-28dae579-uniqore-ai"
APP_TITLE = "HackAlem triage classifier"

# Лексиконы признаков: (регулярное выражение, вес).
_RAW_SIGNALS: dict[str, list[tuple[str, float]]] = {
    CATEGORY_COMPLAINT: [
        (r"жалоб", 3.0),
        (r"не\s+работа(ет|ют)", 2.5),
        (r"\bпропал[аои]?\b", 2.5),
        (r"слома(л|н|лся|ось)", 2.5),
        (r"хамств|груб(ый|ая|о|ость)", 2.5),
        (r"ужасн|отвратительн|безобрази|возмущ|недовол", 2.5),
        (r"\bочеред", 2.0),
        (r"грязн|антисанитар", 2.0),
        (r"долго\s+жду|жду\s+уже|до\s+сих\s+пор\s+не", 2.0),
        (r"холодн|остыв", 1.5),
        (r"отключ(или|ен|ена)|отсутствует|недоступен", 1.5),
        (r"задерж(ка|ивает)|опоздан", 1.5),
        (r"плох(о|ой|ая|ие)|некачествен", 1.5),
        (r"\bне\s+могу\b", 1.0),
        (r"опять|снова\s+не", 1.0),
        (r"!", 0.7),
    ],
    CATEGORY_HELP: [
        (r"справк", 3.0),
        (r"как\s+(получить|оформить|подать|сделать|попасть|доехать)", 2.5),
        (r"как(ие|ой|ая|ого)\s+(документ|бумаг|справк)", 2.5),
        (r"нужн[аоы]?\s+(ли\s+)?(справк|документ)", 2.0),
        (r"подскажите|уточните|хочу\s+узнать|хотел[а]?\s+бы\s+узнать", 1.5),
        (r"режим\s+работы|график\s+работы|расписани", 1.5),
        (r"\bгде\b", 1.5),
        (r"\bкогда\b", 1.5),
        (r"\bсколько\b", 1.5),
        (r"\bкак\b", 0.8),
    ],
    CATEGORY_OTHER: [
        (r"запис(аться|ать|ь\s+на)|запишите", 2.5),
        (r"заброниров|\bбронь\b", 2.0),
        (r"прошу\s+(вас\s+)?(назначить|перенести|организовать)", 2.0),
        (r"отменить\s+запись|перенести\s+(встречу|консультацию)", 2.0),
        (r"предлага(ю|ем)|предложени|\bидея\b", 1.5),
        (r"спасибо|благодар", 1.5),
    ],
}

SIGNALS: dict[str, list[tuple[re.Pattern[str], str, float]]] = {
    category: [(re.compile(pattern, re.IGNORECASE), pattern, weight) for pattern, weight in items]
    for category, items in _RAW_SIGNALS.items()
}

# Вопросительное слово вместе со знаком вопроса — признак запроса информации.
_INTERROGATIVE = re.compile(
    r"\b(как|где|когда|сколько|как(ой|ая|ие|ого)|что|кто|куда|можно\s+ли)\b",
    re.IGNORECASE,
)
_HAS_CYRILLIC = re.compile(r"[а-яё]", re.IGNORECASE)
_CODE_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)

_REPLY_TEMPLATES = {
    CATEGORY_HELP: (
        "Здравствуйте! По вашему вопросу «{topic}» отвечает справочная служба университета. "
        "Напишите, пожалуйста, ФИО и группу — подготовим ответ или документ в течение "
        "2 рабочих дней. Если вопрос срочный, отметьте это в ответном сообщении."
    ),
    CATEGORY_COMPLAINT: (
        "Здравствуйте! Спасибо, что сообщили о проблеме: «{topic}». "
        "Передали обращение в ответственную службу и вернёмся с ответом в течение "
        "1 рабочего дня. Если ситуация повторится, пришлите дату, время и корпус — "
        "это ускорит разбор."
    ),
    CATEGORY_OTHER: (
        "Здравствуйте! Приняли ваше обращение «{topic}». "
        "Чтобы направить его в нужный отдел, уточните, пожалуйста, желаемую дату, время "
        "и удобный контакт для связи. После этого вернёмся с подтверждением."
    ),
}

SYSTEM_PROMPT = (
    "Ты — оператор службы поддержки университета. "
    "Определи ровно одну категорию обращения: справка, жалоба или другое. "
    "«справка» — запрос информации или документа; «жалоба» — сообщение о проблеме "
    "или недовольство качеством; «другое» — всё остальное, включая запись на приём, "
    "предложения и благодарности. "
    "Затем напиши черновик ответа на русском языке: приветствие и 2–3 предложения. "
    "Главное правило: ты ничего не знаешь про этот университет. "
    "Запрещено называть расположение объектов, адреса, корпуса, этажи и кабинеты, "
    "номера телефонов, ссылки, адреса почты, имена сотрудников, цены и сроки, "
    "если они не названы прямо в самом обращении. "
    "Вместо факта, которого нет во входных данных, попроси уточнение или сообщи, "
    "что обращение передано в профильную службу. "
    "Лучше короткий ответ без подробностей, чем правдоподобная выдумка. "
    'Ответь строго JSON-объектом вида {"category": "...", "reason": "...", "reply": "..."}.'
)

# Механическая проверка черновика. Выдуманное расположение регулярным выражением
# не поймать, а вот контакты и ссылки — вполне: это самые дорогие выдумки,
# потому что выглядят как проверяемый факт.
_FORBIDDEN_IN_REPLY = (
    ("ссылка", re.compile(r"https?://|\bwww\.", re.IGNORECASE)),
    ("адрес почты", re.compile(r"[\w.+-]+@[\w-]+\.[a-zа-я]{2,}", re.IGNORECASE)),
    ("номер телефона", re.compile(r"(?:\+7|\b8)[\s(\-]*\d{3}[\s)\-]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}\b")),
    ("длинная последовательность цифр", re.compile(r"\d[\d\s\-()]{8,}\d")),
)


def check_reply(reply: str) -> str | None:
    """Проверить черновик на выдуманные контакты. Возвращает причину отказа или None."""
    for label, regex in _FORBIDDEN_IN_REPLY:
        if regex.search(reply):
            return f"в черновике {label}"
    return None


@dataclass
class Result:
    """Итог разбора одного обращения."""

    text: str
    category: str
    reply: str
    source: str  # "llm" или "rules"
    confidence: float | None = None  # только для правил: отрыв лидера, 0..1
    signals: list[str] = field(default_factory=list)
    note: str | None = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "category": self.category,
            "reply": self.reply,
            "source": self.source,
            "confidence": self.confidence,
            "signals": self.signals,
            "note": self.note,
        }


@dataclass(frozen=True)
class LLMConfig:
    """Настройки LLM. Ключ приходит только из окружения и нигде не печатается."""

    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout: float = DEFAULT_TIMEOUT

    @staticmethod
    def from_env(env: dict | None = None) -> "LLMConfig | None":
        """Собрать конфигурацию из окружения. Нет ключа — нет LLM, это штатный режим."""
        env = os.environ if env is None else env
        api_key = (
            env.get("LLM_API_KEY")
            or env.get("OPENROUTER_API_KEY")
            or env.get("OPENAI_API_KEY")
            or ""
        ).strip()
        if not api_key:
            return None
        base_url = (env.get("LLM_BASE_URL") or DEFAULT_BASE_URL).strip().rstrip("/")
        model = (env.get("LLM_MODEL") or DEFAULT_MODEL).strip()
        try:
            timeout = float(env.get("LLM_TIMEOUT") or DEFAULT_TIMEOUT)
        except ValueError:
            timeout = DEFAULT_TIMEOUT
        if timeout <= 0:
            timeout = DEFAULT_TIMEOUT
        return LLMConfig(api_key=api_key, base_url=base_url, model=model, timeout=timeout)


def classify_by_rules(text: str) -> tuple[str, float, list[str]]:
    """Взвешенный лексический скоринг по трём категориям.

    Возвращает категорию, уверенность (относительный отрыв лидера) и сработавшие признаки.
    """
    scores = {category: 0.0 for category in CATEGORIES}
    matched: list[str] = []

    for category, items in SIGNALS.items():
        for regex, pattern, weight in items:
            if regex.search(text):
                scores[category] += weight
                matched.append(f"{category}: {pattern}")

    if "?" in text and _INTERROGATIVE.search(text):
        scores[CATEGORY_HELP] += 1.0
        matched.append(f"{CATEGORY_HELP}: вопросительное предложение")

    ranked = sorted(CATEGORIES, key=lambda c: (-scores[c], TIE_BREAK_ORDER.index(c)))
    top = ranked[0]
    if scores[top] <= 0:
        return CATEGORY_OTHER, 0.0, matched

    runner_up = scores[ranked[1]]
    confidence = round(min(1.0, (scores[top] - runner_up) / scores[top]), 2)
    return top, confidence, matched


def extract_topic(text: str, limit: int = 80) -> str:
    """Короткая тема обращения для подстановки в шаблон ответа."""
    topic = re.sub(r"\s+", " ", text).strip().strip(" .!?…")
    if not topic:
        return "без текста"
    if len(topic) > limit:
        topic = topic[:limit].rstrip() + "…"
    return topic


def build_reply(text: str, category: str) -> str:
    """Черновик ответа из шаблона категории с подстановкой темы обращения."""
    template = _REPLY_TEMPLATES.get(category, _REPLY_TEMPLATES[CATEGORY_OTHER])
    return template.format(topic=extract_topic(text))


def _extract_json_object(content: str) -> dict | None:
    """Достать JSON-объект из ответа модели.

    Часть моделей OpenRouter оборачивает JSON в ```-блок или добавляет текст вокруг,
    поэтому сначала пробуем разобрать как есть, затем снимаем обёртку.
    """
    content = content.strip()
    fence = _CODE_FENCE.match(content)
    if fence:
        content = fence.group(1).strip()
    try:
        parsed = json.loads(content)
    except ValueError:
        start, end = content.find("{"), content.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            parsed = json.loads(content[start : end + 1])
        except ValueError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _parse_llm_payload(raw: str) -> dict | None:
    """Разобрать и проверить ответ LLM. Любое несоответствие контракту — None."""
    try:
        envelope = json.loads(raw)
        content = envelope["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError):
        return None
    if not isinstance(content, str):
        return None

    parsed = _extract_json_object(content)
    if parsed is None:
        return None

    category = str(parsed.get("category", "")).strip().lower()
    reply = str(parsed.get("reply", "")).strip()
    reason = str(parsed.get("reason", "")).strip()

    if category not in CATEGORIES or not reply:
        return None
    if not _HAS_CYRILLIC.search(reply):
        return None
    return {"category": category, "reply": reply[:MAX_REPLY_LEN], "reason": reason[:300]}


def build_headers(config: "LLMConfig") -> dict[str, str]:
    """Заголовки запроса. Ключ уходит только в Authorization и нигде не логируется."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}",
    }
    if "openrouter.ai" in config.base_url:
        headers["HTTP-Referer"] = APP_URL
        headers["X-Title"] = APP_TITLE
    return headers


def build_body(text: str, config: "LLMConfig", json_mode: bool) -> bytes:
    payload: dict = {
        "model": config.model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text[:MAX_MESSAGE_LEN]},
        ],
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def call_llm(text: str, config: LLMConfig, attempts: int = 3) -> tuple[dict | None, str | None]:
    """Вызов LLM с ретраями. Возвращает (результат, причина отказа).

    Строгий JSON-режим поддерживают не все модели OpenRouter, поэтому на ответ 400/404/422
    повторяем запрос уже без response_format, а разбор ответа терпим к ```-обёртке.
    """
    if not config.base_url.startswith(("http://", "https://")):
        return None, "недопустимый LLM_BASE_URL (ожидается http:// или https://)"

    url = f"{config.base_url}/chat/completions"
    headers = build_headers(config)
    json_mode = True
    last_error: str | None = None

    for _ in range(max(1, attempts)):
        request = urllib.request.Request(
            url,
            data=build_body(text, config, json_mode),
            method="POST",
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=config.timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as error:
            last_error = f"HTTP {error.code}"
            if error.code in (400, 404, 422) and json_mode:
                json_mode = False  # модель не умеет строгий JSON — пробуем без него
                continue
            if error.code < 500:  # ключ, модель или запрос — ретрай не поможет
                break
            continue
        except urllib.error.URLError as error:
            last_error = f"сеть недоступна ({error.reason})"
            continue
        except TimeoutError:
            last_error = f"таймаут {config.timeout:g} с"
            continue
        except Exception as error:  # сбой LLM не должен ронять прогон
            last_error = type(error).__name__
            break

        parsed = _parse_llm_payload(raw)
        if parsed is None:
            last_error = "ответ не соответствует ожидаемому формату"
            continue

        violation = check_reply(parsed["reply"])
        if violation is None:
            return parsed, None
        # Температура 0: повтор вернёт тот же текст, поэтому попытки не тратим.
        return None, violation

    return None, last_error or "неизвестная ошибка"


def triage(text: str, config: LLMConfig | None = None) -> Result:
    """Разобрать одно обращение: сначала LLM (если настроен), иначе и при сбое — правила."""
    text = (text or "").strip()
    note: str | None = None

    if config is not None:
        payload, error = call_llm(text, config)
        if payload is not None:
            return Result(
                text=text,
                category=payload["category"],
                reply=payload["reply"],
                source="llm",
                confidence=None,
                note=payload["reason"] or None,
            )
        note = f"LLM недоступен: {error}. Использованы правила."

    category, confidence, signals = classify_by_rules(text)
    return Result(
        text=text,
        category=category,
        reply=build_reply(text, category),
        source="rules",
        confidence=confidence,
        signals=signals,
        note=note,
    )
