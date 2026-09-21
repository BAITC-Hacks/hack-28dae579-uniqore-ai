#!/usr/bin/env python3
"""Разбор обращений: категория и черновик ответа для каждого сообщения.

Запуск без аргументов читает messages.txt и работает офлайн на правилах.
Если в окружении задан LLM_API_KEY, категорию и ответ формирует LLM,
а правила остаются страховкой на случай сбоя.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap

from triage import MAX_MESSAGE_LEN, LLMConfig, Result, triage

# Файлы по умолчанию ищем рядом со скриптом, а не в текущем каталоге:
# `python3 /путь/к/main.py` должен работать из любого места. Путь, переданный
# через --input, остаётся относительным текущему каталогу — так ожидает пользователь.
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(PROJECT_DIR, "messages.txt")
ENV_FILE = os.path.join(PROJECT_DIR, ".env")
WIDTH = 88


class InputError(Exception):
    """Проблема со входными данными: файла нет или он нечитаем."""


def load_env_file(path: str = ENV_FILE) -> list[str]:
    """Подхватить переменные из .env рядом со скриптом.

    Ключ живёт в проекте, а не в окружении всей системы: его видят только запуски
    этого скрипта. Настоящее окружение главнее файла — уже заданные переменные
    не перетираются, так что `LLM_MODEL=... python3 main.py` продолжает работать.

    Возвращает имена загруженных переменных. Значения не возвращаются и не печатаются.
    """
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except OSError:
        return []

    try:
        if os.stat(path).st_mode & 0o077:
            print(
                f"Предупреждение: файл {path} читается не только владельцем. "
                f"Выполните: chmod 600 {path}",
                file=sys.stderr,
            )
    except OSError:
        pass

    loaded: list[str] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        name, separator, value = line.partition("=")
        name, value = name.strip(), value.strip()
        if not separator or not name:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if name not in os.environ:
            os.environ[name] = value
            loaded.append(name)
    return loaded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Классификатор обращений: справка / жалоба / другое + черновик ответа.",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        metavar="ФАЙЛ",
        help="файл с обращениями, по одному на строку (по умолчанию messages.txt рядом со скриптом)",
    )
    parser.add_argument(
        "--text",
        metavar="ТЕКСТ",
        help="разобрать одно обращение из аргумента вместо файла",
    )
    parser.add_argument("--json", action="store_true", help="вывести результат JSON-массивом")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="не обращаться к LLM даже при заданном ключе",
    )
    return parser


def read_lines(path: str) -> list[str]:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.readlines()
    except FileNotFoundError:
        raise InputError(f"файл «{path}» не найден") from None
    except IsADirectoryError:
        raise InputError(f"«{path}» — это каталог, а не файл") from None
    except OSError as error:
        raise InputError(f"не удалось прочитать «{path}»: {error.strerror}") from None


def prepare(raw_lines: list[str]) -> tuple[list[str], list[str]]:
    """Отбросить пустые строки и комментарии, обрезать слишком длинные обращения."""
    messages: list[str] = []
    warnings: list[str] = []
    for line in raw_lines:
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if len(text) > MAX_MESSAGE_LEN:
            warnings.append(
                f"Обращение {len(messages) + 1} длиннее {MAX_MESSAGE_LEN} символов и обрезано."
            )
            text = text[:MAX_MESSAGE_LEN]
        messages.append(text)
    return messages, warnings


def describe_mode(config: LLMConfig | None, no_llm: bool) -> str:
    if config is not None:
        return f"LLM ({config.model}), правила как страховка"
    if no_llm:
        return "правила (LLM отключён флагом --no-llm)"
    return "правила (ключ LLM не задан)"


def format_result(index: int, result: Result, show_note: bool) -> str:
    confidence = "—" if result.confidence is None else f"{result.confidence:.2f}"
    source = "LLM" if result.source == "llm" else "правила"
    lines = [
        f"[{index}] {result.text}",
        f"    Категория:  {result.category}  (источник: {source}, уверенность: {confidence})",
        textwrap.fill(
            result.reply,
            width=WIDTH,
            initial_indent="    Ответ:      ",
            subsequent_indent="                ",
        ),
    ]
    if show_note and result.note:
        lines.append(
            textwrap.fill(
                result.note,
                width=WIDTH,
                initial_indent="    Примечание: ",
                subsequent_indent="                ",
            )
        )
    return "\n".join(lines)


def use_utf8_streams() -> None:
    """Печатать вывод в UTF-8 независимо от локали системы.

    С консолью Python и так говорит в Unicode, но при перенаправлении
    (`python3 main.py > out.txt`, конвейер) берёт кодировку локали — на русской
    Windows это cp1251. Файл с результатами получался не в UTF-8, а `--json`
    ломал разбор у всего, что ждёт UTF-8. На Linux и macOS локаль обычно уже
    UTF-8, и вызов ничего не меняет.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError, ValueError):
            # Поток подменён (тесты, отладчик) или перенастройку не поддерживает.
            # Вывод останется в кодировке локали, но ронять из-за этого прогон незачем.
            pass


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_env_file()
    config = None if args.no_llm else LLMConfig.from_env()

    try:
        raw_lines = [args.text] if args.text is not None else read_lines(args.input)
    except InputError as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 2

    messages, warnings = prepare(raw_lines)
    for warning in warnings:
        print(f"Предупреждение: {warning}", file=sys.stderr)

    if not messages:
        source = "аргументе --text" if args.text is not None else f"файле «{args.input}»"
        print(f"Обращений не найдено: в {source} нет непустых строк.")
        return 0

    results = [triage(message, config) for message in messages]

    # Один и тот же комментарий у всех обращений (например, недоступность LLM)
    # печатаем один раз в шапке, а не пять раз подряд.
    notes = {result.note for result in results if result.note}
    shared_note = notes.pop() if len(notes) == 1 and len(results) > 1 else None

    if args.json:
        payload = [{"index": i, **r.to_dict()} for i, r in enumerate(results, start=1)]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if shared_note:
            print(f"Примечание: {shared_note}", file=sys.stderr)
        return 0

    print(f"Обращений: {len(results)}  |  режим: {describe_mode(config, args.no_llm)}")
    if shared_note:
        print(f"Примечание: {shared_note}")
    print()
    for index, result in enumerate(results, start=1):
        print(format_result(index, result, show_note=shared_note is None))
        print()
    return 0


if __name__ == "__main__":
    # До первой печати: сообщения об ошибках ниже тоже должны уйти в UTF-8.
    use_utf8_streams()
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Прервано пользователем.", file=sys.stderr)
        sys.exit(130)
    except Exception as error:  # наружу не должен пролезать трейсбек
        print(f"Непредвиденная ошибка: {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
