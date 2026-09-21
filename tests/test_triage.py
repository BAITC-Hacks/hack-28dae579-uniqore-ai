"""Тесты классификатора. Сеть не используется: LLM-путь проверяется на моках."""

from __future__ import annotations

import io
import json
import os
import sys
import unittest
import urllib.error
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main as cli  # noqa: E402
import triage as engine  # noqa: E402

CASE_MESSAGES = [
    ("Как получить справку о месте учёбы?", engine.CATEGORY_HELP),
    ("В столовой очередь, еда холодная.", engine.CATEGORY_COMPLAINT),
    ("Хочу записаться на консультацию завтра.", engine.CATEGORY_OTHER),
    ("Пропал Wi-Fi в корпусе B.", engine.CATEGORY_COMPLAINT),
    ("Где парковка для гостей?", engine.CATEGORY_HELP),
]

# Обращения, которых нет в задании: проверяют, что правила общие,
# а не подогнаны под пять тестовых строк.
UNSEEN_MESSAGES = [
    ("Не работает принтер в библиотеке.", engine.CATEGORY_COMPLAINT),
    ("Какие документы нужны для общежития?", engine.CATEGORY_HELP),
    ("Запишите меня к декану на пятницу.", engine.CATEGORY_OTHER),
    ("Спасибо за помощь!", engine.CATEGORY_OTHER),
    ("Когда работает библиотека?", engine.CATEGORY_HELP),
    ("Третий день нет горячей воды, это безобразие!", engine.CATEGORY_COMPLAINT),
]


class ClassifyByRulesTest(unittest.TestCase):
    def test_case_messages(self):
        for text, expected in CASE_MESSAGES:
            with self.subTest(text=text):
                category, _, _ = engine.classify_by_rules(text)
                self.assertEqual(expected, category)

    def test_unseen_messages(self):
        for text, expected in UNSEEN_MESSAGES:
            with self.subTest(text=text):
                category, _, _ = engine.classify_by_rules(text)
                self.assertEqual(expected, category)

    def test_deterministic(self):
        first = engine.classify_by_rules(CASE_MESSAGES[0][0])
        second = engine.classify_by_rules(CASE_MESSAGES[0][0])
        self.assertEqual(first, second)

    def test_confidence_in_range(self):
        for text, _ in CASE_MESSAGES + UNSEEN_MESSAGES:
            with self.subTest(text=text):
                _, confidence, _ = engine.classify_by_rules(text)
                self.assertGreaterEqual(confidence, 0.0)
                self.assertLessEqual(confidence, 1.0)

    def test_signals_are_reported(self):
        _, _, signals = engine.classify_by_rules("Пропал Wi-Fi в корпусе B.")
        self.assertTrue(signals)

    def test_noise_does_not_crash(self):
        for text in ["", "   ", "Ыыыы 😀😀", "???", "12345", "a" * 5000, "\t\n"]:
            with self.subTest(text=text[:20]):
                category, confidence, _ = engine.classify_by_rules(text)
                self.assertIn(category, engine.CATEGORIES)
                self.assertIsInstance(confidence, float)

    def test_unknown_input_falls_back_to_other(self):
        category, confidence, _ = engine.classify_by_rules("Ыыыы 😀😀")
        self.assertEqual(engine.CATEGORY_OTHER, category)
        self.assertEqual(0.0, confidence)


class ReplyTest(unittest.TestCase):
    def test_reply_is_russian_and_mentions_topic(self):
        for text, expected in CASE_MESSAGES:
            with self.subTest(text=text):
                reply = engine.build_reply(text, expected)
                self.assertTrue(reply.strip())
                self.assertRegex(reply, r"[а-яё]")
                self.assertIn(engine.extract_topic(text), reply)

    def test_reply_for_empty_text(self):
        reply = engine.build_reply("", engine.CATEGORY_OTHER)
        self.assertIn("без текста", reply)

    def test_topic_is_truncated(self):
        topic = engine.extract_topic("слово " * 100)
        self.assertLessEqual(len(topic), 81)
        self.assertTrue(topic.endswith("…"))

    def test_every_category_has_template(self):
        for category in engine.CATEGORIES:
            with self.subTest(category=category):
                self.assertTrue(engine.build_reply("тест", category).strip())


class LLMConfigTest(unittest.TestCase):
    def test_no_key_means_no_llm(self):
        self.assertIsNone(engine.LLMConfig.from_env({}))

    def test_key_from_env(self):
        config = engine.LLMConfig.from_env({"LLM_API_KEY": "secret", "LLM_MODEL": "m"})
        self.assertIsNotNone(config)
        self.assertEqual("m", config.model)
        self.assertEqual(engine.DEFAULT_BASE_URL, config.base_url)

    def test_openai_key_is_accepted(self):
        config = engine.LLMConfig.from_env({"OPENAI_API_KEY": "secret"})
        self.assertIsNotNone(config)

    def test_broken_timeout_falls_back_to_default(self):
        config = engine.LLMConfig.from_env({"LLM_API_KEY": "k", "LLM_TIMEOUT": "нет"})
        self.assertEqual(engine.DEFAULT_TIMEOUT, config.timeout)

    def test_trailing_slash_is_stripped(self):
        config = engine.LLMConfig.from_env({"LLM_API_KEY": "k", "LLM_BASE_URL": "https://x/v1/"})
        self.assertEqual("https://x/v1", config.base_url)


class LLMPayloadTest(unittest.TestCase):
    @staticmethod
    def envelope(content: str) -> str:
        return json.dumps({"choices": [{"message": {"content": content}}]})

    def test_valid_payload(self):
        raw = self.envelope(json.dumps({"category": "жалоба", "reason": "r", "reply": "Здравствуйте!"}))
        parsed = engine._parse_llm_payload(raw)
        self.assertEqual("жалоба", parsed["category"])

    def test_unknown_category_is_rejected(self):
        raw = self.envelope(json.dumps({"category": "urgent", "reply": "Здравствуйте!"}))
        self.assertIsNone(engine._parse_llm_payload(raw))

    def test_non_russian_reply_is_rejected(self):
        raw = self.envelope(json.dumps({"category": "жалоба", "reply": "Hello there"}))
        self.assertIsNone(engine._parse_llm_payload(raw))

    def test_empty_reply_is_rejected(self):
        raw = self.envelope(json.dumps({"category": "жалоба", "reply": "  "}))
        self.assertIsNone(engine._parse_llm_payload(raw))

    def test_malformed_json_is_rejected(self):
        self.assertIsNone(engine._parse_llm_payload("не json"))
        self.assertIsNone(engine._parse_llm_payload(self.envelope("тоже не json")))
        self.assertIsNone(engine._parse_llm_payload(json.dumps({"choices": []})))

    def test_long_reply_is_capped(self):
        raw = self.envelope(json.dumps({"category": "другое", "reply": "я" * 5000}))
        parsed = engine._parse_llm_payload(raw)
        self.assertEqual(engine.MAX_REPLY_LEN, len(parsed["reply"]))


class TriageTest(unittest.TestCase):
    def test_rules_path_without_config(self):
        result = engine.triage("Пропал Wi-Fi в корпусе B.")
        self.assertEqual("rules", result.source)
        self.assertEqual(engine.CATEGORY_COMPLAINT, result.category)
        self.assertIsNone(result.note)

    def test_llm_path_is_used_when_available(self):
        config = engine.LLMConfig(api_key="k")
        payload = {"category": "справка", "reply": "Здравствуйте! Ответ.", "reason": "вопрос"}
        with patch.object(engine, "call_llm", return_value=(payload, None)) as mocked:
            result = engine.triage("Где столовая?", config)
        mocked.assert_called_once()
        self.assertEqual("llm", result.source)
        self.assertEqual("справка", result.category)
        self.assertIsNone(result.confidence)

    def test_llm_failure_falls_back_to_rules(self):
        config = engine.LLMConfig(api_key="k")
        with patch.object(engine, "call_llm", return_value=(None, "таймаут 20 с")):
            result = engine.triage("Пропал Wi-Fi в корпусе B.", config)
        self.assertEqual("rules", result.source)
        self.assertEqual(engine.CATEGORY_COMPLAINT, result.category)
        self.assertIn("таймаут", result.note)

    def test_bad_base_url_is_rejected_without_network(self):
        config = engine.LLMConfig(api_key="k", base_url="file:///etc/passwd")
        payload, error = engine.call_llm("текст", config)
        self.assertIsNone(payload)
        self.assertIn("LLM_BASE_URL", error)

    def test_result_serializes(self):
        data = engine.triage("Где парковка для гостей?").to_dict()
        self.assertEqual(engine.CATEGORY_HELP, data["category"])
        self.assertTrue(data["reply"])
        json.dumps(data, ensure_ascii=False)


class CliTest(unittest.TestCase):
    def run_cli(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cli.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_default_run_prints_five_results(self):
        code, out, _ = self.run_cli(["--no-llm"])
        self.assertEqual(0, code)
        for index in range(1, 6):
            self.assertIn(f"[{index}]", out)

    def test_json_output_is_valid(self):
        code, out, _ = self.run_cli(["--no-llm", "--json"])
        self.assertEqual(0, code)
        payload = json.loads(out)
        self.assertEqual(5, len(payload))
        for item in payload:
            self.assertTrue(item["reply"].strip())
            self.assertIn(item["category"], engine.CATEGORIES)

    def test_missing_file_exits_with_two(self):
        code, _, err = self.run_cli(["--no-llm", "--input", "нет-такого-файла.txt"])
        self.assertEqual(2, code)
        self.assertIn("не найден", err)

    def test_empty_input_is_not_an_error(self):
        code, out, _ = self.run_cli(["--no-llm", "--input", os.devnull])
        self.assertEqual(0, code)
        self.assertIn("Обращений не найдено", out)

    def test_single_text_argument(self):
        code, out, _ = self.run_cli(["--no-llm", "--text", "Ыыыы 😀😀"])
        self.assertEqual(0, code)
        self.assertIn(engine.CATEGORY_OTHER, out)

    def test_blank_lines_and_comments_are_skipped(self):
        messages, warnings = cli.prepare(["# комментарий\n", "\n", "  \n", "Где буфет?\n"])
        self.assertEqual(["Где буфет?"], messages)
        self.assertEqual([], warnings)

    def test_long_line_is_truncated_with_warning(self):
        messages, warnings = cli.prepare(["я" * (engine.MAX_MESSAGE_LEN + 50)])
        self.assertEqual(engine.MAX_MESSAGE_LEN, len(messages[0]))
        self.assertEqual(1, len(warnings))


class FakeResponse:
    """Минимальная замена ответа urlopen для тестов без сети."""

    def __init__(self, body: str):
        self._body = body.encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class OpenRouterTest(unittest.TestCase):
    VALID_CONTENT = json.dumps({"category": "жалоба", "reason": "r", "reply": "Здравствуйте!"})

    @classmethod
    def envelope(cls, content: str | None = None) -> str:
        return json.dumps({"choices": [{"message": {"content": content or cls.VALID_CONTENT}}]})

    def test_defaults_point_to_openrouter(self):
        config = engine.LLMConfig.from_env({"LLM_API_KEY": "k"})
        self.assertEqual("https://openrouter.ai/api/v1", config.base_url)
        self.assertIn("/", config.model)

    def test_openrouter_key_is_accepted(self):
        config = engine.LLMConfig.from_env({"OPENROUTER_API_KEY": "k"})
        self.assertIsNotNone(config)
        self.assertEqual("k", config.api_key)

    def test_attribution_headers_only_for_openrouter(self):
        openrouter = engine.build_headers(engine.LLMConfig(api_key="k"))
        self.assertEqual(engine.APP_URL, openrouter["HTTP-Referer"])
        self.assertEqual(engine.APP_TITLE, openrouter["X-Title"])

        other = engine.build_headers(engine.LLMConfig(api_key="k", base_url="https://api.openai.com/v1"))
        self.assertNotIn("HTTP-Referer", other)
        self.assertEqual("Bearer k", other["Authorization"])

    def test_fenced_json_is_accepted(self):
        raw = self.envelope(f"```json\n{self.VALID_CONTENT}\n```")
        self.assertEqual("жалоба", engine._parse_llm_payload(raw)["category"])

    def test_json_wrapped_in_prose_is_accepted(self):
        raw = self.envelope(f"Вот результат: {self.VALID_CONTENT} — готово.")
        self.assertEqual("жалоба", engine._parse_llm_payload(raw)["category"])

    def test_successful_call(self):
        config = engine.LLMConfig(api_key="k", timeout=1)
        with patch("urllib.request.urlopen", lambda request, timeout=None: FakeResponse(self.envelope())):
            payload, error = engine.call_llm("Пропал Wi-Fi.", config)
        self.assertIsNone(error)
        self.assertEqual("жалоба", payload["category"])

    def test_json_mode_is_dropped_after_400(self):
        """Не все модели OpenRouter умеют response_format — второй запрос идёт без него."""
        config = engine.LLMConfig(api_key="k", timeout=1)
        bodies = []

        def fake_urlopen(request, timeout=None):
            bodies.append(json.loads(request.data.decode("utf-8")))
            if len(bodies) == 1:
                raise urllib.error.HTTPError("url", 400, "bad request", None, None)
            return FakeResponse(self.envelope())

        with patch("urllib.request.urlopen", fake_urlopen):
            payload, error = engine.call_llm("Пропал Wi-Fi.", config)

        self.assertIsNone(error)
        self.assertEqual("жалоба", payload["category"])
        self.assertIn("response_format", bodies[0])
        self.assertNotIn("response_format", bodies[1])

    def test_auth_error_is_not_retried(self):
        config = engine.LLMConfig(api_key="bad", timeout=1)
        calls = []

        def fake_urlopen(request, timeout=None):
            calls.append(1)
            raise urllib.error.HTTPError("url", 401, "unauthorized", None, None)

        with patch("urllib.request.urlopen", fake_urlopen):
            payload, error = engine.call_llm("текст", config)

        self.assertIsNone(payload)
        self.assertEqual("HTTP 401", error)
        self.assertEqual(1, len(calls))

    def test_server_error_is_retried_then_gives_up(self):
        config = engine.LLMConfig(api_key="k", timeout=1)
        calls = []

        def fake_urlopen(request, timeout=None):
            calls.append(1)
            raise urllib.error.HTTPError("url", 503, "unavailable", None, None)

        with patch("urllib.request.urlopen", fake_urlopen):
            payload, error = engine.call_llm("текст", config)

        self.assertIsNone(payload)
        self.assertEqual("HTTP 503", error)
        self.assertEqual(3, len(calls))

    def test_request_targets_chat_completions(self):
        config = engine.LLMConfig(api_key="k", timeout=1)
        seen = {}

        def fake_urlopen(request, timeout=None):
            seen["url"] = request.full_url
            return FakeResponse(self.envelope())

        with patch("urllib.request.urlopen", fake_urlopen):
            engine.call_llm("текст", config)

        self.assertEqual("https://openrouter.ai/api/v1/chat/completions", seen["url"])


if __name__ == "__main__":
    unittest.main()
