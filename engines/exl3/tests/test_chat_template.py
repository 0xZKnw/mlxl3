import json
import tempfile
import unittest
from pathlib import Path

from mlxl3_exl3 import chat_template
from mlxl3_exl3.chat_template import TemplateError

CHATML = ("{{ bos_token }}{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
          "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
          "{% if enable_thinking is defined and not enable_thinking %}<think></think>{% endif %}")


class TemplateTests(unittest.TestCase):
    def folder(self, files):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        for name, content in files.items():
            Path(directory.name, name).write_text(content, encoding="utf-8")
        return directory.name

    def test_precedence_jinja_then_json_then_tokenizer_config(self):
        config = json.dumps({"chat_template": "TC", "bos_token": {"content": "<s>"}, "eos_token": "</s>"})
        folder = self.folder({"tokenizer_config.json": config,
                              "chat_template.json": json.dumps({"chat_template": "JSON"}),
                              "chat_template.jinja": "JINJA"})
        self.assertEqual(chat_template.load(folder), ("JINJA", {"bos_token": "<s>", "eos_token": "</s>"}))
        Path(folder, "chat_template.jinja").unlink()
        self.assertEqual(chat_template.load(folder)[0], "JSON")
        Path(folder, "chat_template.json").unlink()
        self.assertEqual(chat_template.load(folder)[0], "TC")

    def test_named_template_list_prefers_default(self):
        config = json.dumps({"chat_template": [{"name": "tool_use", "template": "T"},
                                               {"name": "default", "template": "D"}]})
        self.assertEqual(chat_template.load(self.folder({"tokenizer_config.json": config}))[0], "D")

    def test_missing_or_broken(self):
        with self.assertRaises(TemplateError):
            chat_template.load(self.folder({}))
        with self.assertRaises(TemplateError):
            chat_template.load(self.folder({"tokenizer_config.json": "{"}))

    def test_render_chatml_and_thinking_flag(self):
        messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "Bonjour"}]
        text = chat_template.render(CHATML, {"bos_token": "<s>", "eos_token": ""}, messages)
        self.assertEqual(text, "<s><|im_start|>system\nS<|im_end|>\n<|im_start|>user\nBonjour<|im_end|>\n"
                               "<|im_start|>assistant\n")
        off = chat_template.render(CHATML, {"bos_token": "", "eos_token": ""}, messages, False)
        self.assertTrue(off.endswith("<think></think>"))

    def test_template_errors_and_sandbox(self):
        special = {"bos_token": "", "eos_token": ""}
        with self.assertRaises(TemplateError) as caught:
            chat_template.render("{{ raise_exception('only user turns') }}", special, [])
        self.assertIn("only user turns", str(caught.exception))
        with self.assertRaises(TemplateError):
            chat_template.render("{% for %}", special, [])
        with self.assertRaises(TemplateError):  # sandbox refuses dunder access
            chat_template.render("{{ messages.__class__.__mro__ }}", special, [])

    def test_tojson_keeps_unicode(self):
        text = chat_template.render("{{ messages | tojson }}", {"bos_token": "", "eos_token": ""},
                                    [{"role": "user", "content": "é"}])
        self.assertEqual(text, '[{"role": "user", "content": "é"}]')


if __name__ == "__main__":
    unittest.main()
