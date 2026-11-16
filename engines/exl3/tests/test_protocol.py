import json
import unittest

from mlxl3_exl3.protocol import MAX_REQUEST_BYTES, RequestError, parse_request


def gen(**fields):
    base = {"type": "generate", "request_id": "r1", "messages": [{"role": "user", "content": "hi"}]}
    base.update(fields)
    return json.dumps(base)


class ParseRequestTests(unittest.TestCase):
    def test_generate_defaults(self):
        request = parse_request(gen())
        self.assertEqual(request["max_tokens"], -1)
        self.assertEqual(request["temperature"], 0.0)
        self.assertEqual(request["top_p"], 1.0)
        self.assertEqual(request["repetition_penalty"], 1.0)
        self.assertIsNone(request["seed"])

    def test_control_requests_keep_only_type_and_id(self):
        self.assertEqual(parse_request('{"type":"cancel","request_id":"x","junk":1}'),
                         {"type": "cancel", "request_id": "x"})
        self.assertEqual(parse_request(b'{"type":"ping"}'), {"type": "ping", "request_id": ""})

    def test_rejections_carry_request_id(self):
        cases = [
            gen(max_tokens=0), gen(max_tokens=-2), gen(max_tokens=1.5), gen(max_tokens=True),
            gen(temperature=-0.1), gen(temperature=5.01), gen(temperature="1"), gen(temperature=True),
            gen(top_k=-1), gen(top_p=0), gen(top_p=1.01), gen(min_p=1.0), gen(min_p=-0.01),
            gen(repetition_penalty=0.49), gen(repetition_penalty=3.01), gen(seed=-1), gen(seed=2**63),
            gen(enable_thinking="yes"), gen(messages=[]), gen(messages="hi"),
            gen(messages=[{"role": "robot", "content": "x"}]),
            gen(messages=[{"role": "user", "content": ["x"]}]),
            gen(messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]),
        ]
        for line in cases:
            with self.subTest(line=line):
                with self.assertRaises(RequestError) as caught:
                    parse_request(line)
                self.assertEqual(caught.exception.request_id, "r1")

    def test_non_finite_numbers(self):
        for value in ("NaN", "Infinity", "-Infinity"):
            line = gen().replace('"r1"', '"r1","temperature":%s' % value)
            with self.assertRaises(RequestError):
                parse_request(line)

    def test_framing_errors(self):
        for line in ("", "[]", "not json", '{"type":"nope"}', '{"type":"generate","messages":[]}',
                     '{"type":"ping","request_id":3}'):
            with self.subTest(line=line):
                with self.assertRaises(RequestError):
                    parse_request(line)
        with self.assertRaises(RequestError):
            parse_request(b"\xff\xfe")
        with self.assertRaises(RequestError):
            parse_request(b"x" * (MAX_REQUEST_BYTES + 1))
        with self.assertRaises(RequestError):
            parse_request("é" * (MAX_REQUEST_BYTES // 2 + 1))

    def test_boundaries_accepted(self):
        request = parse_request(gen(temperature=5, top_p=1, min_p=0.999, repetition_penalty=0.5,
                                    seed=2**63 - 1, max_tokens=1, top_k=0))
        self.assertEqual(request["seed"], 2**63 - 1)
        parse_request(gen(messages=[{"role": "system", "content": ""},
                                    {"role": "tool", "content": "{}"}]))


if __name__ == "__main__":
    unittest.main()


class ToolRequestTests(unittest.TestCase):
    TOOL = {"type": "function", "function": {"name": "exa.search", "parameters": {"type": "object"}}}
    CALL = {"id": "c1", "type": "function", "function": {"name": "exa.search", "arguments": {"q": "x"}}}

    def test_tool_loop_messages_are_kept(self):
        messages = [{"role": "user", "content": "find"},
                    {"role": "assistant", "content": "", "tool_calls": [self.CALL]},
                    {"role": "tool", "content": "result", "name": "exa.search"}]
        request = parse_request(gen(messages=messages, tools=[self.TOOL]))
        self.assertEqual(request["tools"], [self.TOOL])
        self.assertEqual(request["messages"][1]["tool_calls"], [self.CALL])
        self.assertEqual(request["messages"][2]["name"], "exa.search")
        self.assertEqual(parse_request(gen())["tools"], [])
        self.assertNotIn("tool_calls", parse_request(gen(messages=[
            {"role": "assistant", "content": "a", "tool_calls": []}, {"role": "user", "content": "q"}]))["messages"][0])

    def test_bad_tools_are_refused(self):
        user = {"role": "user", "content": "q"}
        cases = [
            gen(tools={"a": 1}), gen(tools=[{"type": "function"}]), gen(tools=[{"function": {"name": ""}}]),
            gen(tools=[3]), gen(tools=[self.TOOL] * 257),
            gen(messages=[{"role": "user", "content": "x", "tool_calls": [self.CALL]}]),
            gen(messages=[{"role": "assistant", "content": "", "tool_calls": [{"id": "c"}]}, user]),
            gen(messages=[{"role": "assistant", "content": "", "tool_calls": "x"}, user]),
            gen(messages=[{"role": "assistant", "content": "", "tool_calls": [self.CALL] * 65}, user]),
            gen(messages=[{"role": "user", "content": "x", "name": "n"}]),
            gen(messages=[{"role": "tool", "content": "x", "name": ""}]),
            gen(messages=[{"role": "tool", "content": "x", "name": "n" * 257}]),
        ]
        for line in cases:
            with self.subTest(line=line[:90]):
                with self.assertRaises(RequestError) as caught:
                    parse_request(line)
                self.assertEqual(caught.exception.request_id, "r1")


class ImageRequestTests(unittest.TestCase):
    PNG = "iVBORw0KGgoAAAANSUhEUg=="  # PNG signature + IHDR start

    def test_images_decode_on_user_messages(self):
        import base64
        jpeg = base64.b64encode(b"\xff\xd8\xff\xe0rest").decode()
        webp = base64.b64encode(b"RIFF\0\0\0\0WEBPVP8X").decode()
        request = parse_request(gen(messages=[{"role": "user", "content": "see", "images": [self.PNG, jpeg, webp]}]))
        images = request["messages"][0]["images"]
        self.assertEqual(len(images), 3)
        self.assertTrue(images[0].startswith(b"\x89PNG"))
        self.assertNotIn("images", parse_request(gen(messages=[{"role": "user", "content": "x", "images": []}]))["messages"][0])

    def test_bad_images_are_refused(self):
        import base64
        gif = base64.b64encode(b"GIF89a....").decode()
        cases = [
            [{"role": "system", "content": "x", "images": [self.PNG]}, {"role": "user", "content": "q"}],
            [{"role": "user", "content": "x", "images": self.PNG}],
            [{"role": "user", "content": "x", "images": [3]}],
            [{"role": "user", "content": "x", "images": ["not base64!"]}],
            [{"role": "user", "content": "x", "images": [gif]}],
            [{"role": "user", "content": "x", "images": [self.PNG] * 17}],
        ]
        for messages in cases:
            with self.subTest(messages=str(messages)[:80]):
                with self.assertRaises(RequestError) as caught:
                    parse_request(gen(messages=messages))
                self.assertEqual(caught.exception.request_id, "r1")
