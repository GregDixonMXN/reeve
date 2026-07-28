import socket
import unittest
from unittest import mock

import web_scrape


def ipv4_result(address: str, port: int = 80):
    return (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, port))


def ipv6_result(address: str, port: int = 80):
    return (
        socket.AF_INET6,
        socket.SOCK_STREAM,
        socket.IPPROTO_TCP,
        "",
        (address, port, 0, 0),
    )


class FakeResponse:
    def __init__(self, body=b"", headers=None, status=200):
        self.body = body
        self.headers = headers or {}
        self.status = status
        self.read_calls = 0

    def getheader(self, name):
        return self.headers.get(name) or self.headers.get(name.lower())

    def getheaders(self):
        return list(self.headers.items())

    def read(self, amount):
        self.read_calls += 1
        return self.body[:amount]


class WebScrapeSecurityTests(unittest.TestCase):
    def test_rejects_non_public_ip_literals(self):
        urls = [
            "http://127.0.0.1/",
            "http://10.1.2.3/",
            "http://169.254.169.254/latest/meta-data/",
            "http://[::1]/",
            "http://[fe80::1]/",
            "http://[fd00:ec2::254]/",
            "http://[::ffff:127.0.0.1]/",
        ]
        for url in urls:
            with self.subTest(url=url):
                with self.assertRaises(web_scrape.ScrapeSecurityError):
                    web_scrape._resolve_target(url)

    def test_rejects_metadata_and_local_hostnames_before_dns(self):
        with mock.patch.object(web_scrape.socket, "getaddrinfo") as resolver:
            for url in (
                "http://localhost/",
                "http://service.localhost/",
                "http://printer.local/",
                "http://metadata.google.internal/computeMetadata/v1/",
            ):
                with self.subTest(url=url):
                    with self.assertRaises(web_scrape.ScrapeSecurityError):
                        web_scrape._resolve_target(url)
            resolver.assert_not_called()

    def test_rejects_dns_resolution_to_private_address(self):
        with mock.patch.object(
            web_scrape.socket,
            "getaddrinfo",
            return_value=[ipv4_result("10.0.0.8")],
        ):
            with self.assertRaises(web_scrape.ScrapeSecurityError):
                web_scrape._resolve_target("http://attacker.example/")

    def test_rejects_nonstandard_loopback_ip_notation_after_resolution(self):
        for host in ("2130706433", "017700000001", "0x7f000001"):
            with self.subTest(host=host):
                with mock.patch.object(
                    web_scrape.socket,
                    "getaddrinfo",
                    return_value=[ipv4_result("127.0.0.1")],
                ):
                    with self.assertRaises(web_scrape.ScrapeSecurityError):
                        web_scrape._resolve_target(f"http://{host}/")

    def test_rejects_mixed_public_and_private_dns_answers(self):
        with mock.patch.object(
            web_scrape.socket,
            "getaddrinfo",
            return_value=[
                ipv4_result("93.184.216.34"),
                ipv4_result("192.168.1.20"),
            ],
        ):
            with self.assertRaises(web_scrape.ScrapeSecurityError):
                web_scrape._resolve_target("http://mixed.example/")

    def test_public_dns_answer_is_pinned_in_resolved_target(self):
        answer = ipv4_result("93.184.216.34")
        with mock.patch.object(
            web_scrape.socket, "getaddrinfo", return_value=[answer]
        ) as resolver:
            target = web_scrape._resolve_target("http://public.example/path")
        self.assertEqual(target.addresses, ((answer[0], answer[1], answer[2], answer[4]),))
        resolver.assert_called_once()

    def test_redirect_target_is_revalidated_before_second_request(self):
        def resolve(host, port, **_kwargs):
            if host == "public.example":
                return [ipv4_result("93.184.216.34", port)]
            if host == "internal.example":
                return [ipv4_result("10.0.0.9", port)]
            raise AssertionError(f"unexpected host {host}")

        requests = []

        def request_once(target):
            requests.append(target.url)
            return 302, {"location": "http://internal.example/secret"}, b""

        with mock.patch.object(web_scrape.socket, "getaddrinfo", side_effect=resolve):
            with mock.patch.object(web_scrape, "_request_once", side_effect=request_once):
                with self.assertRaises(web_scrape.ScrapeSecurityError):
                    web_scrape.secure_download("http://public.example/")
        self.assertEqual(requests, ["http://public.example/"])

    def test_redirect_to_loopback_literal_is_rejected(self):
        with mock.patch.object(
            web_scrape.socket,
            "getaddrinfo",
            return_value=[ipv4_result("93.184.216.34")],
        ):
            with mock.patch.object(
                web_scrape,
                "_request_once",
                return_value=(302, {"location": "http://127.0.0.1/admin"}, b""),
            ) as request_once:
                with self.assertRaises(web_scrape.ScrapeSecurityError):
                    web_scrape.secure_download("http://public.example/")
        request_once.assert_called_once()

    def test_declared_download_too_large_is_rejected_without_reading(self):
        response = FakeResponse(
            headers={"Content-Length": str(web_scrape.ABSOLUTE_MAX_DOWNLOAD_BYTES + 1)}
        )
        with self.assertRaises(web_scrape.ScrapeSecurityError):
            web_scrape._read_limited(response)
        self.assertEqual(response.read_calls, 0)

    def test_streamed_download_too_large_is_rejected(self):
        response = FakeResponse(
            body=b"x" * (web_scrape.ABSOLUTE_MAX_DOWNLOAD_BYTES + 1)
        )
        with self.assertRaises(web_scrape.ScrapeSecurityError):
            web_scrape._read_limited(response)

    def test_max_chars_is_positive_and_bounded(self):
        default = web_scrape.DEFAULT_MAX_CHARS
        maximum = web_scrape.ABSOLUTE_MAX_CHARS
        for value in (None, "invalid", -1, 0, True):
            with self.subTest(value=value):
                self.assertEqual(web_scrape.sanitize_max_chars(value), default)
        self.assertEqual(web_scrape.sanitize_max_chars("12"), 12)
        self.assertEqual(web_scrape.sanitize_max_chars(maximum * 100), maximum)

    def test_rejects_credentials_control_characters_and_non_http_schemes(self):
        for url in (
            "http://user:secret@example.com/",
            "http://example.com/\r\nX-Test: injected",
            "file:///etc/passwd",
            "gopher://example.com/",
        ):
            with self.subTest(url=url):
                with self.assertRaises(web_scrape.ScrapeSecurityError):
                    web_scrape._resolve_target(url)

    def test_compressed_response_is_rejected_before_decompression(self):
        with self.assertRaises(web_scrape.ScrapeSecurityError):
            web_scrape._decode_body(b"compressed", {"content-encoding": "gzip"})


if __name__ == "__main__":
    unittest.main()
