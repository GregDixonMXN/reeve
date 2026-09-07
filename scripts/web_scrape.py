#!/usr/bin/env python3
"""
Reeve Web Scraper Tool (Tool B: "The Eyes — Scrape")
═════════════════════════════════════════════════════
Downloads a public webpage and extracts clean article text.

Input:  {"url": "https://example.com/article", "max_chars": 8000}
Output: {"url": "...", "title": "...", "content": "...", "char_count": 1234}

The downloader resolves and validates every target, pins the connection to the
validated address, repeats validation after redirects, and bounds bytes read
before passing content to trafilatura. It deliberately ignores proxy settings.

Install: pip install trafilatura
"""

from __future__ import annotations

import http.client
import ipaddress
import json
import re
import socket
import ssl
import sys
import urllib.parse
from dataclasses import dataclass
from typing import Mapping


# Output bounds protect the model context; the download bound protects process
# memory before extraction. Neither limit can be raised by tool input.
ABSOLUTE_MAX_CHARS = 16_000
DEFAULT_MAX_CHARS = 8_000
ABSOLUTE_MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024
MAX_REDIRECTS = 5
DOWNLOAD_TIMEOUT_SECONDS = 15

REDIRECT_STATUSES = {301, 302, 303, 307, 308}
BLOCKED_HOSTNAMES = {
    "localhost",
    "localhost.localdomain",
    "metadata",
    "metadata.google.internal",
    "instance-data",
}
BLOCKED_IPV6_TRANSITION_NETWORKS = (
    ipaddress.ip_network("64:ff9b::/96"),  # well-known NAT64 prefix
    ipaddress.ip_network("64:ff9b:1::/48"),  # local-use NAT64 prefix
)


class ScrapeSecurityError(RuntimeError):
    """The requested fetch violates the scraper's network policy."""


class ScrapeDownloadError(RuntimeError):
    """The public target could not be downloaded safely."""


@dataclass(frozen=True)
class ResolvedTarget:
    url: str
    parsed: urllib.parse.SplitResult
    host: str
    port: int
    # Entries are (family, socket type, protocol, sockaddr), copied from
    # getaddrinfo. Connecting to sockaddr rather than host prevents DNS rebinding
    # between authorization and use.
    addresses: tuple[tuple[int, int, int, tuple], ...]


def sanitize_max_chars(value: object) -> int:
    """Return a positive, bounded character limit for model-facing output."""
    if isinstance(value, bool):
        return DEFAULT_MAX_CHARS
    try:
        limit = int(value)
    except (TypeError, ValueError, OverflowError):
        return DEFAULT_MAX_CHARS
    if limit <= 0:
        return DEFAULT_MAX_CHARS
    return min(limit, ABSOLUTE_MAX_CHARS)


def _is_public_address(address: str) -> bool:
    """Only globally routable unicast addresses are valid web targets."""
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return False

    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        elif (
            ip.sixtofour is not None
            or ip.teredo is not None
            or any(ip in network for network in BLOCKED_IPV6_TRANSITION_NETWORKS)
        ):
            # Transition mechanisms can embed an IPv4 loopback/private target
            # inside an otherwise global-looking IPv6 address.
            return False
    return ip.is_global and not ip.is_multicast


def _validate_hostname(host: str) -> str:
    host = host.rstrip(".").lower()
    if not host:
        raise ScrapeSecurityError("URL has no hostname")
    if "%" in host:
        # Scoped IPv6 literals are link-local in practice and should never be
        # sent through a public web-fetch tool.
        raise ScrapeSecurityError("scoped IP addresses are not allowed")
    if (
        host in BLOCKED_HOSTNAMES
        or host.endswith(".localhost")
        or host.endswith(".local")
    ):
        raise ScrapeSecurityError("local and metadata hostnames are not allowed")
    try:
        return host.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise ScrapeSecurityError("hostname is not valid IDNA") from exc


def _resolve_public_addresses(
    host: str, port: int
) -> tuple[tuple[int, int, int, tuple], ...]:
    # Reject private literals without asking the resolver. Non-standard forms
    # such as decimal IPv4 are still handled safely because getaddrinfo's actual
    # result is checked below.
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None and not _is_public_address(str(literal)):
        raise ScrapeSecurityError(
            "loopback, private, link-local, and reserved addresses are not allowed"
        )

    try:
        resolved = socket.getaddrinfo(
            host,
            port,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
        )
    except socket.gaierror as exc:
        raise ScrapeDownloadError(f"hostname resolution failed: {exc}") from exc

    addresses: list[tuple[int, int, int, tuple]] = []
    seen: set[tuple[int, tuple]] = set()
    for family, socktype, proto, _canonname, sockaddr in resolved:
        if family not in (socket.AF_INET, socket.AF_INET6) or not sockaddr:
            continue
        address = sockaddr[0]
        if not _is_public_address(address):
            # If a hostname has a mixture of public and private answers, reject
            # the entire target rather than selecting the convenient answer.
            raise ScrapeSecurityError("hostname resolves to a non-public address")
        key = (family, sockaddr)
        if key in seen:
            continue
        seen.add(key)
        addresses.append((family, socktype, proto, sockaddr))

    if not addresses:
        raise ScrapeDownloadError("hostname did not resolve to a usable public address")
    return tuple(addresses)


def _resolve_target(raw_url: str) -> ResolvedTarget:
    url = str(raw_url).strip()
    if not url:
        raise ScrapeSecurityError("URL is empty")
    if any(ord(char) < 32 or ord(char) == 127 for char in url):
        raise ScrapeSecurityError("URL contains control characters")
    if "://" not in url:
        url = "https://" + url

    try:
        parsed = urllib.parse.urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise ScrapeSecurityError(f"invalid URL: {exc}") from exc

    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise ScrapeSecurityError("only http and https URLs are allowed")
    if parsed.username is not None or parsed.password is not None:
        raise ScrapeSecurityError("URLs containing credentials are not allowed")
    if parsed.hostname is None:
        raise ScrapeSecurityError("URL has no hostname")

    host = _validate_hostname(parsed.hostname)
    if port == 0:
        raise ScrapeSecurityError("URL port must be between 1 and 65535")
    port = port or (443 if scheme == "https" else 80)
    addresses = _resolve_public_addresses(host, port)

    netloc_host = f"[{host}]" if ":" in host else host
    default_port = 443 if scheme == "https" else 80
    netloc = netloc_host if port == default_port else f"{netloc_host}:{port}"
    normalized = urllib.parse.urlunsplit(
        (scheme, netloc, parsed.path or "/", parsed.query, "")
    )
    return ResolvedTarget(
        url=normalized,
        parsed=urllib.parse.urlsplit(normalized),
        host=host,
        port=port,
        addresses=addresses,
    )


def _read_limited(response: http.client.HTTPResponse) -> bytes:
    declared_length = response.getheader("Content-Length")
    if declared_length:
        try:
            declared = int(declared_length)
        except ValueError:
            declared = -1
        if declared > ABSOLUTE_MAX_DOWNLOAD_BYTES:
            raise ScrapeSecurityError(
                f"response exceeds the {ABSOLUTE_MAX_DOWNLOAD_BYTES}-byte download limit"
            )

    body = response.read(ABSOLUTE_MAX_DOWNLOAD_BYTES + 1)
    if len(body) > ABSOLUTE_MAX_DOWNLOAD_BYTES:
        raise ScrapeSecurityError(
            f"response exceeds the {ABSOLUTE_MAX_DOWNLOAD_BYTES}-byte download limit"
        )
    return body


def _request_once(target: ResolvedTarget) -> tuple[int, Mapping[str, str], bytes]:
    request_target = urllib.parse.urlunsplit(
        ("", "", target.parsed.path or "/", target.parsed.query, "")
    )
    last_error: Exception | None = None

    for family, socktype, proto, sockaddr in target.addresses:
        sock: socket.socket | ssl.SSLSocket | None = None
        connection: http.client.HTTPConnection | None = None
        try:
            # Use the already-authorized numeric sockaddr. No hostname is
            # resolved between this point and connect().
            sock = socket.socket(family, socktype, proto)
            sock.settimeout(DOWNLOAD_TIMEOUT_SECONDS)
            sock.connect(sockaddr)
            if target.parsed.scheme == "https":
                context = ssl.create_default_context()
                sock = context.wrap_socket(sock, server_hostname=target.host)

            # HTTPConnection provides correct request framing and a Host header;
            # assigning the pinned socket prevents its connect() from resolving.
            connection = http.client.HTTPConnection(
                target.host, target.port, timeout=DOWNLOAD_TIMEOUT_SECONDS
            )
            connection.sock = sock
            connection.request(
                "GET",
                request_target,
                headers={
                    "Host": target.parsed.netloc,
                    "User-Agent": "Reeve-Web-Scraper/1.0",
                    "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.1",
                    "Accept-Encoding": "identity",
                    "Connection": "close",
                },
            )
            response = connection.getresponse()
            headers = {key.lower(): value for key, value in response.getheaders()}
            body = b"" if response.status in REDIRECT_STATUSES else _read_limited(response)
            return response.status, headers, body
        except ScrapeSecurityError:
            raise
        except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
            last_error = exc
        finally:
            if connection is not None:
                connection.close()
            elif sock is not None:
                sock.close()

    raise ScrapeDownloadError(f"connection failed: {last_error}")


def _decode_body(body: bytes, headers: Mapping[str, str]) -> str:
    content_encoding = headers.get("content-encoding", "").strip().lower()
    if content_encoding not in ("", "identity"):
        # Accept-Encoding requests identity. Refusing unexpected compression
        # avoids turning a bounded wire response into an unbounded decode.
        raise ScrapeSecurityError("compressed responses are not accepted")

    charset = "utf-8"
    content_type = headers.get("content-type", "")
    match = re.search(r"charset\s*=\s*[\"']?([^;\s\"']+)", content_type, re.I)
    if match:
        charset = match.group(1)
    try:
        return body.decode(charset, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


def secure_download(url: str) -> tuple[str, str]:
    """Download a public URL with SSRF, rebinding, redirect, and size defenses."""
    current_url = url
    for redirect_count in range(MAX_REDIRECTS + 1):
        target = _resolve_target(current_url)
        status, headers, body = _request_once(target)

        if status in REDIRECT_STATUSES:
            location = headers.get("location", "").strip()
            if not location:
                raise ScrapeDownloadError("redirect response has no Location header")
            if redirect_count >= MAX_REDIRECTS:
                raise ScrapeSecurityError("too many redirects")
            # The next loop resolves and validates the redirect target before
            # any new socket is created.
            current_url = urllib.parse.urljoin(target.url, location)
            continue

        if status < 200 or status >= 300:
            raise ScrapeDownloadError(f"HTTP {status}")
        return target.url, _decode_body(body, headers)

    raise ScrapeSecurityError("too many redirects")


def scrape(url: str, max_chars: int = DEFAULT_MAX_CHARS) -> dict:
    max_chars = sanitize_max_chars(max_chars)

    try:
        import trafilatura
    except ImportError:
        return {"error": "trafilatura not installed. Run: pip install trafilatura"}

    try:
        final_url, downloaded = secure_download(url)

        content = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
            favor_precision=True,
        )

        if not content:
            return {"error": "No extractable content found", "url": final_url}

        metadata = trafilatura.extract(
            downloaded,
            output_format="json",
            include_comments=False,
        )
        title = ""
        if metadata:
            try:
                meta_dict = json.loads(metadata)
                title = meta_dict.get("title", "")
            except (json.JSONDecodeError, TypeError):
                pass

        original_len = len(content)
        if original_len > max_chars:
            content = content[:max_chars]
            content += (
                f"\n\n[TRUNCATED: showing {max_chars} of {original_len} characters]"
            )

        return {
            "url": final_url,
            "title": title,
            "content": content,
            "char_count": len(content),
            "original_char_count": original_len,
            "truncated": original_len > max_chars,
        }
    except ScrapeSecurityError as exc:
        return {"error": f"Scrape blocked: {exc}", "url": url}
    except ScrapeDownloadError as exc:
        return {"error": f"Scrape failed: {exc}", "url": url}
    except Exception as exc:
        return {"error": f"Scrape failed: {exc}", "url": url}


def main() -> None:
    raw = sys.stdin.read().strip()
    if not raw:
        print(json.dumps({"error": "No input"}))
        return

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"url": raw}

    url = data.get("url", "")
    if not url:
        print(json.dumps({"error": "No URL provided"}))
        return

    max_chars = sanitize_max_chars(data.get("max_chars", DEFAULT_MAX_CHARS))
    result = scrape(url, max_chars)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
