from __future__ import annotations

import argparse
import os
import socket


def _bypass_proxy_for_localhost() -> None:
    local_hosts = ("localhost", "127.0.0.1", "::1")
    for key in ("NO_PROXY", "no_proxy"):
        existing = [part.strip() for part in os.environ.get(key, "").split(",") if part.strip()]
        for host in local_hosts:
            if host not in existing:
                existing.append(host)
        os.environ[key] = ",".join(existing)

    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")


_bypass_proxy_for_localhost()

from ui.gradio_app import build_app


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _local_network_addresses(port: int) -> list[str]:
    addresses: set[str] = set()
    try:
        hostname = socket.gethostname()
        for item in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = item[4][0]
            if not ip.startswith("127."):
                addresses.add(ip)
    except OSError:
        pass

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            ip = sock.getsockname()[0]
            if not ip.startswith("127."):
                addresses.add(ip)
    except OSError:
        pass

    return [f"http://{ip}:{port}" for ip in sorted(addresses)]


def _is_port_free(host: str, port: int) -> bool:
    bind_host = "127.0.0.1" if host == "0.0.0.0" else host
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            sock.bind((bind_host, port))
            return True
    except OSError:
        return False


def _resolve_port(host: str, requested_port: int, strict: bool) -> int:
    if _is_port_free(host, requested_port):
        return requested_port
    if strict:
        raise OSError(
            f"Port {requested_port} is busy. Close the old GRADV/Python process or run with --port {requested_port + 1}."
        )
    for candidate in range(requested_port + 1, requested_port + 51):
        if _is_port_free(host, candidate):
            print(
                f"Port {requested_port} is busy, so GRADV will use port {candidate} instead.\n"
                f"If you need exactly {requested_port}, close the old Python/GRADV process first."
            )
            return candidate
    raise OSError(
        f"Cannot find a free port near {requested_port}. Close old Python/GRADV processes or pass --port manually."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GRADV: поиск входных аудио-образов класса для моделей распознавания речи."
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("GRADV_HOST", "127.0.0.1"),
        help="Адрес сервера. 127.0.0.1 только для этого ПК, 0.0.0.0 для локальной сети.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("GRADV_PORT", "7860")),
        help="Порт Gradio.",
    )
    parser.add_argument(
        "--lan",
        action="store_true",
        help="Открыть приложение для других устройств в той же локальной сети.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=_env_bool("GRADV_SHARE", False),
        help="Создать временную публичную ссылку Gradio. Работает, пока запущен этот процесс.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Не открывать браузер автоматически.",
    )
    parser.add_argument(
        "--strict-port",
        action="store_true",
        help="Do not switch to another port automatically if the requested port is busy.",
    )
    return parser.parse_args()


def _print_access_help(host: str, port: int, share: bool) -> None:
    print("\nGRADV is starting...")
    print(f"Local URL: http://127.0.0.1:{port}")
    if host == "0.0.0.0":
        network_urls = _local_network_addresses(port)
        if network_urls:
            print("LAN URLs for devices in the same network:")
            for url in network_urls:
                print(f"  {url}")
        else:
            print("LAN mode is enabled, but no local network IP was detected.")
    if share:
        print("Public sharing is enabled. Gradio will print a temporary public URL when it is ready.")
        print("Keep this terminal and computer running while another person uses the link.")
        print("If Gradio prints 'Could not create share link', the local app is still running, but the public tunnel failed.")
        print("Common fixes: disable VPN, try another network, use --lan for the same Wi-Fi, or send the release ZIP.")
        print("If a gradio.live page says 'No interface is running right now', the old share link has expired or the process stopped.")
    print()


if __name__ == "__main__":
    args = _parse_args()
    host = "0.0.0.0" if args.lan else args.host
    port = _resolve_port(host, args.port, args.strict_port)
    _print_access_help(host, port, args.share)
    app = build_app()
    app.launch(
        server_name=host,
        server_port=port,
        inbrowser=not args.no_browser,
        show_error=True,
        share=args.share,
    )
