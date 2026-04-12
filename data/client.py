"""
Binance REST API 客户端
- 自动重试 & 指数退避
- 请求速率限制
- 完整错误处理
- 请求延迟统计
- 可选 HTTP(S) 代理（Clash 等）
"""
import os
import re
import subprocess
import time
import hmac
import hashlib
import threading
from collections import deque
from urllib.parse import urlencode, urlparse
from typing import Optional, Any

import requests
import pandas as pd

from config.loader import Config
from utils.logger import get_logger

logger = get_logger("client")


def _is_wsl() -> bool:
    try:
        with open("/proc/version", "r", encoding="utf-8") as f:
            v = f.read().lower()
        return "microsoft" in v or "wsl" in v
    except OSError:
        return False


# WSL 新版里 resolv.conf 的 nameserver 常为 10.255.255.254，仅作 DNS 转发，不能连 Clash 端口
_WSL_BAD_NAMESERVERS = frozenset({"10.255.255.254", "127.0.0.1", "::1"})


def _wsl_default_gateway() -> Optional[str]:
    """WSL2 默认网关一般是 Windows 宿主机在 vSwitch 上的 IP，可访问 Allow LAN 后的 Clash。"""
    try:
        out = subprocess.run(
            ["ip", "route", "show", "default"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode != 0 or not (out.stdout or "").strip():
            return None
        m = re.search(r"^default\s+via\s+(\S+)", out.stdout.strip(), re.MULTILINE)
        return m.group(1) if m else None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _wsl_resolv_nameservers() -> list[str]:
    ips: list[str] = []
    try:
        with open("/etc/resolv.conf", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("nameserver "):
                    parts = line.split()
                    if len(parts) >= 2:
                        ips.append(parts[1])
    except OSError:
        pass
    return ips


def _wsl_windows_host_ip() -> Optional[str]:
    """
    优先 default via（ip route），再尝试 resolv.conf 里非伪地址的 nameserver。
    """
    gw = _wsl_default_gateway()
    if gw:
        return gw
    for ip in _wsl_resolv_nameservers():
        if ip not in _WSL_BAD_NAMESERVERS:
            return ip
    return None


def _normalize_proxy_url(url: str) -> Optional[str]:
    u = (url or "").strip()
    if not u or u.startswith("${"):
        return None
    return u


def _parse_wsl_host_field(val: Any) -> Optional[str]:
    """配置项 wsl_host：纯 IP、或带端口 172.x.x.x:7890、或 http://172.x.x.x"""
    if val is None:
        return None
    v = str(val).strip()
    if not v or v.startswith("${"):
        return None
    if "://" in v:
        h = urlparse(v).hostname
        return h
    return v.split(":")[0].strip() or None


def _resolve_rest_proxy(config: Config, *, _log_wsl: bool = True) -> Optional[str]:
    """
    优先级：exchange.proxy（字符串或对象）→ wsl_clash_port（仅 WSL）→ 环境变量 HTTPS_PROXY / HTTP_PROXY。
    """
    exc = config.exchange
    try:
        p = exc.proxy
    except AttributeError:
        p = None

    if isinstance(p, str):
        url = _normalize_proxy_url(p)
        d = {}
    elif p is not None and hasattr(p, "to_dict"):
        d = p.to_dict()
        url = _normalize_proxy_url(
            str(d.get("url") or d.get("https") or d.get("http") or ""))
    elif p is not None and hasattr(p, "_data"):
        d = p._data
        url = _normalize_proxy_url(
            str(d.get("url") or d.get("https") or d.get("http") or ""))
    else:
        d = {}
        url = None

    wsl_port = d.get("wsl_clash_port")
    try:
        wsl_port = int(wsl_port) if wsl_port is not None and str(wsl_port).strip() != "" else 0
    except (TypeError, ValueError):
        wsl_port = 0

    # WSL 内 127.0.0.1 指向 Linux，Clash 在 Windows 上时应走宿主机 IP（优先网关，非 10.255.255.254）
    if _is_wsl() and wsl_port > 0:
        manual_host = _parse_wsl_host_field(d.get("wsl_host"))
        host = manual_host or _wsl_windows_host_ip()
        if host:
            url = f"http://{host}:{wsl_port}"
            if _log_wsl:
                src = "wsl_host 手动" if manual_host else "自动(默认网关/resolv)"
                logger.info(f"WSL REST 代理 | {url} | {src}")

    if not url:
        for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
            v = os.environ.get(key)
            if v and v.strip():
                url = v.strip()
                break

    return _normalize_proxy_url(url) if url else None


def network_probe_diagnostics(config: Config) -> dict[str, Any]:
    """
    本机网络/代理诊断信息（供 main.py --network-probe 打印）。
    在 Cursor Agent 的沙箱或云端无法访问你的 WSL/Clash，必须在用户机器上执行才能闭环。
    """
    d_proxy: dict = {}
    try:
        p = config.exchange.proxy
        if isinstance(p, str):
            d_proxy = {"url": p}
        elif p is not None and hasattr(p, "to_dict"):
            d_proxy = dict(p.to_dict())
        elif p is not None and hasattr(p, "_data"):
            d_proxy = dict(p._data)
    except AttributeError:
        pass
    wsl = _is_wsl()
    return {
        "is_wsl": wsl,
        "wsl_default_gateway": _wsl_default_gateway() if wsl else None,
        "wsl_resolv_nameservers": _wsl_resolv_nameservers() if wsl else [],
        "wsl_effective_host_auto": _wsl_windows_host_ip() if wsl else None,
        "config_proxy_keys": {k: d_proxy.get(k) for k in ("url", "wsl_clash_port", "wsl_host") if k in d_proxy},
        "resolved_rest_proxy": _resolve_rest_proxy(config, _log_wsl=False),
        "rest_base_url": (
            config.exchange.base_url_test
            if config.exchange.testnet
            else config.exchange.base_url_live
        ),
    }


class RateLimiter:
    """令牌桶限流器"""

    def __init__(self, max_per_minute: int = 1200):
        self._max = max_per_minute
        self._tokens = deque()
        self._lock = threading.Lock()

    def acquire(self):
        with self._lock:
            now = time.time()
            # 清除 60 秒前的令牌
            while self._tokens and self._tokens[0] < now - 60:
                self._tokens.popleft()
            if len(self._tokens) >= self._max:
                sleep_time = 60 - (now - self._tokens[0])
                if sleep_time > 0:
                    logger.warning(f"触发限流，等待 {sleep_time:.1f}s")
                    time.sleep(sleep_time)
            self._tokens.append(time.time())


class BinanceAPIError(Exception):
    def __init__(self, status_code: int, code: int, msg: str):
        self.status_code = status_code
        self.code = code
        self.msg = msg
        super().__init__(f"[{status_code}] code={code}: {msg}")


class BinanceClient:
    """
    生产级 Binance REST 客户端
    """

    def __init__(self, config: Config):
        exc = config.exchange
        self.api_key = exc.api_key
        self.api_secret = exc.api_secret
        self.base_url = exc.base_url_test if exc.testnet else exc.base_url_live

        retry_cfg = exc.retry.to_dict() if hasattr(exc.retry, "to_dict") else exc.retry._data
        self._max_retries = retry_cfg.get("max_retries", 3)
        self._backoff = retry_cfg.get("backoff_factor", 0.5)
        self._retry_on = set(retry_cfg.get("retry_on_status", [429, 500, 502, 503]))

        timeout_cfg = exc.timeout.to_dict() if hasattr(exc.timeout, "to_dict") else exc.timeout._data
        self._timeout = (timeout_cfg.get("connect", 5), timeout_cfg.get("read", 15))

        rl_cfg = exc.rate_limit.to_dict() if hasattr(exc.rate_limit, "to_dict") else exc.rate_limit._data
        self._limiter = RateLimiter(rl_cfg.get("requests_per_minute", 1200))

        self._session = requests.Session()
        self._session.headers.update({
            "X-MBX-APIKEY": self.api_key,
            "Content-Type": "application/x-www-form-urlencoded",
        })

        proxy_url = _resolve_rest_proxy(config)
        if proxy_url:
            self._session.proxies = {"http": proxy_url, "https": proxy_url}
            self._session.trust_env = False
            logger.info(f"Binance REST 代理 | {proxy_url}")
        else:
            self._session.trust_env = True

        # 统计
        self._request_count = 0
        self._error_count = 0
        self._total_latency = 0.0

        logger.info(f"Binance 客户端初始化 | URL={self.base_url} | 重试={self._max_retries}")

    # ----------------------------------------------------------
    # 核心请求方法
    # ----------------------------------------------------------
    def _sign(self, params: dict) -> dict:
        params["timestamp"] = int(time.time() * 1000)
        qs = urlencode(params)
        sig = hmac.new(
            self.api_secret.encode(), qs.encode(), hashlib.sha256
        ).hexdigest()
        params["signature"] = sig
        return params

    def _request(self, method: str, path: str,
                 params: Optional[dict] = None,
                 signed: bool = False) -> Any:
        params = dict(params or {})
        if signed:
            params = self._sign(params)

        url = f"{self.base_url}{path}"
        last_error = None

        for attempt in range(self._max_retries + 1):
            self._limiter.acquire()
            t0 = time.time()
            try:
                resp = self._session.request(
                    method, url, params=params, timeout=self._timeout)
                latency = (time.time() - t0) * 1000
                self._request_count += 1
                self._total_latency += latency

                if resp.status_code == 200:
                    return resp.json()

                # 限流
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 5))
                    logger.warning(f"429 限流，等待 {retry_after}s")
                    time.sleep(retry_after)
                    continue

                # IP 封禁
                if resp.status_code == 418:
                    ban_until = resp.headers.get("Retry-After", 120)
                    logger.error(f"418 IP被封禁 {ban_until}s")
                    time.sleep(int(ban_until))
                    continue

                # 地区 / 合规限制（换 VPN 出口，非代码可修）
                if resp.status_code == 451:
                    body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                    msg = body.get("msg", resp.text[:300])
                    logger.error(
                        "HTTP 451 地区受限: Binance 拒绝当前出口 IP。"
                        "请更换代理节点到其服务条款允许的地区。"
                    )
                    raise BinanceAPIError(451, body.get("code", 0), msg)

                # 可重试的错误
                if resp.status_code in self._retry_on and attempt < self._max_retries:
                    wait = self._backoff * (2 ** attempt)
                    logger.warning(f"HTTP {resp.status_code}，{wait:.1f}s 后重试 ({attempt+1}/{self._max_retries})")
                    time.sleep(wait)
                    continue

                # 不可重试的错误
                body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                raise BinanceAPIError(
                    resp.status_code,
                    body.get("code", -1),
                    body.get("msg", resp.text[:200]))

            except requests.exceptions.Timeout as e:
                last_error = e
                self._error_count += 1
                if attempt < self._max_retries:
                    wait = self._backoff * (2 ** attempt)
                    logger.warning(f"请求超时，{wait:.1f}s 后重试")
                    time.sleep(wait)
                else:
                    raise
            except requests.exceptions.ConnectionError as e:
                last_error = e
                self._error_count += 1
                if attempt < self._max_retries:
                    wait = self._backoff * (2 ** attempt)
                    logger.warning(f"连接错误，{wait:.1f}s 后重试")
                    time.sleep(wait)
                else:
                    raise

        raise last_error or Exception("请求失败，已超过最大重试次数")

    # ----------------------------------------------------------
    # 公共接口
    # ----------------------------------------------------------
    def ping(self) -> bool:
        try:
            self._request("GET", "/api/v3/ping")
            return True
        except Exception:
            return False

    def get_server_time(self) -> int:
        return self._request("GET", "/api/v3/time")["serverTime"]

    def get_exchange_info(self, symbol: Optional[str] = None) -> dict:
        params = {}
        if symbol:
            params["symbol"] = symbol
        return self._request("GET", "/api/v3/exchangeInfo", params)

    def get_ticker_price(self, symbol: str) -> float:
        data = self._request("GET", "/api/v3/ticker/price", {"symbol": symbol})
        return float(data["price"])

    def get_ticker_24h(self, symbol: str) -> dict:
        return self._request("GET", "/api/v3/ticker/24hr", {"symbol": symbol})

    def get_all_tickers(self) -> dict[str, float]:
        data = self._request("GET", "/api/v3/ticker/price")
        return {d["symbol"]: float(d["price"]) for d in data}

    def get_klines(self, symbol: str, interval: str = "1h",
                   limit: int = 1000,
                   start_time: Optional[int] = None,
                   end_time: Optional[int] = None) -> pd.DataFrame:
        """获取K线，单次最多 1000 条"""
        params = {"symbol": symbol, "interval": interval, "limit": min(limit, 1000)}
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        raw = self._request("GET", "/api/v3/klines", params)
        if not raw:
            return pd.DataFrame()

        df = pd.DataFrame(raw, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades_count",
            "taker_buy_base", "taker_buy_quote", "ignore"])
        df = df.drop(columns=["ignore"])
        for c in ["open", "high", "low", "close", "volume",
                   "quote_volume", "taker_buy_base", "taker_buy_quote"]:
            df[c] = df[c].astype(float)
        df["trades_count"] = df["trades_count"].astype(int)
        df["open_time"] = df["open_time"].astype(int)
        df["close_time"] = df["close_time"].astype(int)
        return df

    def get_orderbook(self, symbol: str, limit: int = 20) -> dict:
        return self._request("GET", "/api/v3/depth",
                             {"symbol": symbol, "limit": limit})

    def get_recent_trades(self, symbol: str, limit: int = 500) -> list:
        return self._request("GET", "/api/v3/trades",
                             {"symbol": symbol, "limit": limit})

    # ----------------------------------------------------------
    # 账户接口（签名）
    # ----------------------------------------------------------
    def get_account(self) -> dict:
        return self._request("GET", "/api/v3/account", signed=True)

    def get_balances(self) -> dict[str, dict]:
        """返回 {asset: {"free": float, "locked": float}}"""
        acc = self.get_account()
        return {
            b["asset"]: {"free": float(b["free"]), "locked": float(b["locked"])}
            for b in acc["balances"]
            if float(b["free"]) > 0 or float(b["locked"]) > 0
        }

    # ----------------------------------------------------------
    # 订单接口
    # ----------------------------------------------------------
    def create_order(self, symbol: str, side: str, order_type: str,
                     quantity: float, price: Optional[float] = None,
                     client_order_id: Optional[str] = None,
                     time_in_force: str = "GTC",
                     **kwargs) -> dict:
        params = {
            "symbol": symbol,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": f"{quantity:.8f}".rstrip("0").rstrip("."),
        }
        if client_order_id:
            params["newClientOrderId"] = client_order_id
        if order_type.upper() == "LIMIT":
            params["price"] = f"{price:.8f}".rstrip("0").rstrip(".")
            params["timeInForce"] = time_in_force
        elif order_type.upper() in ("STOP_LOSS_LIMIT", "TAKE_PROFIT_LIMIT"):
            params["price"] = f"{price:.8f}".rstrip("0").rstrip(".")
            params["stopPrice"] = f"{kwargs['stop_price']:.8f}".rstrip("0").rstrip(".")
            params["timeInForce"] = time_in_force
        params.update(kwargs)

        t0 = time.time()
        result = self._request("POST", "/api/v3/order", params, signed=True)
        latency = (time.time() - t0) * 1000
        logger.info(
            f"下单 {side} {quantity:.6f} {symbol} "
            f"type={order_type} price={price} "
            f"-> orderId={result.get('orderId')} "
            f"latency={latency:.0f}ms")
        return result

    def cancel_order(self, symbol: str, order_id: int) -> dict:
        return self._request("DELETE", "/api/v3/order",
                             {"symbol": symbol, "orderId": order_id}, signed=True)

    def cancel_all_orders(self, symbol: str) -> list:
        return self._request("DELETE", "/api/v3/openOrders",
                             {"symbol": symbol}, signed=True)

    def get_order(self, symbol: str, order_id: int) -> dict:
        return self._request("GET", "/api/v3/order",
                             {"symbol": symbol, "orderId": order_id}, signed=True)

    def get_open_orders(self, symbol: Optional[str] = None) -> list:
        params = {"symbol": symbol} if symbol else {}
        return self._request("GET", "/api/v3/openOrders", params, signed=True)

    def get_all_orders(self, symbol: str, limit: int = 500,
                       start_time: Optional[int] = None) -> list:
        params = {"symbol": symbol, "limit": limit}
        if start_time:
            params["startTime"] = start_time
        return self._request("GET", "/api/v3/allOrders", params, signed=True)

    # ----------------------------------------------------------
    # 统计
    # ----------------------------------------------------------
    @property
    def stats(self) -> dict:
        avg_latency = (self._total_latency / self._request_count
                       if self._request_count > 0 else 0)
        return {
            "total_requests": self._request_count,
            "total_errors": self._error_count,
            "avg_latency_ms": round(avg_latency, 1),
            "error_rate": self._error_count / max(self._request_count, 1),
        }
