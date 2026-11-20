"""security_utils 的脱敏行为测试。"""

from utils.security_utils import redact_secrets, safe_endpoint


def test_redacts_sk_key():
    out = redact_secrets("auth failed for sk-abcdef1234567890")
    assert "sk-abcdef1234567890" not in out
    assert "sk-***" in out


def test_redacts_exact_secret():
    secret = "custom-secret-value-123"
    out = redact_secrets(f"bad key {secret}", secret)
    assert secret not in out
    assert "***" in out


def test_redacts_bearer_and_api_key():
    out = redact_secrets(
        "Authorization: Bearer abcdefgh12345678; api_key=secretvalue123"
    )
    assert "abcdefgh12345678" not in out
    assert "secretvalue123" not in out


def test_redacts_url_userinfo():
    out = redact_secrets("POST https://user:pass@proxy.example.com/v1 failed")
    assert "user:pass" not in out
    assert "proxy.example.com" in out


def test_safe_endpoint_strips_credentials_path_and_query():
    endpoint = safe_endpoint(
        "https://user:pass@proxy.example.com:8443/v1?token=abc#fragment"
    )
    assert endpoint == "https://proxy.example.com:8443"


def test_safe_endpoint_keeps_default_endpoint():
    assert safe_endpoint("https://api.deepseek.com") == "https://api.deepseek.com"


def test_safe_endpoint_invalid():
    assert safe_endpoint("not a url") == "***"
