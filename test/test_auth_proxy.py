import http.server
import json
import os
import threading
import unittest
import urllib.request
from unittest.mock import patch, MagicMock

from xpu_shell.auth_proxy import (
  is_allowed_client_ip,
  RWUrlFilter,
  BearerTokenProvider,
  HeaderAuthProvider,
  SigV4AuthProvider,
  detect_proxy_routes,
  create_auth_proxy,
  XpuAuthProxy,
  ProxyRoute
)


class TestAuthProxy(unittest.TestCase):

  def test_is_allowed_client_ip(self):
    self.assertTrue(is_allowed_client_ip("127.0.0.1"))
    self.assertTrue(is_allowed_client_ip("::1"))
    self.assertTrue(is_allowed_client_ip("10.0.0.5"))
    self.assertTrue(is_allowed_client_ip("192.168.1.100"))
    self.assertTrue(is_allowed_client_ip("172.17.0.2"))
    self.assertFalse(is_allowed_client_ip("8.8.8.8"))
    self.assertFalse(is_allowed_client_ip("invalid-ip"))

  def test_rw_url_filter(self):
    # Empty filter denies writes
    empty_filter = RWUrlFilter(None)
    self.assertFalse(empty_filter.is_write_allowed("/s3", "/my-bucket/checkpoint.pt"))
    self.assertFalse(empty_filter.has_gcs_rules())

    # Wildcard filter
    wild_filter = RWUrlFilter("*")
    self.assertTrue(wild_filter.is_write_allowed("/s3", "/my-bucket/checkpoint.pt"))
    self.assertTrue(wild_filter.has_gcs_rules())

    # Specific bucket pattern
    bucket_filter = RWUrlFilter("s3://train-bucket/checkpoints/*;gs://data-bucket/*")
    self.assertTrue(bucket_filter.is_write_allowed("/s3", "/train-bucket/checkpoints/epoch-1.pt"))
    self.assertFalse(bucket_filter.is_write_allowed("/s3", "/other-bucket/file.txt"))
    self.assertTrue(bucket_filter.has_gcs_rules())

  def test_bearer_token_provider(self):
    provider = BearerTokenProvider("hf_test_12345")
    headers = provider.get_headers("GET", "https://huggingface.co/models")
    self.assertEqual(headers.get("Authorization"), "Bearer hf_test_12345")
    cleaned = provider.clean_headers({"Authorization": "OldAuth", "User-Agent": "test"})
    self.assertNotIn("Authorization", cleaned)
    self.assertIn("User-Agent", cleaned)
    self.assertIn("hf_test_12345", provider.get_secrets())

  def test_bearer_token_provider_from_env(self):
    with patch.dict(os.environ, {"MY_HF_TOKEN": "hf_secret_token"}):
      provider = BearerTokenProvider("MY_HF_TOKEN")
      headers = provider.get_headers("GET", "https://huggingface.co")
      self.assertEqual(headers.get("Authorization"), "Bearer hf_secret_token")

  def test_header_auth_provider(self):
    provider = HeaderAuthProvider(
      headers=lambda: {"X-Custom": "val123"},
      secrets=lambda: ["val123"]
    )
    headers = provider.get_headers("GET", "https://api.test.com")
    self.assertEqual(headers.get("X-Custom"), "val123")
    self.assertIn("val123", provider.get_secrets())

  def test_sigv4_auth_provider(self):
    provider = SigV4AuthProvider(
      access_key="AKIA1111111111111111",
      secret_key="secretkey1234567890",
      service="s3",
      region="us-east-1"
    )
    headers = provider.get_headers("GET", "https://my-bucket.s3.us-east-1.amazonaws.com/test.txt")
    self.assertIn("Authorization", headers)
    self.assertTrue(headers["Authorization"].startswith("AWS4-HMAC-SHA256"))
    self.assertTrue(any(k.lower() == "x-amz-date" for k in headers))
    self.assertIn("AKIA1111111111111111", provider.get_secrets())

  def test_detect_proxy_routes(self):
    mock_env = {
      "HF_TOKEN": "hf_abc123",
      "WANDB_API_KEY": "wandb_key_999",
      "AWS_ACCESS_KEY_ID": "AKIA999",
      "AWS_SECRET_ACCESS_KEY": "secret999",
      "CIVITAI_API_TOKEN": "civitai_tok_123"
    }
    routes = detect_proxy_routes(env=mock_env)
    prefixes = [r.prefix for r in routes]
    self.assertIn("/hf", prefixes)
    self.assertIn("/wandb", prefixes)
    self.assertIn("/s3", prefixes)
    self.assertIn("/civitai", prefixes)

    # Check remote env mapping
    proxy = XpuAuthProxy(routes=routes)
    remote_env = proxy.get_remote_env(remote_host="127.0.0.1", remote_port=8080)
    self.assertEqual(remote_env.get("HF_ENDPOINT"), "http://127.0.0.1:8080/hf")
    self.assertEqual(remote_env.get("WANDB_BASE_URL"), "http://127.0.0.1:8080/wandb")
    self.assertEqual(remote_env.get("AWS_ENDPOINT_URL"), "http://127.0.0.1:8080/s3")
    self.assertEqual(remote_env.get("CIVITAI_API_URL"), "http://127.0.0.1:8080/civitai")

  def test_proxy_healthcheck_endpoint(self):
    proxy = XpuAuthProxy(routes=[])
    proxy.start()
    try:
      url = f"http://127.0.0.1:{proxy.port}/__xpu_status__"
      with urllib.request.urlopen(url, timeout=3) as resp:
        self.assertEqual(resp.status, 200)
        data = json.loads(resp.read().decode())
        self.assertEqual(data.get("status"), "ok")
    finally:
      proxy.stop()

  def test_proxy_request_routing_and_auth(self):
    # Setup mock upstream target server
    received_requests = []

    class UpstreamHandler(http.server.BaseHTTPRequestHandler):
      def do_GET(self):
        received_requests.append({
          "path": self.path,
          "auth": self.headers.get("Authorization")
        })
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"upstream hello")

      def log_message(self, format, *args):
        pass

    upstream_server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
    upstream_port = upstream_server.server_address[1]
    upstream_thread = threading.Thread(target=upstream_server.serve_forever, daemon=True)
    upstream_thread.start()

    try:
      route = ProxyRoute(
        prefix="/test",
        target_base_url=f"http://127.0.0.1:{upstream_port}",
        auth_provider=BearerTokenProvider("my-test-bearer-token"),
        allowed_methods=frozenset({"GET"})
      )
      proxy = XpuAuthProxy(routes=[route])
      proxy.start()

      try:
        proxy_url = f"http://127.0.0.1:{proxy.port}/test/v1/resource"
        with urllib.request.urlopen(proxy_url, timeout=3) as resp:
          self.assertEqual(resp.status, 200)
          self.assertEqual(resp.read(), b"upstream hello")

        self.assertEqual(len(received_requests), 1)
        self.assertEqual(received_requests[0]["path"], "/v1/resource")
        self.assertEqual(received_requests[0]["auth"], "Bearer my-test-bearer-token")

        # Test method not allowed
        req_post = urllib.request.Request(proxy_url, data=b"data", method="POST")
        with self.assertRaises(urllib.error.HTTPError) as ctx:
          urllib.request.urlopen(req_post, timeout=3)
        self.assertEqual(ctx.exception.code, 403)
      finally:
        proxy.stop()
    finally:
      upstream_server.shutdown()
      upstream_server.server_close()


if __name__ == "__main__":
  unittest.main()
