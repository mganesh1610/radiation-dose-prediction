"""Fail-closed boundaries for the portable submission configuration."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from cloud_io import Cloud
from app.server import verify_identity


class ConfigurationTests(unittest.TestCase):
    def test_missing_cloud_destination_cannot_retrieve_credentials_or_send_request(self):
        with patch('cloud_io.PROJECT', ''), patch('cloud_io.subprocess.run') as token, patch('cloud_io.urllib.request.urlopen') as http:
            with self.assertRaises(RuntimeError):
                Cloud().request('https://storage.googleapis.com/storage/v1/b/example')
            token.assert_not_called()
            http.assert_not_called()

    def test_missing_audience_does_not_accept_a_mock_verified_identity(self):
        with patch('app.server.AUDIENCE', ''), patch('google.oauth2.id_token.verify_token') as verify:
            with self.assertRaises(PermissionError):
                verify_identity({'X-Goog-IAP-JWT-Assertion':'not-a-real-token'})
            verify.assert_not_called()

    def test_explicit_audience_is_forwarded_to_signature_verification(self):
        audience = '/projects/123/locations/us-west1/services/example'
        with patch('app.server.AUDIENCE', audience), patch.dict('os.environ', {'ALLOWED_EMAILS':'reviewer@example.test'}), patch('google.oauth2.id_token.verify_token', return_value={'iss':'https://cloud.google.com/iap','sub':'1','email':'reviewer@example.test'}) as verify:
            self.assertEqual(verify_identity({'X-Goog-IAP-JWT-Assertion':'not-a-real-token'}), 'reviewer@example.test')
            self.assertEqual(verify.call_args.kwargs['audience'], audience)


if __name__ == '__main__':
    unittest.main()
