"""Cloud credentials independent of the agent SDK; tokens stay in memory."""
import datetime
import os
from google.auth.credentials import Credentials


class OperatorCredentials(Credentials):
    def refresh(self, request):
        from ..audit_cloud import gcloud_token
        self.token = gcloud_token()
        self.expiry = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None) + datetime.timedelta(minutes=45)


def cloud_credentials():
    if os.environ.get("K_SERVICE"):
        import google.auth
        return google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])[0]
    return OperatorCredentials()
