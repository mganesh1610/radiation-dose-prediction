"""Public preview failures preserve existing directories and never install partial results."""
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_public_preview as preview
from public_site_assets import FILES, GUEST_EXTRAS, GUEST_VARIANTS, RECORDS, share_template


def test_preview_builds_both_roles_from_their_own_templates_and_assets():
    root = preview.ROOT
    guest = share_template((root / FILES['index.html']).read_text(), (root / 'app/static/index.html').read_text())
    assert 'id="guest-training"' in guest and 'id="reviewer-results"' in guest
    assert len(RECORDS) == len(set(RECORDS)) == 30
    for name in GUEST_VARIANTS:
        assert GUEST_EXTRAS['static/' + name] == 'app/public/' + name
        assert (root / 'app/static' / name).is_file() and (root / 'app/public' / name).is_file()


def test_refuses_a_changed_publication_and_removes_partial_downloads(tmp_path):
    with patch.object(preview, 'download', return_value=b'{}'), pytest.raises(ValueError, match='Publication'):
        preview.prepare(tmp_path / 'site')
    assert list(tmp_path.iterdir()) == []


def test_preserves_existing_preview(tmp_path):
    existing = tmp_path / 'index.html'
    existing.write_text('previous build')
    with patch.object(preview, 'download') as fetch, pytest.raises(FileExistsError):
        preview.prepare(tmp_path)
    fetch.assert_not_called()
    assert existing.read_text() == 'previous build'
