import pytest
from google.api_core.exceptions import NotFound,PreconditionFailed
from research_agents.persistence.control_store import GcsControlStore,StorageUnavailable


@pytest.mark.parametrize('error',[NotFound('replaced generation'),PreconditionFailed('changed generation')])
def test_replaced_generation_never_appears_as_empty_ledger(error):
    class Blob:
        size=20;generation=2
        def reload(self,**kwargs):pass
        def download_as_bytes(self,**kwargs):
            if not attempts:attempts.append(1);raise error
            return b'{"runs":{"r":1}}'
    attempts=[];store=object.__new__(GcsControlStore);store.blob=lambda key:Blob()
    value,generation=store.read('ledger.json')
    assert value['runs']=={'r':1} and generation=='2'


def test_repeated_read_conflicts_fail_closed():
    class Blob:
        size=20;generation=2
        def reload(self,**kwargs):pass
        def download_as_bytes(self,**kwargs):raise NotFound('replaced')
    store=object.__new__(GcsControlStore);store.blob=lambda key:Blob()
    with pytest.raises(StorageUnavailable):store.read('ledger.json')


def test_missing_metadata_is_empty_record():
    class Blob:
        def reload(self,**kwargs):raise NotFound('absent')
    store=object.__new__(GcsControlStore);store.blob=lambda key:Blob()
    assert store.read('ledger.json')==(None,'0')
