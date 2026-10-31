"""Bound the pilot ledger while keeping non-ledger control records smaller."""
import json
from types import SimpleNamespace
import pytest
from research_agents.persistence.control_store import GcsControlStore,record_limit

@pytest.mark.parametrize('key,limit',[('ledger.json',2_000_000),('tasks/run-fixture.json',1_000_000)])
def test_record_at_limit_roundtrips_and_one_more_byte_fails_before_upload(key,limit):
 assert record_limit(key)==limit
 value={'payload':'x'*(limit-len(b'{"payload":""}'))};data=json.dumps(value,separators=(',',':')).encode()
 assert len(data)==limit
 uploads=[]
 blob=SimpleNamespace(size=len(data),generation=1,reload=lambda **kw:None,download_as_bytes=lambda **kw:data,upload_from_string=lambda data,**kw:uploads.append(data))
 store=object.__new__(GcsControlStore);store.blob=lambda key:blob
 assert store.read(key)==(value,'1');store.write(key,value,'1');assert uploads==[data]
 value['payload']+='x'
 with pytest.raises(ValueError,match='exceeds limit'):store.write(key,value,'1')
 assert len(uploads)==1

@pytest.mark.parametrize('key',['ledger.json','tasks/run-fixture.json'])
@pytest.mark.parametrize('metadata_oversize',[True,False])
def test_oversized_metadata_or_response_rejected(key,metadata_oversize):
 limit=record_limit(key);downloads=[]
 def download(**kw):downloads.append(1);return b'x'*(limit+1)
 blob=SimpleNamespace(size=limit+1 if metadata_oversize else 0,generation=1,reload=lambda **kw:None,download_as_bytes=download)
 store=object.__new__(GcsControlStore);store.blob=lambda key:blob
 with pytest.raises(ValueError,match='exceeds limit'):store.read(key)
 assert bool(downloads) is not metadata_oversize
