"""Small fabricated failure fixtures, separate from real-cloud benchmark data."""
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import threading
import time
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
from pydantic import ValidationError

from research_agents.audit_cloud import CloudScope
from research_agents.policies.access import AccessPolicy, Grant, Principal
from research_agents.tools.query_sql import ApprovedRelease
from research_agents.tools.matrix_cache import MatrixCache, CacheBusy
from research_agents.tools.expression_reader import (MatrixAdmission, ExpressionReader, decode_matrix,
    IntegrityError, CloudMatrixSource)

SCOPE = CloudScope("sample-project", "research", "sample-bucket", "us-west1")
CORE = ApprovedRelease("core_fixture", "a"*64,
    (("GSM1","GSE1","donor1"),("GSM2","GSE1","donor1"),("GSM3","GSE2","donor1")),
    (("samples","s1"),("sample_annotations","a1")))
PRINCIPAL = Principal("usr_test",200)


def arrays():
    return {"values": np.array([[1.,np.nan,3.],[4.,5.,6.],[7.,8.,9.]],dtype=np.float64),
        "detected": np.array([[True,False,False],[True,True,True],[True,True,True]]),
        "sample_ids":np.array(["GSM1","GSM2","GSM3"]), "probe_ids":np.array(["probe1","probe2","probe3"]),
        "gene_keys":np.array(["7157","7157","101"]), "study_ids":np.array(["GSE1","GSE1","GSE2"]),
        "split_group_ids":np.array(["donor1"]*3), "y":np.array([0.,2.,0.])}


def payload(a=None):
    stream=io.BytesIO();np.savez_compressed(stream,**(arrays() if a is None else a));return stream.getvalue()


def admission(raw=None):
    raw=payload() if raw is None else raw
    return MatrixAdmission(SCOPE,CORE,"core_fixture_linear","training/v1/releases/core_fixture/core.npz",
        "123",hashlib.sha256(raw).hexdigest(),len(raw),(("probe1","7157"),("probe2","7157"),("probe3","101")),
        "b"*64,(("samples","s1"),("features","f1"),("expression_sets","e1")))


def plan(**changes):
    return dict({"version":1,"release_id":"core_fixture","expression_set_id":"core_fixture_linear",
        "sample_ids":["GSM2","GSM1"],"gene_ids":["7157"],"gene_namespace":"NCBI_GeneID","aggregation":"none"},**changes)


def setup():
    raw=payload();adm=admission(raw); state={"grant":Grant(PRINCIPAL.user_id,"workspace",frozenset({CORE.release_id}),"r1"),"admitted":adm,
       "loads":0,"checks":0,"bad_source":False,"after_load":lambda:None,"delay":0}
    class Source:
        def check(self, value):
            state["checks"]+=1
            if state["bad_source"]:raise IntegrityError("Cloud revoked")
        def load(self, value):
            state["loads"]+=1;time.sleep(state["delay"]);state["after_load"]();return raw
        def stats(self):return {"cloud_requests":0,"downloaded_response_bytes":0,"matrix_downloads":0}
    access=AccessPolicy(lambda u,w:state["grant"] if u==PRINCIPAL.user_id and w=="workspace" else None,clock=lambda:100)
    cache=MatrixCache()
    reader=ExpressionReader(cache,access,lambda w,r,e:state["admitted"] if w=="workspace" else None,lambda a:Source())
    return reader,cache,state


def test_exact_axis_order_missing_detection_and_multi_probe_cache():
    reader,cache,state=setup()
    first=reader.run(PRINCIPAL,"workspace",plan());second=reader.run(PRINCIPAL,"workspace",plan())
    assert first["answer"]==second["answer"]
    assert first["answer"]["values"]==[[None,1.],[5.,4.]]
    assert first["answer"]["detected"]==[[False,True],[True,True]]
    assert first["answer"]["missing_values"]==1
    assert [f["probe_id"] for f in first["answer"]["features"]]==["probe1","probe2"]
    assert first["receipt"]["cache"]["hit"] is False and second["receipt"]["cache"]["hit"] is True
    assert state["loads"]==1 and state["checks"]==4
    assert cache.hits==1 and cache.misses==1
    json.dumps(second,allow_nan=False)


@pytest.mark.parametrize("changes", [
    {"sample_ids":[]},{"sample_ids":["GSM1"]*51},{"sample_ids":["GSM1","GSM1"]},
    {"gene_ids":["7157","7157"]},{"gene_ids":["TP53"]},{"gene_namespace":"SYMBOL"},
    {"aggregation":"mean"},{"object_name":"gs://attacker"},{"workspace":"other"},
    {"user_id":"admin"},{"scale":"log2"},{"gene_ids":["0"]},{"gene_ids":[7157]}])
def test_bad_requests_never_access_source(changes):
    reader,_,state=setup()
    with pytest.raises(ValidationError):reader.run(PRINCIPAL,"workspace",plan(**changes))
    assert state["loads"]==state["checks"]==0


@pytest.mark.parametrize("changes", [{"sample_ids":["GSM999"]},{"gene_ids":["99999"]},
    {"expression_set_id":"not_admitted"}])
def test_unknown_selectors_fail_without_partial_result(changes):
    reader,_,state=setup()
    with pytest.raises((ValueError,PermissionError)):reader.run(PRINCIPAL,"workspace",plan(**changes))
    assert state["checks"]==0


@pytest.mark.parametrize("kind", ["principal","workspace","grant","admission","source"])
def test_cache_hit_never_bypasses_current_authority(kind):
    reader,cache,state=setup();reader.run(PRINCIPAL,"workspace",plan())
    who=PRINCIPAL;workspace="workspace"
    if kind=="principal":who=replace(who,user_id="stranger")
    if kind=="workspace":workspace="other"
    if kind=="grant":state["grant"]=None
    if kind=="admission":state["admitted"]=None
    if kind=="source":state["bad_source"]=True
    with pytest.raises((PermissionError,IntegrityError)):reader.run(who,workspace,plan())
    assert state["loads"]==1 and cache.hits==0


def test_revoked_during_download_returns_nothing_and_discards_cache():
    reader,cache,state=setup();state["after_load"]=lambda:state.update(grant=None)
    with pytest.raises(PermissionError):reader.run(PRINCIPAL,"workspace",plan())
    state["grant"]=Grant(PRINCIPAL.user_id,"workspace",frozenset({CORE.release_id}),"r2")
    state["after_load"]=lambda:None
    assert reader.run(PRINCIPAL,"workspace",plan())["receipt"]["cache"]["hit"] is False


@pytest.mark.parametrize("kind", ["sha","size","axis","gene","mask","negative","infinite","dtype","object","duplicates","shape"])
def test_matrix_integrity_and_unsafe_arrays(kind):
    a=arrays()
    if kind=="axis":a["sample_ids"][0]="GSM9"
    if kind=="gene":a["gene_keys"][0]="9999"
    if kind=="mask":a["detected"]=a["detected"].astype(np.int8)
    if kind=="negative":a["values"][0,0]=-1
    if kind=="infinite":a["values"][0,0]=np.inf
    if kind=="dtype":a["values"]=a["values"].astype(np.complex128)
    if kind=="object":a["sample_ids"]=a["sample_ids"].astype(object)
    if kind=="duplicates":a["probe_ids"][0]=a["probe_ids"][1]
    if kind=="shape":a["values"]=np.ones((2,3))
    raw=payload(a);adm=admission(raw)
    if kind=="sha":adm=replace(adm,sha256="c"*64)
    if kind=="size":adm=replace(adm,compressed_bytes=len(raw)+1)
    with pytest.raises(IntegrityError):decode_matrix(raw,adm)


def test_npy_declared_allocation_is_checked_before_numpy_load(monkeypatch):
    original=payload();stream=io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(original)) as src,zipfile.ZipFile(stream,"w") as dst:
        for name in src.namelist():
            data=src.read(name)
            if name=="values.npy":
                fake=io.BytesIO();np.lib.format.write_array_header_1_0(fake,{"descr":"<f8","fortran_order":False,"shape":(10**9,10**9)})
                data=fake.getvalue()
            dst.writestr(name,data)
    raw=stream.getvalue();called=[]
    monkeypatch.setattr(np,"load",lambda *a,**k:called.append(True))
    with pytest.raises(IntegrityError):decode_matrix(raw,admission(raw))
    assert not called


def test_cache_memory_ttl_eviction_and_version_keys():
    now=[0.];cache=MatrixCache(max_bytes=100,max_entries=2,ttl_seconds=10,clock=lambda:now[0])
    calls=[]
    def load():calls.append(1);return SimpleNamespace(resident_bytes=60)
    def use(x):return "copy"
    key=("w","r","v1")
    assert cache.read(key,60,load,use)[1]["hit"] is False
    assert cache.read(key,60,load,use)[1]["hit"] is True
    now[0]=11
    assert cache.read(key,60,load,use)[1]["hit"] is False
    assert cache.read(("w","r","v2"),60,load,use)[1]["accounted_resident_bytes"]==60
    assert cache.evictions==2 and len(calls)==3
    with pytest.raises(ValueError):cache.read(key,101,load,use)
    with pytest.raises(ValueError):cache.read(key,60,lambda:SimpleNamespace(resident_bytes=61),use)
    assert MatrixCache(max_bytes=100).read(key,60,load,use)[1]["hit"] is False


def test_one_load_for_concurrent_identical_requests_and_readonly_storage():
    reader,cache,state=setup();state["delay"]=0.03
    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs=list(pool.map(lambda _:reader.run(PRINCIPAL,"workspace",plan()),range(2)))
    assert state["loads"]==1 and sorted(o["receipt"]["cache"]["hit"] for o in outputs)==[False,True]
    frame=decode_matrix(payload(),admission())
    with pytest.raises(ValueError):frame.values[0,0]=999
    with pytest.raises(TypeError):frame.probe_index["extra"]=3


def test_concurrency_and_cache_wait_have_bounds():
    reader,cache,_=setup();reader.slots.acquire();reader.slots.acquire()
    try:
        with pytest.raises(RuntimeError,match="busy"):reader.run(PRINCIPAL,"workspace",plan())
    finally:reader.slots.release();reader.slots.release()
    cache=MatrixCache(lock_timeout=.001);cache._lock.acquire()
    try:
        with pytest.raises(CacheBusy):cache.read(("w","r","v"),10,lambda:None,lambda x:None)
    finally:cache._lock.release()


def test_resolved_probe_limit_is_checked_before_cloud():
    reader,_,state=setup()
    extra=tuple(("p"+str(i),"7157") for i in range(201))
    state["admitted"]=replace(state["admitted"],features=extra)
    with pytest.raises(ValueError,match="bounds"):reader.run(PRINCIPAL,"workspace",plan())
    assert state["checks"]==0


def test_cloud_registry_and_object_checks_fail_closed():
    source=CloudMatrixSource(SCOPE,token_provider=lambda:"unused")
    source.cloud=SimpleNamespace(scope=SCOPE,table_metadata=lambda t:{"etag":"wrong"})
    with pytest.raises(IntegrityError):source.check(admission())
