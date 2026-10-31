"""Protected finite allowances still enforce the ledger boundary across instances."""
import copy
import pytest
from research_agents.policies.runtime import ChatConfig
from research_agents.persistence.control_store import MemoryControlStore,QuotaExceeded
from research_agents.persistence.ledger import Ledger


def config(**kwargs):
    return ChatConfig('workspace','core_test','copilot','https://research.test','https://evidence.test',**kwargs)


def test_default_remains_twenty():
    assert (config().max_runs,config().max_chats)==(20,20)


@pytest.mark.parametrize('bad',[0,-1,281,True,20.5,'47',None])
def test_invalid_allowance_fails_closed(bad):
    with pytest.raises(ValueError):config(max_runs=bad)
    with pytest.raises(ValueError):config(max_chats=bad)


@pytest.mark.parametrize('base,limit',[(139,149),(149,157),(157,163),(163,168),(168,180),(180,188),(188,189),(189,199),(199,200),(200,201),(201,202),(202,203),(203,211),(211,212),(212,220),(220,240),(240,244),(244,252),(252,256),(256,280)])
def test_expanded_limit_preserves_accounting_and_idempotency(base,limit):
    c=config(max_runs=base,max_chats=base);store=MemoryControlStore()
    ledger=Ledger(store,c.workspace,max_runs=c.max_runs,max_chats=c.max_chats)
    for i in range(base):
        chat=ledger.create_chat('owner',str(i),'core_test');ledger.ready('owner',chat['chat_id'])
        turn=ledger.admit('owner',chat['chat_id'],str(i),'question')
        ledger.finish('owner',turn['run_id'],turn['fence'],'completed')
    before=copy.deepcopy(ledger.read())
    with pytest.raises(QuotaExceeded):ledger.create_chat('reviewer','first','core_test')
    c=config(max_runs=limit,max_chats=limit)
    ledger=Ledger(store,c.workspace,max_runs=c.max_runs,max_chats=c.max_chats)
    for i in range(base,limit):
        chat=ledger.create_chat('reviewer',str(i),'core_test');ledger.ready('reviewer',chat['chat_id'])
        turn=ledger.admit('reviewer',chat['chat_id'],str(i),'question')
        ledger.finish('reviewer',turn['run_id'],turn['fence'],'completed')
    restarted=Ledger(store,c.workspace,max_runs=c.max_runs,max_chats=c.max_chats)
    assert len(restarted.read()['runs'])==limit and len(restarted.read()['chats'])==limit
    assert all(restarted.read()['runs'][k]==v for k,v in before['runs'].items())
    assert all(restarted.read()['chats'][k]==v for k,v in before['chats'].items())
    assert restarted.admit('reviewer',chat['chat_id'],str(limit-1),'question')['run_id']==turn['run_id']
    with pytest.raises(FileNotFoundError):restarted.run('owner',turn['run_id'])
    with pytest.raises(QuotaExceeded):restarted.create_chat('reviewer','overflow','core_test')
    with pytest.raises(QuotaExceeded):restarted.admit('reviewer',chat['chat_id'],'overflow','question')
