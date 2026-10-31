"""Saved-state truth across user boundaries, expiry and mutation lifecycle."""
import asyncio
import json
import time
from test_preferences import setup, change


def state(text):
    return json.loads(text.splitlines()[1])


def test_authoritative_state_empty_active_updated_deleted():
    async def run():
        service,provider,store,enabled,alice,bob=setup()
        initial=state(await service.instruction(alice))
        assert all(v == {'status':'unset','value':None} for v in initial.values())
        await service.save(alice,'display_format',change('bullets'))
        a=state(await service.instruction(alice));b=state(await service.instruction(bob))
        assert a['display_format']=={'status':'active','value':'bullets'}
        assert b['display_format']=={'status':'unset','value':None}
        await service.save(alice,'display_format',change('prose',1))
        assert state(await service.instruction(alice))['display_format']['value']=='prose'
        await service.delete(alice,'display_format',2)
        final=state(await service.instruction(alice))
        assert final['display_format']=={'status':'deleted','value':None}
        assert not provider.values
    asyncio.run(run())


def test_expired_pending_and_provider_failure_not_misrepresented_as_active():
    async def run():
        s,p,store,enabled,a,b=setup()
        await s.save(a,'response_style',change('detailed'))
        s.clock=lambda:time.time()+31*86400
        assert state(await s.instruction(a))['response_style']=={'status':'expired','value':None}
        s.clock=time.time
        original=p.get
        async def fail(*args):raise RuntimeError('provider unavailable')
        p.get=fail
        try:
            await s.instruction(a)
        except RuntimeError:pass
        else:raise AssertionError('An unavailable provider must not look like empty preferences')
        p.get=original
        _,path,_=await s._read(a,'response_style')
        value,generation=store.read(path);value['status']='pending';store.write(path,value,generation)
        assert state(await s.instruction(a))['response_style']=={'status':'pending','value':None}
    asyncio.run(run())


def test_active_state_does_not_offer_an_inactive_example_to_copy():
    async def run():
        service,provider,store,enabled,alice,bob=setup()
        await service.save(alice,'response_style',change('detailed'))
        await service.save(alice,'display_format',change('prose'))
        instruction=await service.instruction(alice)
        assert 'Active saved preferences: response_style=detailed; display_format=prose.' in instruction
        assert 'No saved presentation preferences are currently active.' not in instruction
    asyncio.run(run())
