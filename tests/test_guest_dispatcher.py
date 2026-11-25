import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from test_training_integration import MemoryStore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import guest_dispatcher as guest
from research_agents.persistence.control_store import MemoryControlStore, QuotaExceeded
from research_agents.policies.demo_quota import START, DAY, END
from job_queue import Queue, job_name


class GuestTrainingTests(unittest.TestCase):
    def test_demo_provider_ledger_reserves_seats_and_preserves_the_team_pool(self):
        store = MemoryControlStore()
        config = {'workspace':'test', 'max_chats':260, 'max_runs':280}
        people = ['guest_'+format(i,'064x') for i in range(51)]
        for actor in people[:50]:
            ledger = guest.guest_ledger(store, actor, START, config)
            self.assertEqual(ledger.max_runs,20)
        with self.assertRaises(QuotaExceeded):
            guest.guest_ledger(store, people[50], START, config)
        actor = people[0]
        ledger = guest.guest_ledger(store,actor,START,config)
        for number in range(20):
            if number % 10 == 0:
                chat = ledger.create_chat(actor,str(number),'test')['chat_id']
                ledger.ready(actor,chat)
            run = ledger.admit(actor,chat,str(number),'A research question')
            ledger.finish(actor,run['run_id'],run['fence'],'completed')
        with self.assertRaises(QuotaExceeded):
            ledger.admit(actor,chat,'extra','Another question')
        self.assertEqual(guest.guest_ledger(store,actor,START+DAY,config).read()['runs'],{})
        self.assertEqual(guest.guest_ledger(store,actor,END,config).record_key,'ledger.json')
        self.assertIsNone(store.read('ledger.json')[0])

    def test_training_reuses_shared_admission_and_an_active_cloud_lease(self):
        work = {'id':str(uuid.uuid4()),'created':1704067200,'input':{'method':'gradient_boosting','holdout':'GSE8917'}}
        store = MemoryStore()
        owner = 'guest_'+'a'*64

        def execute(command, **options):
            self.assertEqual(command[:-1],[sys.executable,str(ROOT/'scripts/training_worker.py'),'run','--job-id'])
            Queue(store).claim(command[-1])

        with patch.object(guest,'Cloud',return_value=store), patch.object(guest.subprocess,'run',side_effect=execute) as runner:
            self.assertEqual(guest.training(work,owner)[0],'queued')
            self.assertEqual(guest.training(work,owner)[0],'queued')
            with self.assertRaises(ValueError):
                guest.training(work,'guest_'+'b'*64)
            self.assertEqual(runner.call_count,1)
        state,_ = guest.DispatchControl(store,'2024-01').read()
        self.assertEqual(len(state['jobs']),1)
        job_id, entry = next(iter(state['jobs'].items()))
        self.assertEqual(entry['owner'],owner)
        self.assertEqual(entry['request']['dataset'],guest.COMBINED_DATASET)
        self.assertEqual(store.read_json(job_name(job_id))[0]['attempt'],1)

    def test_invalid_preset_does_not_touch_the_cloud(self):
        with patch.object(guest,'Cloud') as cloud, self.assertRaises(ValueError):
            guest.training({'input':{'method':'custom','holdout':'GSE8917'}},'guest_owner')
        cloud.assert_not_called()
