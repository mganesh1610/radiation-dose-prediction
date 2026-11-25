import time

import pytest
from test_chat import create, headers, key, setup
from research_agents.persistence.control_store import MemoryControlStore, QuotaExceeded
from research_agents.persistence.ledger import DailyTeamLimit, Ledger
from research_agents.policies.demo_quota import DAY, END, POLICY, START


def test_team_daily_runs_reopen_at_midnight_and_preserve_the_original_pool():
    clock = [START - 1]
    ledger = Ledger(MemoryControlStore(), 'test', max_runs=1, max_chats=1,
                    clock=lambda: clock[0], team_demo=True)

    def chat(owner, key):
        item = ledger.create_chat(owner, key, 'release')
        ledger.ready(owner, item['chat_id'])
        return item['chat_id']

    def turn(owner, chat_id, key):
        run = ledger.admit(owner, chat_id, key, 'Compare the models')
        ledger.finish(owner, run['run_id'], run['fence'], 'completed')
        return run['run_id']

    old_chat = chat('alice', 'original')
    clock[0] = START
    assert ledger.team_quota()['remaining'] == 100
    for number in range(POLICY['team_messages_per_day']):
        owner = 'alice' if number < 50 else 'bob'
        if number % 10 == 0:
            thread = old_chat if number == 0 else chat(owner, str(number))
        run = turn(owner, thread, str(number))
    assert ledger.admit(owner, thread, str(number), 'Compare the models')['run_id'] == run
    assert ledger.team_quota()['remaining'] == 0
    another = chat(owner, 'tomorrow')
    with pytest.raises(DailyTeamLimit):
        turn(owner, another, 'over')
    clock[0] += DAY
    assert ledger.team_quota()['remaining'] == 100
    turn(owner, another, 'on-new-day')
    assert ledger.team_quota()['remaining'] == 99
    clock[0] = END
    assert ledger.team_quota() is None
    turn(owner, another, 'original-allowance')
    with pytest.raises(QuotaExceeded):
        turn(owner, another, 'over-original')


def test_signed_in_quota_api_announces_the_window_and_blocks_the_101st_message(monkeypatch):
    client, service, calls, _ = setup()
    now = [START - 1]
    monkeypatch.setattr(time, 'time', lambda: now[0])
    service.ledger.team_demo = True
    service.ledger.clock = time.time
    # Fixture identities and quota records must use the same simulated date.
    service.access.clock = time.time
    quota_url = '/api/copilot/quota'
    assert client.get(quota_url).status_code == 403
    upcoming = client.get(quota_url, headers=headers()).json()['quota']['upcoming']
    assert upcoming == {'starts_at': POLICY['starts_at'], 'ends_at': POLICY['ends_at'], 'daily_messages': 100}

    now[0] = START
    for index in range(100):
        owner = 'alice' if index < 50 else 'bob'
        if index % 10 == 0:
            thread = create(client, owner)['chat_id']
        response = client.post(f'/api/copilot/chats/{thread}/messages', headers=headers(owner),
                               json={'request_id': key(), 'text': 'Compare the published models'})
        assert response.status_code == 200 and response.json()['status'] == 'completed'
    status = client.get(quota_url, headers=headers('alice')).json()['quota']
    assert status['used'] == 100 and status['remaining'] == 0 and status['reason'] == 'team_quota'
    thread = create(client, 'alice')['chat_id']
    response = client.post(f'/api/copilot/chats/{thread}/messages', headers=headers('alice'),
                           json={'request_id': key(), 'text': 'Compare the published models'})
    assert response.status_code == 429
    assert response.json()['code'] == 'team_quota' and 'midnight Phoenix time' in response.json()['error']
    assert len(calls) == 100

    now[0] += DAY
    assert client.get(quota_url, headers=headers('bob')).json()['quota']['remaining'] == 100
    now[0] = END
    assert client.get(quota_url, headers=headers('bob')).json()['quota'] is None
