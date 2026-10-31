"""Run-owned persistent A2A task records for the pinned protobuf SDK."""
import asyncio
from google.protobuf.json_format import MessageToDict,ParseDict
from a2a.server.tasks import TaskStore
from a2a.types import Task,ListTasksResponse
from ..policies.run_scope import current_run
from .control_store import mutate,Conflict


class ScopedTaskStore(TaskStore):
    def __init__(self, store, ledger):
        self.store,self.ledger=store,ledger

    def key(self):
        s=current_run.get()
        self.ledger.check(s.principal.user_id,s.run_id,s.fence)
        return "tasks/"+s.run_id+".json"

    async def save(self, task, context=None):
        key=await asyncio.to_thread(self.key)
        def op(old):
            old=old or {"tasks":{}}
            if old.get('purged'):raise Conflict('Task payloads already removed')
            if task.id not in old["tasks"] and len(old["tasks"])>=2:raise Conflict("Task allowance reached")
            previous=old["tasks"].get(task.id)
            terminal={"TASK_STATE_COMPLETED","TASK_STATE_FAILED","TASK_STATE_CANCELED","TASK_STATE_REJECTED"}
            new=MessageToDict(task,preserving_proto_field_name=True)
            if previous and previous.get("status",{}).get("state") in terminal and previous!=new:
                raise Conflict("Terminal task is immutable")
            old["tasks"][task.id]=new
            return old
        await asyncio.to_thread(mutate,self.store,key,op)

    async def get(self, task_id, context=None):
        key=await asyncio.to_thread(self.key)
        data,_=await asyncio.to_thread(self.store.read,key)
        value=(data or {}).get("tasks",{}).get(task_id)
        return ParseDict(value,Task()) if value else None

    async def list(self, params, context=None):
        key=await asyncio.to_thread(self.key)
        data,_=await asyncio.to_thread(self.store.read,key)
        tasks=[ParseDict(v,Task()) for v in (data or {}).get("tasks",{}).values()]
        if params.context_id:tasks=[t for t in tasks if t.context_id==params.context_id]
        if params.status:tasks=[t for t in tasks if t.status.state==params.status]
        return ListTasksResponse(tasks=tasks,total_size=len(tasks),page_size=len(tasks))

    async def delete(self, task_id, context=None):
        key=await asyncio.to_thread(self.key)
        def op(old):
            old=old or {"tasks":{}};old["tasks"].pop(task_id,None);return old
        await asyncio.to_thread(mutate,self.store,key,op)
