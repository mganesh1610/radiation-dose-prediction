"""Keep agent SDK startup and its I/O pool off the HTTP control event loop."""
import asyncio
import concurrent.futures
import contextvars
import threading


class AgentExecutionLoop:
    """One worker loop, at most two executions; durable admission stays authoritative.

    This is in-process execution, not a detached job or a retry queue. Each
    caller waits for its original result. Cancellation sets the existing fence
    before cancelling the worker task. Managed Sessions caches clients per loop.
    """
    def __init__(self, executor):
        self.executor=executor
        self._ready=threading.Event()
        self._closing=False
        self._tasks={}
        self._thread=threading.Thread(target=self._run,daemon=True,name='agent-execution')
        self._thread.start()
        if not self._ready.wait(5):raise RuntimeError('Agent worker failed to start')

    def _run(self):
        async def serve():
            self._loop=asyncio.get_running_loop()
            self._loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(
                max_workers=8,thread_name_prefix='agent-io'))
            self._slots=asyncio.Semaphore(2)
            self._stop=asyncio.Event()
            self._ready.set()
            await self._stop.wait()
            pending=list(self._tasks)
            for task in pending:
                self._tasks[task].stop.set()
                task.cancel()
            await asyncio.gather(*pending,return_exceptions=True)
        asyncio.run(serve())

    async def __call__(self, scope, session):
        if self._closing:raise RuntimeError('Agent worker is closing')
        result=concurrent.futures.Future()
        done=concurrent.futures.Future()
        cancelled=threading.Event()
        holder=[]

        async def execute():
            async with self._slots:
                if cancelled.is_set() or scope.stop.is_set():raise asyncio.CancelledError()
                return await self.executor(scope,session)

        def completed(task):
            self._tasks.pop(task,None)
            if task.cancelled():result.cancel()
            elif task.exception() is not None:result.set_exception(task.exception())
            else:result.set_result(task.result())
            done.set_result(None)

        def dispatch():
            if self._closing or cancelled.is_set():
                result.cancel();done.set_result(None);return
            task=self._loop.create_task(execute())
            self._tasks[task]=scope
            holder.append(task)
            task.add_done_callback(completed)

        self._loop.call_soon_threadsafe(dispatch,context=contextvars.copy_context())
        wrapped=asyncio.wrap_future(result)
        try:
            return await asyncio.shield(wrapped)
        except asyncio.CancelledError:
            scope.stop.set()
            cancelled.set()
            def cancel():
                if holder:holder[0].cancel()
            self._loop.call_soon_threadsafe(cancel)
            # Observe actual task completion before releasing the caller's run.
            # The durable fence blocks any late provider/storage effects.
            await asyncio.shield(asyncio.wrap_future(done))
            if not wrapped.cancelled() and wrapped.done():wrapped.exception()
            raise

    async def aclose(self):
        if self._closing:return
        self._closing=True
        self._loop.call_soon_threadsafe(self._stop.set)
        await asyncio.to_thread(self._thread.join,20)
        if self._thread.is_alive():raise RuntimeError('Agent worker shutdown did not complete')
