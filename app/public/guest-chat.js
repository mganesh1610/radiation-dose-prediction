'use strict';
(() => {
  let session;
  async function json(path, method = 'GET', body) {
    const response = await fetch('/api/guest/' + path, {
      method, credentials:'same-origin', cache:'no-store',
      ...(body === undefined ? {} : {headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),
    });
    const data = await response.json();
    if (!response.ok) {
      const error = new Error(data.error || 'The guest workspace is temporarily unavailable.');
      error.status = response.status;
      error.quota = data.quota;
      throw error;
    }
    return data;
  }
  function ready() {
    if (!session) session = json('session').catch(error => { session = null; throw error; });
    return session;
  }

  // Published-record questions stay in this visit. Only Gemini turns use the
  // guest queue and its existing browser-owned, 30-day conversation history.
  function create(records) {
    const visits = new Map(), receipts = new Map();
    const modeFor = id => id?.startsWith('record-') ? 'records' : 'gemini';
    const store = {mode:'gemini',modeFor,request};
    async function request(path, method = 'GET', body) {
      const [resource,id,action] = path.split('?')[0].split('/');
      if (resource === 'chats' && !id && method === 'POST') {
        const chat = {chat_id:(store.mode === 'records' ? 'record-' : '') + body.request_id,
          created_at:Date.now()/1000,active_run:null};
        if (store.mode === 'records') visits.set(chat.chat_id,{chat,messages:[],answers:new Map()});
        return chat;
      }
      if (resource === 'chats' && modeFor(id) === 'records') {
        const visit = visits.get(id);
        if (!visit) throw Error('This published-answer chat has ended.');
        if (method === 'DELETE') {
          for (const turn of visit.answers.values()) receipts.delete(turn);
          visits.delete(id); return {deleted:true};
        }
        if (action === 'messages' && method === 'POST') {
          if (!visit.answers.has(body.request_id)) {
            const answer = window.defenderAnswers.answer(body.text,records);
            const run = 'run-record-' + body.request_id;
            visit.chat.title ||= body.text.slice(0,72);
            visit.messages.push({role:'user',text:body.text},{role:'assistant',text:answer.answer,invocation_id:run});
            visit.answers.set(body.request_id,run);
            receipts.set(run,{run_id:run,status:answer.status,sources:answer.sources,model_calls:0,tool_calls:1});
          }
          return {status:'completed'};
        }
        return {chat:visit.chat,messages:visit.messages,next_offset:null};
      }
      if (resource === 'runs' && id?.startsWith('run-record-')) return receipts.get(id);
      await ready();
      const result = await json('copilot/' + path,method,body);
      if (resource === 'chats' && !id) result.chats.push(...[...visits.values()].map(visit => visit.chat));
      return result;
    }
    return store;
  }
  window.defenderGuestChat = Object.freeze({create,ready,json});
})();
