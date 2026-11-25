(() => {
  const training = document.getElementById('guest-training');
  if (!training) return;
  const methods = {
    gradient_boosting:'Gradient boosting (WBC + whole blood data)',
    elastic_net:'Elastic Net regression', pls:'Partial least squares regression', median_null:'Training-median baseline',
  };
  training.innerHTML = `<h3>Start a training run</h3>
    <p>Choose a whole-blood study to hold out. The final model refits on all 567 development profiles.</p>
    <form id="guest-training-form"><label>Training model<select name="method"></select></label>
    <label>Held-out study<select name="holdout"><option>GSE8917</option><option>GSE102971</option><option>GSE65292</option></select></label>
    <button type="submit" class="exp-button-primary">Start training</button></form>
    <p class="muted">Runs use the shared monthly allowance and start when the training worker is online.</p>
    <p role="status" data-feedback></p><div data-history></div>`;
  const form = training.querySelector('form'), feedback = training.querySelector('[data-feedback]');
  for (const [value, label] of Object.entries(methods)) form.elements.method.add(new Option(label,value));
  const api = window.defenderGuestChat;
  let sending = false, timer, refreshTask, pendingRequest;
  function activity() {
    return !document.hidden && location.hash === '#experiments' && !document.getElementById('engineering-demo').hidden;
  }
  function node(tag, text) {
    const el = document.createElement(tag);
    el.textContent = text;
    return el;
  }
  function render(rows) {
    const recent = rows.filter(row=>row.kind==='training');
    const pending = recent.some(row=>['queued','running'].includes(row.state));
    form.querySelector('button').disabled = pending || sending;
    const content = document.createDocumentFragment();
    if (recent.length) content.append(node('h3','Your training runs'));
    for (const row of recent) {
      const section = document.createElement('article');
      section.className = 'guest-result';
      section.append(node('h4',methods[row.input.method]+' · '+row.input.holdout));
      const result = row.result || {};
      if (row.state==='completed') {
        section.append(node('p','Completed · '+result.train_samples+' fit · '+result.validation_samples+' held out · '+result.refit_samples+' in the final refit'));
        const metrics = document.createElement('dl');
        for (const [label, key] of [['Donor-balanced MAE','mae'],['RMSE','rmse'],['Bias','bias']]) {
          const pair = document.createElement('div');
          pair.append(node('dt',label),node('dd',result.metrics[key].toFixed(3)+' Gy')); metrics.append(pair);
        }
        section.append(metrics);
      } else {
        const fallback = row.state==='failed'?'The request could not finish.':'Waiting for the training worker…';
        section.append(node('p',result.message || fallback));
      }
      content.append(section);
    }
    training.querySelector('[data-history]').replaceChildren(content);
    return pending;
  }
  async function refresh() {
    if (refreshTask) return refreshTask;
    clearTimeout(timer);
    refreshTask = (async()=>{
      try {
        await api.ready();
        const data = await api.json('session');
        if (render(data.requests) && activity()) timer=setTimeout(refresh,4000);
      } catch (error) {
        feedback.textContent=error.message;
        if (activity()) timer=setTimeout(refresh,10000);
      } finally {refreshTask=null;}
    })();
    return refreshTask;
  }
  const pendingName = 'defender-pending-training';
  function requestKey(input) {
    try { pendingRequest = JSON.parse(sessionStorage.getItem(pendingName)) || pendingRequest; } catch {}
    const body = JSON.stringify(input);
    if (!pendingRequest || pendingRequest.body !== body) {
      pendingRequest = {key:crypto.randomUUID(),body};
      try { sessionStorage.setItem(pendingName,JSON.stringify(pendingRequest)); } catch {}
    }
    return pendingRequest.key;
  }
  form.addEventListener('submit',async event => {
    event.preventDefault();
    if (sending) return;
    sending=true;
    form.querySelector('button').disabled=true;
    feedback.textContent='Sending…';
    try {
      const input=Object.fromEntries(new FormData(form));
      await api.ready();
      await api.json('requests','POST',{kind:'training',input,key:requestKey(input)});
      pendingRequest=null;
      try { sessionStorage.removeItem(pendingName); } catch {}
      feedback.textContent='Training request received.';
    } catch (error) {
      feedback.textContent=error.message;
    } finally {
      sending=false;
      form.querySelector('button').disabled=false;
      if (refreshTask) await refreshTask;
      await refresh();
    }
  });
  const activate=()=>{if (activity()) refresh();};
  new MutationObserver(activate).observe(document.getElementById('engineering-demo'),{attributes:true,attributeFilter:['hidden']});
  window.addEventListener('hashchange',activate);
  document.addEventListener('visibilitychange',activate);
  activate();
})();
