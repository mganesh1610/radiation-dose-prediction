'use strict';
(()=>{
 const host=document.getElementById('prediction-workflow');if(!host)return;
 const get=id=>document.getElementById(id),el=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;};

 // Async work belongs to the profile that started it. A new selection can supersede it at any time.
 let catalog, selected = null, predicting = null, loaded = false;
 let compatibilityModel = '';
 function compatibilityPath(path) {
  return compatibilityModel ? path + '?model=' + encodeURIComponent(compatibilityModel) : path;
 }
 const savedButtons = new Map();
 const modelNames = window.defenderModelNames;
 async function api(path,body){const r=await fetch(path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json','X-Defender-Request':'1'},body:JSON.stringify(body)});const d=await r.json();if(!r.ok)throw Error(d.error||'Request could not be completed.');return d;}
 function exportJSON(value, name) {
  const dialog = el('dialog', undefined, 'development-export'), title = el('h2', 'Export research record');
  title.id = 'prediction-export-title';
  dialog.setAttribute('aria-labelledby', title.id);
  const area = el('textarea');
  area.value = JSON.stringify(value, null, 2);
  area.readOnly = true;
  area.setAttribute('aria-label', 'Research record JSON');
  const save = el('button', 'Save JSON'), close = el('button', 'Close', 'secondary');
  save.onclick = () => {
   const url = URL.createObjectURL(new Blob([area.value], { type: 'application/json' }));
   const link = el('a');
   link.href = url;
   link.download = name;
   link.click();
   setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  close.onclick = () => dialog.close();
  dialog.addEventListener('close', () => dialog.remove());
  dialog.append(title, el('p', 'Save the file or copy the complete record below.'), area, save, close);
  document.body.append(dialog);
  dialog.showModal();
 }

 function setStatus(message, error = false) {
  const status = get('prediction-status');
  status.textContent = message;
  status.classList.toggle('workflow-error', error);
  status.dataset.approved = String(!error && message === 'Compatibility approved');
 }

 function controls() {
  const profile = selected;
  const ready = Boolean(profile?.input || profile?.uploadId);
  const waiting = Boolean(profile?.loading || profile?.checking || profile?.reviewing);
  const blocked = profile?.record && profile.record.status !== 'approved';
  get('upload-input').disabled = waiting || Boolean(predicting);
  get('predict-upload').disabled = !ready || blocked || Boolean(profile?.reviewing || predicting);
  get('predict-upload').setAttribute('aria-busy', String(Boolean(predicting)));
  if (get('prediction-result').dataset.ready === 'false') updatePlaceholder();
 }

 const placeholderCopy = {
  empty: ['Waiting for a sample', 'Choose a study profile or upload an expression file,\nthen choose Predict to see the estimate.'],
  checking: ['Checking compatibility', 'The selected profile is being checked against the frozen model.'],
  ready: ['Ready to predict', 'This sample passed its compatibility check.\nChoose Predict to calculate its dose estimate.'],
  running: ['Prediction in progress', 'The frozen model is processing this sample.\nThe dose will appear when the prediction finishes.'],
  runFailed: ['Prediction could not finish', 'Check the message under Prediction setup,\nthen choose Predict to try again.'],
  review: ['Compatibility review needed', 'Open Stored uploads to review the profile’s compatibility record.'],
  blocked: ['Profile cannot be used', 'Review the compatibility message or choose another expression profile.'],
  failed: ['Profile needs attention', 'Review the message under Prediction setup before continuing.']
 };

 function placeholderState() {
  if (!selected) return 'empty';
  if (predicting?.running) return 'running';
  if (selected.runFailed) return 'runFailed';
  if (selected.problem) return 'failed';
  if (selected.loading || selected.checking || selected.reviewing) return 'checking';
  if (selected.record?.status === 'approved') return 'ready';
  return selected.record?.status === 'pending_compatibility_review' ? 'review' : 'blocked';
 }

 function updatePlaceholder() {
  const state = placeholderState();
  const [title, instruction] = placeholderCopy[state];
  const mark = ['ready', 'running', 'runFailed'].includes(state) ? 'selected-profile' : 'awaiting-profile';
  const model = get('prediction-model').value;
  get('dose-placeholder-icon').src = '/static/assets/predictions/' + mark + '.svg';
  get('dose-placeholder-title').textContent = title;
  get('dose-placeholder-message').textContent = instruction;
  get('dose-placeholder-sample').textContent = selected?.record?.sample_id || selected?.input?.sample_id || selected?.label || 'None selected';
  get('dose-placeholder-model').textContent = modelNames[model] || model || 'Not selected';
  get('dose-run-status').textContent = ({ running: 'Running', runFailed: 'Failed' })[state] || 'Not run';
 }

 function emptyResult() {
  const box = get('prediction-result');
  if (box.dataset.ready !== 'false') {
   box.dataset.ready = 'false';
   box.innerHTML = `<div class="dose-result-heading"><h2>Research estimate</h2><span class="dose-badge" id="dose-run-status">Not run</span></div>
    <div class="dose-placeholder">
     <img id="dose-placeholder-icon" width="122" height="87" alt="">
     <h3 id="dose-placeholder-title"></h3><p id="dose-placeholder-message"></p>
    </div>
    <dl class="dose-input-summary">
     <div><dt>Study profile</dt><dd id="dose-placeholder-sample"></dd></div>
     <div><dt>Frozen model</dt><dd id="dose-placeholder-model"></dd></div>
    </dl>`;
  }
  updatePlaceholder();
 }

 function beginSelection(description) {
  selected = description ? { label: description, input: null, uploadId: null, record: null, checking: null, loading: true, reviewing: false } : null;
  predicting = null;
  emptyResult();
  get('input-preview').textContent = description || '';
  get('compatibility-record').replaceChildren();
  get('example-study-note').textContent = '';
  setStatus(description ? 'Loading profile…' : 'No profile selected');
  controls();
  return selected;
 }

 function describeCheck(r) {
  if (r.status === 'approved') return 'Compatibility approved';
  if (r.status === 'incompatible') return ['Incompatible profile.', ...r.qc.blockers].join(' ');
  if (r.status === 'rejected') return 'Compatibility review rejected this profile. Select another profile.';
  return 'Awaiting researcher review. Prediction will be available after approval.';
 }

 function checkProfile(profile) {
  if (profile.checking) return profile.checking;
  profile.problem = false;
  profile.runFailed = false;
  emptyResult();
  setStatus('Checking compatibility…');
  profile.checking = (async () => {
   try {
    const r = profile.uploadId
     ? await api(compatibilityPath('/api/uploads/' + profile.uploadId + '/check'), {})
     : await api(compatibilityPath('/api/uploads'), profile.input);
    if (selected === profile) {
     showRecord(profile, r);
     setStatus(describeCheck(r), r.status === 'incompatible' || r.status === 'rejected');
    }
    return r;
   } catch (error) {
    if (selected === profile) {
     profile.record = null;
     profile.problem = true;
     emptyResult();
     get('compatibility-record').replaceChildren();
     setStatus(error.message + ' Use Upload and check to try again.', true);
    }
    return null;
   } finally {
    profile.checking = null;
    if (selected === profile) controls();
   }
  })();
  controls();
  return profile.checking;
 }

 async function loadProfile(profile, read) {
  try {
   const input = await read();
   if (profile !== selected) return;
   if (!input || typeof input !== 'object' || Array.isArray(input)) throw Error('Choose an adapter JSON object.');
   profile.input = input;
   get('input-preview').textContent = [input.sample_id || 'Unnamed profile', input.metadata?.study || 'Study not recorded',
    (input.values?.length || 0).toLocaleString() + ' measurements'].join(' · ');
   profile.loading = false;
   await checkProfile(profile);
  } catch (error) {
   if (profile === selected) {
    profile.problem = true;
    setStatus(error.message, true);
   }
  } finally {
   profile.loading = false;
   if (profile === selected) controls();
  }
 }

 function chooseExample() {
  const picker = get('prediction-example');
  const example = catalog.examples.find(row => row.id === picker.value);
  get('prediction-file').value = '';
  const profile = beginSelection(example?.id);
  if (!example) return;
  get('example-study-note').textContent = example.training_overlap === false
   ? example.study + ' · Held out of training. Previously used in the completed evaluation.'
   : example.study + ' · Used in model training.';
  setStatus('Loading study profile…');
  return loadProfile(profile, () => api('/api/prediction-examples/' + encodeURIComponent(example.id)));
 }

 function chooseFile() {
  const file = get('prediction-file').files[0];
  if (!file) return;
  get('prediction-example').value = '';
  get('prediction-file').value = '';
  const profile = beginSelection(file.name);
  setStatus('Reading file…');
  return loadProfile(profile, async () => {
   if (file.size > 850000) throw Error('File exceeds the 800 kB adapter limit.');
   const raw = await file.text();
   try { return JSON.parse(raw); }
   catch { throw Error('This file is not valid JSON. See Supported inputs for the adapter format.'); }
  });
 }

 async function predict() {
  const profile = selected;
  if ((!profile?.input && !profile?.uploadId) || profile.reviewing || predicting) return;
  profile.runFailed = false;
  const request = { model: get('prediction-model').value };
  predicting = request;
  emptyResult();
  controls();
  try {
   const record = profile.checking ? await profile.checking : profile.record || await checkProfile(profile);
   if (predicting !== request || record?.status !== 'approved') return;
   request.running = true;
   updatePlaceholder();
   setStatus('Running the frozen model…');
   const result = await api('/api/predict', { upload_id: record.upload_id, model: request.model });
   if (predicting !== request) return;
   if (result.status !== 'completed') {
    if (result.record) showRecord(profile, result.record);
    setStatus(result.record ? describeCheck(result.record) : result.explanation, true);
    return;
   }
   if (!Number.isFinite(result.estimate_gy) || result.sample_id !== record.sample_id || result.model !== request.model) {
    throw Error('A complete estimate for this sample and model was not returned. Choose Predict to retry.');
   }
   showResult(result);
   setStatus(describeCheck(record));
  } catch (error) {
   if (predicting === request) {
    profile.runFailed = true;
    setStatus(error.message, true);
   }
  } finally {
   if (predicting === request) {
    predicting = null;
    controls();
   }
  }
 }

 function addStoredUpload(uploadId, label) {
  let button = savedButtons.get(uploadId);
  if (!button) {
   const list = get('upload-list');
   if (!savedButtons.size) list.replaceChildren();
   button = el('button', '', 'secondary');
   button.type = 'button';
   button.onclick = () => {
    get('prediction-example').value = '';
    get('prediction-file').value = '';
    const profile = beginSelection(button.textContent);
    profile.uploadId = uploadId;
    profile.loading = false;
    return checkProfile(profile);
   };
   list.append(button);
   savedButtons.set(uploadId, button);
  }
  button.textContent = label;
 }

 function showRecord(profile, r) {
  profile.record = r;
  profile.uploadId = r.upload_id;
  if (!profile.input) get('input-preview').textContent = r.sample_id + ' · stored profile';
  if (r.status !== 'approved') emptyResult();
  addStoredUpload(r.upload_id, r.sample_id);
  const node=get('compatibility-record');node.replaceChildren();node.append(el('p',r.status.replaceAll('_',' '),'workflow-state'),el('h3',r.sample_id));const qc=el('dl',undefined,'profile-qc');for(const [label,value]of [['Usable probes',r.qc.usable_probes],['Masked probes',r.qc.masked_probes]]){const pair=el('div');pair.append(el('dt',label),el('dd',value.toLocaleString()));qc.append(pair);}node.append(qc);
  for(const message of r.qc.blockers)node.append(el('p',message,'workflow-error'));
  if(r.status==='pending_compatibility_review')node.append(el('p','Stored for a researcher to review. Valid file structure alone does not establish assay compatibility. No dose is available.'));
  if(r.curated)node.append(el('p',trainingOverlapText(r.training_overlap)));
  const details=el('details'),sum=el('summary','Input provenance and compatibility record'),pre=el('pre',JSON.stringify(r,null,2));details.append(sum,pre);node.append(details);
  if(catalog.role==='researcher'){
   const review=el('details'),summary=el('summary','Record a compatibility review'),form=el('form'),checks=el('fieldset'),legend=el('legend','Verify the source records and the exact adapter before approval');checks.append(legend);
   const names={assay:'Assay adapter matches this frozen model',gene_mapping:'Probe identities and mapping are verified',normalization:'Units and transformations are verified',technical_flags:'Technical flags were applied before normalization',tissue_exposure_time:'Human whole blood, exposure and timing are eligible',missingness:'Missingness and imputation have been reviewed'};
   for(const [key,label]of Object.entries(names)){const l=el('label'),i=el('input');i.type='checkbox';i.name=key;l.append(i,document.createTextNode(label));checks.append(l);}
   const label=el('label','Evidence and QC rationale'),evidence=el('textarea');evidence.required=true;evidence.minLength=40;evidence.maxLength=2000;label.append(evidence);
   const overlapLabel=el('label','Training overlap'),overlap=el('select');for(const v of ['unknown','yes','no']){const o=el('option',v);o.value=v;overlap.append(o);}overlapLabel.append(overlap);
   const decisionLabel=el('label','Decision'),decision=el('select');for(const v of ['rejected','approved']){const o=el('option',v);o.value=v;decision.append(o);}decisionLabel.append(decision);
   const submit=el('button','Save compatibility decision');submit.type='submit';form.append(checks,label,overlapLabel,decisionLabel,submit);

   form.onsubmit = async event => {
    event.preventDefault();
    if (selected !== profile || profile.reviewing || profile.checking || predicting) return;
    profile.reviewing = true;
    emptyResult();
    submit.disabled = true;
    controls();
    setStatus('Saving compatibility review…');
    try {
     const result = await api(compatibilityPath('/api/uploads/' + r.upload_id + '/review'), {
      decision: decision.value, evidence: evidence.value, training_overlap: overlap.value,
      checks: Object.fromEntries([...checks.querySelectorAll('input')].map(i => [i.name, i.checked]))
     });
     if (selected === profile) {
      emptyResult();
      showRecord(profile, result);
      setStatus(describeCheck(result));
     }
    } catch (error) {
     if (selected === profile) setStatus(error.message, true);
    } finally {
     profile.reviewing = false;
     submit.disabled = false;
     if (selected === profile) controls();
    }
   };
   review.append(summary, form);
   node.append(review);
  }
 }
 function trainingOverlapText(value) {
  if (value === true || value === 'yes') return 'This profile was used in training. Assessing accuracy on new samples requires an independent evaluation.';
  if (value === false) return 'This profile was held out of model training and was part of the completed GSE90909 evaluation.';
  if (value === 'no') return 'The compatibility review found no known training overlap.';
  return 'Whether this profile was used in training is unknown.';
 }
 function showResult(result) {
  const box = get('prediction-result');
  const heading = el('div', undefined, 'dose-result-heading');
  heading.append(el('h2', 'Research estimate'), el('span', 'Complete', 'dose-badge'));

  const readout = el('div', undefined, 'dose-readout');
  const value = el('p', undefined, 'estimate-value');
  value.append(el('span', result.estimate_gy.toFixed(3)), el('small', 'Gy'));
  const sample = el('p', result.sample_id, 'result-sample');
  const comparison = result.dose_comparison;
  if (comparison) {
   const link = el('a', result.sample_id);
   link.href = 'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=' + encodeURIComponent(result.sample_id);
   link.target = '_blank';
   link.rel = 'noopener noreferrer';
   link.setAttribute('aria-label', result.sample_id + ' study record on GEO (opens in a new tab)');
   sample.replaceChildren(link, '  /  ' + comparison.study);
  }
  readout.append(value, sample);

  const comparisons = el('dl', undefined, 'dose-comparison');
  function metric(label, text) {
   const pair = el('div');
   pair.append(el('dt', label), el('dd', text));
   comparisons.append(pair);
  }
  metric('Recorded study dose', comparison ? comparison.observed_dose_gy + ' Gy' : 'Unavailable for this profile');
  if (comparison) {
   const difference = comparison.estimate_minus_observed_gy.toLocaleString('en-US', {
    minimumFractionDigits: 3, maximumFractionDigits: 3, signDisplay: 'exceptZero'
   }).replace('-', '−');
   metric('Estimate − recorded', difference + ' Gy');
  }

  const evidence = el('div', undefined, 'dose-evidence');
  const seed = result.seed == null ? 'fixed median' : 'seed ' + result.seed;
  evidence.append(el('p', [modelNames[result.model] || result.model, result.release, seed].join('  ·  '), 'dose-model-record'));
  const membership = result.training_overlap === false ? 'Held out of training.' : trainingOverlapText(result.training_overlap);
  const intervalNote = result.individual_prediction_interval === null ? ' No individual prediction interval is available.' : '';
  evidence.append(el('p', membership + intervalNote));
  if (result.outside_training_dose_range) evidence.append(el('p', 'Outside the 0–8 Gy training range. The estimate has not been clipped.', 'result-caution'));

  const actions = el('div', undefined, 'dose-result-actions');
  const details = el('details', undefined, 'dose-disclosure');
  details.append(el('summary', 'Interpretation & limitations'), el('p', trainingOverlapText(result.training_overlap)));
  for (const limitation of result.limitations) details.append(el('p', limitation));
  const exportButton = el('button', 'Export estimate and provenance', 'dose-export');
  exportButton.type = 'button';
  exportButton.onclick = () => exportJSON(result, 'defender-' + result.sample_id + '-' + result.model + '.json');
  actions.append(details, exportButton);

  box.replaceChildren(heading, readout, comparisons, evidence, actions);
  box.dataset.ready = 'true';
  box.tabIndex = -1;
  box.focus({ preventScroll: !window.matchMedia('(max-width: 1050px)').matches });
 }

 async function initialize() {
  if (loaded || location.hash !== '#predictions') return;
  loaded = true;
  host.setAttribute('aria-busy', 'true');
  try {
   catalog = await api('/api/prediction-catalog');
   host.innerHTML = `
    <div class="dose-layout">
     <section class="dose-setup" aria-labelledby="input-title">
      <h2 id="input-title">Prediction setup</h2>
      <div class="dose-inputs">
       <div class="dose-choices">
        <div class="dose-field">
         <div class="dose-field-heading"><label for="prediction-example">Study profile</label><span id="dose-picker-key">Recorded dose</span></div>
         <select id="prediction-example" name="study-profile" autocomplete="off" aria-describedby="dose-picker-key example-study-note"><option value="">Choose a profile</option></select>
        </div>
        <div class="dose-field">
         <label for="prediction-model">Frozen model</label>
         <select id="prediction-model" name="frozen-model" autocomplete="off"></select>
        </div>
       </div>
       <label class="dose-file">
        <span class="dose-file-icon" aria-hidden="true"></span>
        <span class="dose-file-name"><strong>Choose an expression file</strong><small>Agilent adapter JSON · up to 800 kB</small></span>
        <span class="dose-browse" aria-hidden="true">Browse</span>
        <input id="prediction-file" name="profile" type="file" aria-label="Choose an adapter JSON file" aria-describedby="profile-check-hint" accept=".json,application/json">
       </label>
       <p id="profile-check-hint">Selecting a profile or file starts the compatibility check automatically.</p>
      </div>
      <div class="dose-commands">
       <p id="prediction-status" role="status">No profile selected</p>
       <div class="dose-actions">
        <button id="upload-input" type="button" class="secondary">Upload and check</button>
        <button id="predict-upload" type="button" disabled>Predict</button>
       </div>
      </div>
      <div class="dose-disclosures">
       <details class="dose-disclosure">
        <summary>Supported inputs</summary>
        <p>Use the Agilent JSON adapter for one de-identified sample. New profiles require a researcher’s compatibility review. Raw CEL/FASTQ files and normalized matrices require an audited preprocessing adapter.</p>
        <button id="export-input" type="button" class="text-button">View input format</button>
       </details>
       <details class="dose-disclosure stored-inputs"><summary>Stored uploads</summary>
        <p id="input-preview"></p><p id="example-study-note"></p><div id="upload-list"></div><div id="compatibility-record"></div>
       </details>
      </div>
     </section>
     <section id="prediction-result" aria-live="polite" aria-label="Prediction result"></section>
    </div>`;
   emptyResult();
   const picker = get('prediction-example');
   const trainingGroup = el('optgroup'), heldOutGroup = el('optgroup');
   trainingGroup.label = 'Used in training';
   heldOutGroup.label = 'Held out of training · GSE90909';
   const doseFormat = new Intl.NumberFormat(undefined, { maximumSignificantDigits: 15 });
   for (const row of catalog.examples) {
    const heldOut = row.training_overlap === false;
    const option = el('option');
    option.value = row.id;
    const recorded = Number.isFinite(row.observed_dose_gy)
     ? doseFormat.format(row.observed_dose_gy) + '\u00a0Gy'
     : 'Dose unavailable';
    option.append(el('span', row.id, 'dose-accession'), el('span', ' · ', 'dose-option-divider'), el('span', recorded, 'dose-option-value'));
    (heldOut ? heldOutGroup : trainingGroup).append(option);
   }
   for (const group of [trainingGroup, heldOutGroup]) {
    if (group.childElementCount) {
     group.setAttribute('aria-label', group.label);
     picker.append(group);
    }
   }
   for (const method of catalog.methods) {
    const option = el('option', modelNames[method] || method);
    option.value = method;
    get('prediction-model').append(option);
   }
   const modelPicker = get('prediction-model');
   const requestedModel = new URLSearchParams(location.search).get('model');
   const preferredModel = [requestedModel, 'gradient_boosting_wb_wbc'].find(model => catalog.methods.includes(model));
   modelPicker.value = preferredModel || catalog.methods[0] || '';
   const selectedScope = () => catalog.model_release_overrides?.[modelPicker.value] ? modelPicker.value : '';
   compatibilityModel = selectedScope();
   picker.onchange = chooseExample;
   get('prediction-file').onchange = chooseFile;
   modelPicker.onchange = () => {
    const previousScope = compatibilityModel;
    compatibilityModel = selectedScope();
    predicting = null;
    if (previousScope !== compatibilityModel && selected && !selected.loading && (selected.input || selected.uploadId)) {
     // A pending response for the previous release belongs to the old selection.
     selected = { ...selected, checking:null, reviewing:false, record:null };
     get('compatibility-record').replaceChildren();
     checkProfile(selected);
     return;
    }
    if (selected) selected.runFailed = false;
    emptyResult();
    if (selected?.record && !selected.checking && !selected.reviewing) setStatus(describeCheck(selected.record));
    controls();
   };
   get('upload-input').onclick = () => {
    if (selected?.input || selected?.uploadId) return checkProfile(selected);
    if (picker.value) return chooseExample();
    get('prediction-file').click();
   };
   get('predict-upload').onclick = predict;
   get('export-input').onclick = async () => {
    const profile = selected;
    try {
     const body = profile?.input || await api('/api/prediction-examples/' + encodeURIComponent(picker.value || catalog.examples[0].id));
     if (profile === selected) exportJSON(body, body.sample_id + '.json');
    } catch (error) {
     if (profile === selected) setStatus(error.message, true);
    }
   };
   if (!catalog.uploads.length) get('upload-list').append(el('p', 'No stored uploads yet.'));
   catalog.uploads.forEach((row, i) => addStoredUpload(row.upload_id, 'Saved profile ' + String(i + 1).padStart(2, '0')));
   controls();
  } catch (error) {
   loaded = false;
   host.replaceChildren(el('p', error.message));
   const retry = el('button', 'Retry prediction workspace');
   retry.onclick = initialize;
   host.append(retry);
  } finally {
   host.removeAttribute('aria-busy');
  }
 }
 window.addEventListener('hashchange', initialize);
 initialize();
})();
