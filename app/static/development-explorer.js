'use strict';
// Read-only views of saved research results. No model fitting or training requests.
(()=>{
 const get=id=>document.getElementById(id), el=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;};
 const palette=['#80dfb9','#eda871','#94bdec','#c5a4e8','#e6d582','#efa6bd'];
 const studies=['GSE102971','GSE8917','GSE65292','GSE85570','GSE97000','GSE20173'];
 const methods={kmeans:'K-means',ward:'Ward hierarchical',gmm:'Gaussian mixture'};
 let data, pending, clusterReady=false, comparisonReady=false, displayed;
 let clusterScene, sceneLoading, clusterPlot, selectedId=null, projection='3d', webglUnavailable=false;
 const num=(n,d=3)=>n===null?'Not defined':Number(n).toFixed(d);
 function download(value,name){
  const text=JSON.stringify(value,null,2),dialog=el('dialog'),title=el('h2','Export data'),note=el('p','Save the JSON file, or select and copy the data below if your browser does not support downloads.'),area=el('textarea'),save=el('button','Save JSON'),close=el('button','Close export','secondary'),actions=el('div',undefined,'export-actions'),opener=document.activeElement;
  dialog.className='development-export';title.id='development-export-title';dialog.setAttribute('aria-labelledby',title.id);area.value=text;area.readOnly=true;area.setAttribute('aria-label','Export JSON');save.onclick=()=>{const url=URL.createObjectURL(new Blob([text],{type:'application/json'}));const a=el('a');a.href=url;a.download=name;a.hidden=true;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);};close.onclick=()=>dialog.close();dialog.addEventListener('close',()=>{dialog.remove();opener?.focus();},{once:true});actions.append(save,close);dialog.append(title,note,area,actions);document.body.append(dialog);dialog.showModal();
 }
 function table(parent,headers,rows){const wrap=el('div',undefined,'table-scroll'),t=el('table'),head=el('thead'),hr=el('tr'),body=el('tbody');for(const h of headers){const th=el('th',h);th.scope='col';hr.append(th);}head.append(hr);for(const row of rows){const tr=el('tr');for(const c of row)tr.append(el('td',c));body.append(tr);}t.append(head,body);wrap.append(t);parent.replaceChildren(wrap);}
 function select(parent,id,label,options,value){const l=el('label',label),s=el('select');s.id=id;for(const [v,name]of options){const o=el('option',name);o.value=v;s.append(o);}s.value=value;l.append(s);parent.append(l);return s;}
 async function load(){if(data)return data;if(!pending)pending=(async()=>{if(window.defenderPublic)return window.defenderPublic.read('/published/development-results');const r=await fetch('/static/development-results.json');if(!r.ok)throw Error('Saved development results are unavailable.');return r.json();})().then(d=>{if(![2,3].includes(d.schema_version)||d.eligible_profiles!==567||d.held_out_profiles!==145||d.final_test_included!==false||d.regression.length!==20)throw Error('Unsupported development release.');data=d;return d;}).catch(e=>{pending=null;throw e;});return pending;}
 async function ensure(kind){const host=get(kind==='clusters'?'clustering-explorer':'development-results');if(kind==='clusters'?clusterReady:comparisonReady)return;host.textContent='Loading saved development results…';host.setAttribute('aria-busy','true');try{await load();if(kind==='clusters'){if(!clusterReady){buildClusters(host);clusterReady=true;}}else if(!comparisonReady){buildComparison(host);comparisonReady=true;}}catch(e){host.replaceChildren(el('p',e.message));const b=el('button','Retry saved results','secondary');b.onclick=()=>ensure(kind);host.append(b);}finally{host.removeAttribute('aria-busy');}}
 function buildClusters(host){
  host.innerHTML=`<div class="dev-heading"><p class="eyebrow">DEVELOPMENT · EXPLORATORY</p><h2>Where do the profiles group?</h2><p>Compare expression similarity across studies. Change the overlays to see what each cluster contains.</p></div>
   <div class="cluster-controls" id="cluster-controls"></div>
   <p id="cluster-scope" class="muted" role="status" aria-live="polite"></p>
   <div class="cluster-grid"><figure class="cluster-figure">
    <div class="cluster-plot-tools"><div role="group" aria-label="Clustering projection"><button type="button" id="cluster-3d" aria-pressed="false">3D</button><button type="button" id="cluster-2d" aria-pressed="true">2D</button></div><button type="button" id="cluster-reset" hidden>Reset view</button></div>
    <div id="cluster-legend" class="plot-legend"></div><div id="cluster-flat"><svg id="cluster-chart" role="group" aria-label="Saved PCA coordinates and cluster assignments"></svg></div>
    <div id="cluster-spatial" hidden></div><figcaption id="cluster-caption" aria-live="polite"></figcaption></figure>
   <div class="cluster-facts"><h3>Full fitted cohort</h3><dl id="cluster-metrics"></dl><p>Study ARI measures agreement with study membership: 1 = identical partitions, around 0 = chance agreement. It is not prediction accuracy.</p><p id="cluster-k-note"></p></div></div>
   <p id="cluster-point" class="point-detail" role="status">Select a point to inspect its profile, study, known dose and cluster. Arrow keys move between points.</p>
   <div class="dev-section-heading"><h3>Who is in each cluster?</h3><button id="cluster-download" class="secondary">Export displayed data</button></div><p class="muted">Profile counts for the displayed studies. Cluster numbers are arbitrary labels.</p><div id="cluster-contingency"></div>
   <details><summary>View profiles and coordinates</summary><div id="cluster-point-table"></div></details>
   <div class="reviewer-finding"><strong>Study-associated differences remain visible.</strong><p>The core Gaussian mixture groups match the three studies. Controls also separate by study. This does not identify a unique technical or biological cause, and clusters are never used here to remove samples or assign dose classes.</p></div>
   <details><summary>Clustering methods and limits</summary><div class="reviewer-prose"><p>Saved diagnostic fits use 500 genes selected by weighted variance, without dose labels, then study/donor-balanced scaling and ten principal components. Log2 and within-profile rank are separate representations. K-means uses study/donor weights; Ward and Gaussian mixture fits are unweighted after the balanced representation.</p><p>These descriptive references use the full selected development pool. Controls-only sensitivity reuses that pool’s PCA and fixes k to the number of studies. It does not estimate the natural number of biological groups. Silhouette and ARI describe profiles, not independent donors.</p><p>The 567 eligible profiles comprise 145 core and 422 auxiliary profiles. The source inventory held 30,053,455 expression values across 607 candidate profiles; exclusions and technical-repeat grouping account for the reduction. Expression values are not independent training samples. Predictive clustering experiments use separate preprocessing fitted inside training folds.</p></div></details>`;
  const controls=get('cluster-controls'),query=new URLSearchParams(location.search);
  select(controls,'cluster-pool','Reference cohort',[['core','Core · 145 profiles'],['all','All six studies · 567 profiles']],query.get('pool')==='all'?'all':'core');
  select(controls,'cluster-study','Study to display',[['all','All studies']],'all');
  select(controls,'cluster-representation','Representation',[['log2','Normalized log2'],['rank','Within-profile rank']],'log2');
  select(controls,'cluster-method','Clustering method',Object.entries(methods),'gmm');
  select(controls,'cluster-controls-only','Profiles',[['false','All eligible profiles'],['true','Unexposed controls only']],'false');
  select(controls,'cluster-color','Color by',[['study','Study'],['cluster','Cluster'],['dose','Known dose group']],'study');
  for(const s of controls.querySelectorAll('select'))s.onchange=()=>{if(s.id==='cluster-pool')updateStudies();drawClusters();};
  get('cluster-3d').onclick=()=>{projection='3d';showProjection();};
  get('cluster-2d').onclick=()=>{projection='2d';showProjection();};
  get('cluster-reset').onclick=()=>clusterScene?.reset();
  get('cluster-download').onclick=()=>download(displayed,'defender-clustering-displayed.json');updateStudies();if([...get('cluster-study').options].some(o=>o.value===query.get('study')))get('cluster-study').value=query.get('study');drawClusters();
 }
 function updateStudies(){const s=get('cluster-study'),old=s.value;s.replaceChildren();const pool=studies.filter(st=>data.views.find(v=>v.scope===get('cluster-pool').value).points.some(p=>p.study===st));for(const value of ['all',...pool]){const o=el('option',value==='all'?'All studies':value);o.value=value;s.append(o);}s.value=pool.includes(old)?old:'all';}
 function pointDetails(p) {
  const pc3=Number.isFinite(p.z)?`, PC3 ${num(p.z)}`:'';
  return `${p.id} · ${p.study} · ${p.dose_gy} Gy · Cluster ${p.cluster+1} · PC1 ${num(p.x)}, PC2 ${num(p.y)}${pc3}`;
 }
 function selectClusterPoint(i) {
  const point=clusterPlot.points[i];
  selectedId=point?.id??null;
  get('cluster-point').textContent=point?pointDetails(point):'Select a point to inspect its profile, study, known dose and cluster. Arrow keys move between points.';
  clusterPlot.nodes.forEach((node,n)=>node.setAttribute('r',n===i?7:4.5));
  clusterScene?.select(i);
 }
 function fallBackTo2D() {
  webglUnavailable=true;
  showProjection();
 }
 function showProjection() {
  const hasPC3=clusterPlot.hasPC3;
  const wanted=projection==='3d'&&hasPC3&&!webglUnavailable;
  const active=Boolean(wanted&&clusterScene);
  get('cluster-3d').disabled=!hasPC3||webglUnavailable;
  get('cluster-3d').setAttribute('aria-pressed',String(active));
  get('cluster-2d').setAttribute('aria-pressed',String(!active));
  get('cluster-flat').hidden=active;
  get('cluster-spatial').hidden=!active;
  get('cluster-reset').hidden=!active;
  if(active){
   try {
    clusterScene.setResult(clusterPlot);
    clusterScene.select(clusterPlot.points.findIndex(p=>p.id===selectedId));
    clusterScene.setActive(true);
    get('cluster-caption').textContent='Weighted PC1 and PC2 form the base. Height shows weighted PC3. The fit uses ten components. Drag or use Shift + arrow keys to rotate. Scroll to zoom.';
   }catch{fallBackTo2D();}
   return;
  }
  clusterScene?.setActive(false);
  let caption='Weighted PC1 and PC2 of the ten components used for clustering.';
  if(!hasPC3)caption+=' This saved view contains two coordinates per profile.';
  else if(webglUnavailable)caption+=' 3D rendering is unavailable in this browser. PC3 is in the coordinate table.';
  else if(wanted)caption+=' Loading the 3D view…';
  get('cluster-caption').textContent=caption;
  if(wanted&&!sceneLoading){
   sceneLoading=import('/static/analysis-scene.mjs').then(module=>{
    clusterScene=module.createClusterScene(get('cluster-spatial'),{onSelect:selectClusterPoint,onUnavailable:fallBackTo2D});
    showProjection();
   }).catch(fallBackTo2D);
  }
 }
 function drawClusters(){
  const scope=get('cluster-pool').value,representation=get('cluster-representation').value,method=get('cluster-method').value,study=get('cluster-study').value,controls=get('cluster-controls-only').value==='true',color=get('cluster-color').value;
  const view=data.views.find(v=>v.scope===scope&&v.representation===representation),fit=view.fits.find(f=>f.method===method&&f.controls_only===controls);
  const full=view.points.map((p,i)=>({...p,cluster:fit.labels[i]})).filter(p=>p.cluster!==null),points=full.filter(p=>study==='all'||p.study===study);
  get('cluster-scope').textContent=`${points.length} profiles displayed · ${full.length} profiles clustered · ${fit.k} clusters. Study filtering changes the display only; the fit, coordinates and axis limits stay fixed.`;
  get('cluster-k-note').textContent=controls?`Controls-only sensitivity: k = ${fit.k} fixed to the study count; PCA was fitted to all ${view.points.length} profiles.`:`k = ${fit.k} selected from 2–6 by highest primary-seed silhouette within this method and representation. No dose labels used.`;
  const metrics=get('cluster-metrics');metrics.replaceChildren();for(const [name,value]of [['Study ARI',fit.study_ari],['Silhouette',fit.silhouette],['Dose-group ARI',fit.dose_group_ari],['Exposure ARI',fit.exposure_ari]]){const d=el('div');d.append(el('dt',name),el('dd',num(value)));metrics.append(d);}
  const keys=color==='study'?studies.filter(st=>points.some(p=>p.study===st)):color==='cluster'?Array.from({length:fit.k},(_,i)=>`Cluster ${i+1}`):['0 Gy','>0–2 Gy','>2 Gy'];
  const key=p=>color==='study'?p.study:color==='cluster'?`Cluster ${p.cluster+1}`:p.dose_gy===0?'0 Gy':p.dose_gy<=2?'>0–2 Gy':'>2 Gy';
  const paint=k=>palette[(color==='study'?studies.indexOf(k):keys.indexOf(k))%palette.length];
  get('cluster-legend').replaceChildren(...keys.map(k=>{const n=el('span'),dot=el('i');dot.style.background=paint(k);n.append(dot,el('span',k));return n;}));
  const svg=get('cluster-chart');svg.replaceChildren();const W=matchMedia('(max-width:580px)').matches?340:700;svg.setAttribute('viewBox',`0 0 ${W} 370`);
  function sn(tag,attrs,text,parent=svg){const n=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;parent.append(n);return n;}
  const xs=view.points.map(p=>p.x),ys=view.points.map(p=>p.y),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys),dx=(xmax-xmin)||1,dy=(ymax-ymin)||1;
  const x=v=>65+((v-xmin)/dx)*(W-95),y=v=>315-((v-ymin)/dy)*285;
  for(let i=0;i<=4;i++){const xx=x(xmin+dx*i/4),yy=y(ymin+dy*i/4);sn('line',{x1:xx,x2:xx,y1:20,y2:325,class:'cluster-gridline'});sn('line',{x1:55,x2:W-20,y1:yy,y2:yy,class:'cluster-gridline'});sn('text',{x:xx,y:345,'text-anchor':'middle'},num(xmin+dx*i/4,1));sn('text',{x:48,y:yy+4,'text-anchor':'end'},num(ymin+dy*i/4,1));}
  sn('text',{x:W/2,y:368,'text-anchor':'middle'},'Weighted PC1');sn('text',{x:10,y:170,transform:'rotate(-90 10 170)','text-anchor':'middle'},'Weighted PC2');
  const nodes=[];
  points.forEach((p,i)=>{const detail=pointDetails(p),n=sn('circle',{cx:x(p.x),cy:y(p.y),r:4.5,fill:paint(key(p)),stroke:'#112e26','stroke-width':.6,tabindex:i===0?0:-1,role:'button','aria-label':detail});sn('title',{},detail,n);const show=()=>selectClusterPoint(i);n.onclick=show;n.onfocus=show;n.onkeydown=e=>{if(['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home','End'].includes(e.key)){e.preventDefault();const j=e.key==='Home'?0:e.key==='End'?points.length-1:Math.max(0,Math.min(points.length-1,i+(['ArrowLeft','ArrowUp'].includes(e.key)?-1:1)));n.tabIndex=-1;nodes[j].setAttribute('tabindex','0');nodes[j].focus();}else if(['Enter',' '].includes(e.key)){e.preventDefault();show();}};nodes.push(n);});
  const hasPC3=view.points.every(p=>Number.isFinite(p.z));
  clusterPlot={id:scope+'-'+representation,reference:view.points,points,nodes,hasPC3,colors:points.map(p=>paint(key(p)))};
  selectClusterPoint(points.findIndex(p=>p.id===selectedId));
  showProjection();
  const visibleStudies=studies.filter(st=>points.some(p=>p.study===st));
  table(get('cluster-contingency'),['Cluster',...visibleStudies,'Total'],Array.from({length:fit.k},(_,k)=>[`Cluster ${k+1}`,...visibleStudies.map(st=>points.filter(p=>p.cluster===k&&p.study===st).length),points.filter(p=>p.cluster===k).length]));
  const coordinateKeys=hasPC3?['x','y','z']:['x','y'];
  table(get('cluster-point-table'),['Profile','Study','Known dose (Gy)','Cluster',...coordinateKeys.map((_,i)=>'PC'+(i+1))],
   points.map(p=>[p.id,p.study,p.dose_gy,p.cluster+1,...coordinateKeys.map(k=>num(p[k]))]));
  displayed={release_id:data.release_id,purpose:data.purpose,scope,representation,method,controls_only:controls,study_filter:study,filter_refits:false,k:fit.k,full_cohort_metrics:{study_ari:fit.study_ari,silhouette:fit.silhouette,dose_group_ari:fit.dose_group_ari,exposure_ari:fit.exposure_ari},points};
 }
 function externalScore(row) {
  const evaluation = data.retrospective_evaluation;
  return row.model === 'gradient_boosting' ? evaluation?.fixed_whole_blood_control : evaluation?.models[row.model];
 }
 const excludedPredictors = new Set(['xgboost', 'random_forest', 'svr_rbf']);
 function visiblePredictor(row) {
  const score = externalScore(row)?.mae_gy;
  return !excludedPredictors.has(row.model) && !(Number.isFinite(score) && score < 0.770);
 }
 function comparisonTable(parent, items, firstHeading, ascending, toggleSort) {
  const t = el('table', undefined, 'exp-table exp-benchmark');
  const head = t.createTHead().insertRow();
  for (const [index, label] of [firstHeading, 'Development MAE (Gy)', 'GSE90909 MAE (Gy)'].entries()) {
   const th = el('th');
   th.scope = 'col';
   if (index === 1) {
    th.setAttribute('aria-sort', ascending ? 'ascending' : 'descending');
    const sort = el('button', label, 'exp-dev-sort');
    sort.type = 'button'; sort.setAttribute('aria-label', 'Sort ' + firstHeading.toLowerCase() + ' by development MAE, ' + (ascending ? 'highest' : 'lowest') + ' first');
    sort.onclick = toggleSort; th.append(sort);
   } else th.textContent = label;
   head.append(th);
  }
  const body = t.createTBody();
  const rows = [...items].sort((a,b) => (a.mae_gy - b.mae_gy) * (ascending ? 1 : -1));
  for (const row of rows) {
   const tr = body.insertRow();
   const heading = el('th', window.defenderModelNames[row.model] || row.label);
   heading.scope = 'row';
   const external = externalScore(row);
   tr.append(heading, el('td', num(row.mae_gy), 'exp-mae-cell'),
    el('td', external ? num(external.mae_gy) : '—', 'exp-external-mae'));
  }
  parent.replaceChildren(t);
 }
 function buildComparison(host) {
  host.innerHTML = `<div class="exp-dev-title"><h2>Dose model comparison</h2>
   <p>567 training profiles · 145 whole blood + 422 WBC</p><p class="exp-sub">Development MAE covers 145 profiles from three held-out studies (30 donor groups). GSE90909 MAE covers 52 profiles from 12 donors in a previously evaluated study, after fitting on all 567. Lower error is better.</p><p id="comparison-run-date" class="exp-sub"></p></div>
   <button type="button" id="development-download" class="exp-button"><img src="/static/assets/experiments/download.svg" alt="">
    <span class="exp-wide">Export comparison</span><span class="exp-narrow">Export data</span></button>
   <div id="model-family-control" class="exp-family"></div>
   <section class="exp-card exp-tuned" aria-labelledby="exp-tuned-title"><div class="exp-card-heading"><div>
    <h3 id="exp-tuned-title">Dose predictors</h3>
    <p id="model-count" class="exp-sub" role="status"></p></div></div><div id="development-regression"></div></section>
   <section id="matched-boosting" class="exp-card" aria-labelledby="exp-pair-title" hidden><div class="exp-card-heading"><div>
    <h3 id="exp-pair-title">Gradient boosting with fixed settings</h3><p class="exp-sub">The main table tunes boosting separately in each development fold. Here the settings are fixed across folds. Both tables use the same combined model for GSE90909. The whole-blood-only fit uses 145 profiles.</p></div></div>
    <div id="matched-boosting-table"></div></section>
   <details class="exp-classifiers"><summary>Exposure classification · 3 models</summary><p>Development holdouts only. Higher balanced accuracy is better.</p>
    <div id="classification-study-control"></div><div id="development-classification"></div></details>`;
  const retrospective = data.retrospective_evaluation;
  get('comparison-run-date').textContent = retrospective ?
   'GSE90909 comparison run · ' + new Intl.DateTimeFormat('en-GB',{day:'numeric',month:'short',year:'numeric',timeZone:'UTC'}).format(new Date(retrospective.run_date+'T00:00:00Z')) :
   'GSE90909 scores are unavailable in this saved version.';
  const visibleRows = data.regression.filter(visiblePredictor);
  const family = select(get('model-family-control'),'development-family','Model family',
   [['all','All model families'],...Array.from(new Set(visibleRows.map(r=>r.family))).map(f=>[f,f])], 'all');
  const familyLabel = family.parentElement;
  familyLabel.firstChild.replaceWith(el('span','Model family','exp-family-name'));
  let tunedAscending = true, pairAscending = true;
  function renderTuned() {
   const rows = visibleRows.filter(r => family.value === 'all' || r.family === family.value);
   get('model-count').textContent = rows.length+' of '+data.regression.length+' predictors shown' + (family.value === 'all' ? '' : ' · '+family.value+' family') + '. Models below 0.770 Gy on GSE90909 are hidden. The export includes all predictors.';
   comparisonTable(get('development-regression'),rows,'Model',tunedAscending,() => {
    tunedAscending = !tunedAscending; renderTuned(); get('development-regression').querySelector('.exp-dev-sort').focus();
   });
  }
  function renderPair() {
   const pair = data.matched_gradient_boosting;
   get('matched-boosting').hidden = !pair || !['all','Ensemble'].includes(family.value);
   if (get('matched-boosting').hidden) return;
   comparisonTable(get('matched-boosting-table'),pair.rows.filter(visiblePredictor),'Training variant',pairAscending,() => {
    pairAscending = !pairAscending; renderPair(); get('matched-boosting-table').querySelector('.exp-dev-sort').focus();
   });
  }
  family.onchange = () => {renderTuned(); renderPair();};
  renderTuned(); renderPair();
  const study = select(get('classification-study-control'), 'classification-study', 'Classification cohort',
   [['pooled','All three held-out studies'], ...studies.slice(0,3).map(s=>[s,s])], 'pooled');
  function classification() {
   table(get('development-classification'), ['Classifier','Balanced accuracy','Control specificity','Exposed sensitivity'],
    data.classification.filter(r=>r.study===study.value).map(r=>[r.label,...['balanced_accuracy','specificity','sensitivity'].map(k=>num(100*r[k],1)+'%')]));
  }
  study.onchange=classification;classification();
  get('development-download').onclick=()=>download({release_id:data.release_id,purpose:'dose_model_comparison',development_scores_include_final_test:data.final_test_included,retrospective_evaluation:data.retrospective_evaluation,eligible_profiles:data.eligible_profiles,whole_blood_profiles:data.core_profiles,wbc_profiles:data.wbc_profiles,held_out_profiles:data.held_out_profiles,held_out_donor_groups:data.held_out_donor_groups,folds:data.folds,development_run_date:data.development_run_date,primary_seed:data.primary_seed,benchmark_protocol:data.benchmark_protocol,regression:data.regression,matched_gradient_boosting:data.matched_gradient_boosting,classification:data.classification},'defender-development-comparison.json');
 }
 matchMedia('(max-width:580px)').addEventListener('change',()=>{if(clusterReady)drawClusters();});
 window.defenderDevelopment=Object.freeze({ensureClusters:()=>ensure('clusters'),ensureComparison:()=>ensure('comparison')});
})();
