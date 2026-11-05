'use strict';
/* Presentation adapter only. The existing explorer owns requests, caching, settings and scientific plots. */
(()=>{
 const get=id=>document.getElementById(id),page=get('page-analysis');
 if(!page||!get('workspace-plot')||!get('analysis-table'))return;
 const make=(tag,text)=>{const node=document.createElement(tag);if(text!==undefined)node.textContent=text;return node;};
 const styles={GSE102971:{color:'#77d8a9',shape:'circle'},GSE8917:{color:'#edb45e',shape:'diamond'},GSE65292:{color:'#9cacf8',shape:'square'}};
 const state={result:null,view:'pca',perspective:'spatial',selection:-1,highlight:null,epoch:0,unavailable:false,expanded:false};
 let scene=null,scenePromise=null,baseCaption='',priorFocus=null;
 const ratio=value=>(value*100).toFixed(1)+'%',numeric=value=>Number.isFinite(value)?value.toFixed(3):'Not recorded';
 const iconPaths={file:'M5 2h9l5 5v15H5Zm9 0v6h5M8 12h8m-8 4h8',download:'M12 3v12m-5-5 5 5 5-5M4 15v6h16v-6',left:'m15 6-6 6 6 6',right:'m9 6 6 6-6 6',close:'m5 5 14 14M19 5 5 19',reset:'M3 10a9 9 0 1 1 1 8M3 3v7h7',rotate:'M12 3v18m-3-3 3 3 3-3m-6-12 3-3 3 3M4 8c-4 6 2 11 8 10s13-8 8-11',expand:'M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5'};
 function icon(name){const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox','0 0 24 24');svg.setAttribute('fill','none');svg.setAttribute('stroke','currentColor');svg.setAttribute('stroke-width','1.6');svg.setAttribute('stroke-linecap','round');svg.setAttribute('stroke-linejoin','round');svg.setAttribute('aria-hidden','true');const path=document.createElementNS(svg.namespaceURI,'path');path.setAttribute('d',iconPaths[name]);svg.append(path);return svg;}
 for(const [id,name]of [['workspace-previous','left'],['workspace-next','right'],['workspace-clear','close'],['workspace-reset','reset'],['workspace-rotate','rotate'],['workspace-expand','expand'],['workspace-methods-close','close']])get(id).replaceChildren(icon(name));
 get('workspace-methods').prepend(icon('file'));get('workspace-export').prepend(icon('download'));
 const brand=document.querySelector('body>aside .brand');
 if(brand){const mark=icon('file');mark.classList.add('workspace-brand-icon');mark.querySelector('path').setAttribute('d','M12 2 21 5v7c0 5-4 8-9 11C7 20 3 17 3 12V5ZM7 14l4-5 3 3 4-5');brand.prepend(mark);}
 document.body.classList.add('workspace-enhanced');
 const details=get('analysis-data-details'),methods=get('workspace-methods-details');
 get('workspace-methods-content').append(methods);methods.open=true;
 get('core-analysis').append(details);
 const tableSearch=make('label');tableSearch.className='workspace-table-search';
 const search=make('input');search.type='search';search.placeholder='Search plotted data';search.setAttribute('aria-label','Search plotted data');
 const rowCount=make('span');rowCount.setAttribute('role','status');tableSearch.append(search,rowCount);details.querySelector('summary').after(tableSearch);
 function searchRows(){
  const rows=[...get('analysis-table').querySelectorAll('tbody tr')],query=search.value.trim().toLowerCase();let count=0;
  for(const row of rows){row.hidden=Boolean(query)&&!row.textContent.toLowerCase().includes(query);if(!row.hidden)count++;}
  rowCount.textContent=`${count.toLocaleString()} of ${rows.length.toLocaleString()} rows`;
 }
 search.addEventListener('input',searchRows);
 function signal(text){get('point-detail').textContent=text;}
 function isPcaVisible(){return location.hash==='#analysis'&&state.view==='pca'&&Boolean(state.result);}
 function syncRoute(){
  const route=location.hash.slice(1)||'overview';document.body.dataset.workspace=['overview','datasets','analysis','experiments','predictions','assistant'].includes(route)?route:'overview';
  if(route!=='analysis'){scene?.setActive(false);hideTooltip();if(state.expanded)expand(false);}
  else if(state.result)showPerspective();
 }
 window.addEventListener('hashchange',syncRoute);syncRoute();
 function hideTooltip(){get('workspace-tooltip').hidden=true;}
 function hoverPoint(hit){
  if(!hit||!state.result)return hideTooltip();
  const sample=state.result.samples[hit.index];if(!sample)return;
  const tooltip=get('workspace-tooltip');tooltip.replaceChildren(make('strong',sample.sample_id),make('div',`${sample.study_id} · ${sample.dose_gy} Gy`));tooltip.hidden=false;
  const r=get('workspace-plot').getBoundingClientRect();
  tooltip.style.left=Math.max(7,Math.min(hit.x-r.left+12,r.width-tooltip.offsetWidth-9))+'px';
  tooltip.style.top=Math.max(7,Math.min(hit.y-r.top+12,r.height-tooltip.offsetHeight-45))+'px';
 }
 function rotationChanged(rotating){get('workspace-rotate').setAttribute('aria-pressed',String(rotating));get('workspace-rotate').setAttribute('aria-label',rotating?'Pause automatic rotation':'Start automatic rotation');}
 function fallback(){
  state.unavailable=true;state.perspective='flat';scene?.setActive(false);get('workspace-spatial').replaceChildren();
  showPerspective();signal('The spatial view is unavailable in this browser. The measured 2D chart and data remain available.');
 }
 function loadScene(){
  if(scenePromise)return scenePromise;
  scenePromise=import('/static/analysis-scene.mjs').then(async module=>{
   await document.fonts.ready;
   scene=module.createAnalysisScene(get('workspace-spatial'),{onSelect:selectProfile,onHover:hoverPoint,onRotation:rotationChanged,onUnavailable:fallback});
   return scene;
  }).catch(()=>{fallback();return null;});
  return scenePromise;
 }
 function showPerspective(){
  const pca=isPcaVisible(),spatial=pca&&state.perspective==='spatial'&&!state.unavailable;
  get('workspace-dimensions').hidden=state.view!=='pca'||!state.result;
  for(const button of document.querySelectorAll('[data-perspective]')){
   button.setAttribute('aria-pressed',String(button.dataset.perspective===state.perspective));
   button.disabled=button.dataset.perspective==='spatial'&&state.unavailable;
  }
  get('workspace-spatial').hidden=!spatial;get('chart-scroll').hidden=spatial&&Boolean(scene);
  get('workspace-axis-key').hidden=!spatial;get('workspace-spatial-hint').hidden=!spatial;get('workspace-spatial-tools').hidden=!spatial;
  get('workspace-export-png').hidden=!spatial;get('workspace-export-png').disabled=!spatial||!scene;
  const epoch=++state.epoch;
  if(!spatial){scene?.setActive(false);if(state.view==='pca'&&baseCaption)get('analysis-caption').textContent=baseCaption;return;}
  const result=state.result;
  get('workspace-axis-key').textContent=`Floor: PC1 ${ratio(result.pca.explained_variance_ratio[0])} × PC2 ${ratio(result.pca.explained_variance_ratio[1])} · Height: recorded dose (Gy)`;
  get('workspace-spatial').setAttribute('aria-busy',String(!scene));
  get('workspace-spatial-hint').textContent=scene?'Drag to rotate · Scroll to zoom · Select a profile':'Preparing the spatial view. The measured 2D chart is available.';
  loadScene().then(renderer=>{
   if(!renderer||epoch!==state.epoch||!isPcaVisible()||state.perspective!=='spatial')return;
   try{renderer.setResult(result);renderer.select(state.selection);renderer.highlight(state.highlight);renderer.setActive(true);get('chart-scroll').hidden=true;get('workspace-spatial').setAttribute('aria-busy','false');get('workspace-spatial-hint').textContent='Drag to rotate · Scroll to zoom · Select a profile';get('analysis-caption').textContent='One point per measured profile. Height shows recorded dose, not a third principal component. Repeated donor profiles are not independent observations.';get('workspace-export-png').disabled=false;}
   catch{fallback();}
  });
 }
 function paintNativeChart(){
  const data=state.result;if(!data)return;
  if(state.view==='pca'){
   const nodes=[...get('analysis-chart').querySelectorAll('circle[data-inspect]')];
   nodes.forEach((node,index)=>{const sample=data.samples[index],style=styles[sample?.study_id];if(style)node.setAttribute('fill',style.color);node.setAttribute('stroke','#10261f');node.setAttribute('opacity',!state.highlight||sample?.study_id===state.highlight?'0.95':'0.15');});
  }else if(state.view==='heatmap'){
   const samples=new Map(data.samples.map(s=>[s.sample_id,s.study_id]));
   get('analysis-chart').querySelectorAll('[data-sample]').forEach(node=>node.setAttribute('opacity',!state.highlight||samples.get(node.dataset.sample)===state.highlight?'1':'0.18'));
  }
 }
 function composition(){
  const counts=new Map();for(const sample of state.result.samples)counts.set(sample.study_id,(counts.get(sample.study_id)||0)+1);
  const host=get('workspace-studies');host.replaceChildren();
  for(const [study,count]of counts){
   const style=styles[study]||{color:'#a3b29b',shape:'circle'},button=make('button');button.type='button';button.className='workspace-study';button.dataset.study=study;button.disabled=!['pca','heatmap'].includes(state.view);
   button.classList.toggle('dimmed',Boolean(state.highlight)&&state.highlight!==study);button.setAttribute('aria-pressed',String(state.highlight===study));button.setAttribute('aria-label',`Highlight ${study}, ${count} profiles`);
   button.style.setProperty('--study-color',style.color);button.style.setProperty('--study-share',`${count/state.result.samples.length*100}%`);
   const mark=make('span');mark.className='workspace-study-mark '+style.shape;const bar=make('span');bar.className='workspace-study-bar';bar.append(make('i'));const total=make('span',count);total.className='workspace-study-count';button.append(mark,make('span',study),bar,total);
   button.onclick=()=>{state.highlight=state.highlight===study?null:study;scene?.highlight(state.highlight);paintNativeChart();composition();signal(state.highlight?`Highlighting ${state.highlight}. All ${state.result.samples.length} measured profiles remain in the plot and export.`:'All measured profiles are shown.');[...host.children].find(b=>b.dataset.study===study)?.focus();};host.append(button);
  }
 }
 function profileCard(){
  const host=get('workspace-profile-content'),selected=state.selection>=0;get('workspace-clear').hidden=!selected;
  get('workspace-profile-heading').textContent=selected?'Selected profile':'A closer look';host.replaceChildren();
  if(!selected){host.append(make('p','Select a profile to inspect its recorded dose and measured coordinates.'));return;}
  const sample=state.result.samples[state.selection],scores=state.result.pca.scores[state.selection],title=make('div');title.className='workspace-profile-accession';
  const mark=make('span');mark.className='workspace-study-mark '+(styles[sample.study_id]?.shape||'circle');mark.style.setProperty('--study-color',styles[sample.study_id]?.color||'#a3b29b');title.append(mark,make('span',sample.sample_id));
  const facts=make('dl');
  for(const [label,value]of [['Recorded dose',`${sample.dose_gy} Gy`],['Study',sample.study_id],['PC1',numeric(scores[0])],['PC2',numeric(scores[1])],['Donor group',sample.group_id||'Not recorded'],['Missing genes',state.result.qc.missing_by_sample[state.selection]]]){
   const group=make('div'),description=make('dd',value);if(label==='Recorded dose')description.className='recorded-dose';group.append(make('dt',label),description);facts.append(group);
  }
  host.append(title,facts);
 }
 function selectProfile(index){
  if(!state.result||!Number.isInteger(index)||!state.result.samples[index])return;
  state.selection=index;scene?.select(index);profileCard();
  const sample=state.result.samples[index],p=state.result.pca.scores[index];
  signal(`${sample.sample_id} · ${sample.study_id} · ${sample.group_id} · ${sample.dose_gy} Gy · PC1 ${numeric(p[0])}, PC2 ${numeric(p[1])}`);
  const card=get('workspace-profile'),form=get('analysis-form'),gap=card.getBoundingClientRect().bottom-form.getBoundingClientRect().bottom+12;
  if(form.scrollHeight>form.clientHeight&&gap>0)form.scrollTop+=gap;
 }
 function update({result,view}){
  const changed=result!==state.result||view!==state.view;
  if(result!==state.result){state.selection=-1;state.highlight=null;get('analysis-form').scrollTop=0;}
  state.result=result;state.view=view;page.dataset.workspaceView=view;
  const core=view!=='clustering',ready=core&&Boolean(result);
  get('workspace-summary').hidden=!ready;get('workspace-export').disabled=!ready;details.hidden=!ready;
  get('workspace-cohorts').hidden=!ready;get('workspace-profile').hidden=!ready;
  if(changed)search.value='';
  if(result){
   get('workspace-count').textContent=result.samples.length.toLocaleString();get('workspace-study-count').textContent=new Set(result.samples.map(s=>s.study_id)).size;
   get('workspace-gene-count').textContent=result.genes_used.toLocaleString();get('workspace-variance').textContent=ratio(result.pca.explained_variance_ratio.slice(0,2).reduce((sum,value)=>sum+value,0));
   composition();profileCard();paintNativeChart();searchRows();
   if(view==='pca')baseCaption=get('analysis-caption').textContent;
  }
  showPerspective();
 }
 window.defenderAnalysisView=Object.freeze({update,selectProfile});
 for(const button of document.querySelectorAll('[data-perspective]'))button.onclick=()=>{state.perspective=button.dataset.perspective;hideTooltip();showPerspective();};
 get('workspace-previous').onclick=()=>{if(state.result)selectProfile(state.selection<0?state.result.samples.length-1:(state.selection-1+state.result.samples.length)%state.result.samples.length);};
 get('workspace-next').onclick=()=>{if(state.result)selectProfile((state.selection+1)%state.result.samples.length);};
 get('workspace-clear').onclick=()=>{state.selection=-1;scene?.select(-1);profileCard();signal('Select a profile to inspect its measured values.');get('workspace-next').focus();};
 get('workspace-reset').onclick=()=>scene?.reset();
 get('workspace-rotate').onclick=()=>{if(scene&&!scene.rotate(get('workspace-rotate').getAttribute('aria-pressed')!=='true')&&matchMedia('(prefers-reduced-motion:reduce)').matches)signal('Automatic rotation is off for reduced motion. You can still rotate by dragging.');};
 function expand(value){
  state.expanded=value;const plot=get('workspace-plot');plot.classList.toggle('expanded',value);get('workspace-expand').setAttribute('aria-pressed',String(value));get('workspace-expand').setAttribute('aria-label',value?'Close expanded visualization':'Expand visualization');document.body.style.overflow=value?'hidden':'';
  if(value){priorFocus=document.activeElement;plot.setAttribute('role','dialog');plot.setAttribute('aria-modal','true');plot.setAttribute('aria-label','Expanded measured visualization');get('workspace-expand').focus({preventScroll:true});}
  else{plot.removeAttribute('role');plot.removeAttribute('aria-modal');plot.removeAttribute('aria-label');priorFocus?.focus({preventScroll:true});}
  // A focused absolute toolbar must not scroll the clipped chart when its size changes.
  plot.scrollTop=0;plot.scrollLeft=0;hideTooltip();requestAnimationFrame(()=>{plot.scrollTop=0;plot.scrollLeft=0;scene?.resize();});
 }
 get('workspace-expand').onclick=()=>expand(!state.expanded);
 function closeExports(){get('workspace-export-options').hidden=true;get('workspace-export').setAttribute('aria-expanded','false');}
 get('workspace-export').onclick=()=>{const open=get('workspace-export-options').hidden;get('workspace-export-options').hidden=!open;get('workspace-export').setAttribute('aria-expanded',String(open));if(open)get('workspace-export-options').querySelector('button')?.focus();};
 function download(blob,name){const link=make('a'),url=URL.createObjectURL(blob);link.href=url;link.download=name;document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1500);}
 function csvCell(value){let text=String(value);if(/^[=+@-]/.test(text)&&!/^[-+]?\d*\.?\d+(?:e[-+]?\d+)?$/i.test(text))text="'"+text;return '"'+text.replaceAll('"','""')+'"';}
 get('workspace-export-options').onclick=async event=>{
  const button=event.target.closest('[data-workspace-export]');if(!button||!state.result)return;
  const name=`defender-${state.result.analysis_id}-${state.view}`;
  try{
   if(button.dataset.workspaceExport==='snapshot')get('download-analysis').click();
   if(button.dataset.workspaceExport==='csv'){
    const rows=[...get('analysis-table').rows].map(row=>[...row.cells].map(cell=>csvCell(cell.textContent)).join(','));download(new Blob(['\uFEFF'+rows.join('\r\n')],{type:'text/csv;charset=utf-8'}),name+'.csv');
   }
   if(button.dataset.workspaceExport==='png'&&scene)download(await scene.image(),name+'-recorded-dose.png');
  }catch{signal('The export could not be created. Please try again.');}
  closeExports();get('workspace-export').focus();
 };
 get('workspace-methods').onclick=()=>get('workspace-methods-dialog').showModal();get('workspace-methods-close').onclick=()=>get('workspace-methods-dialog').close();
 document.addEventListener('click',event=>{if(!event.target.closest('.workspace-export'))closeExports();});
 document.addEventListener('keydown',event=>{
  if(event.key==='Escape'){if(state.expanded)expand(false);if(!get('workspace-export-options').hidden){closeExports();get('workspace-export').focus();}hideTooltip();}
  if(event.key==='Tab'&&state.expanded){const items=[...get('workspace-plot').querySelectorAll('button,canvas[tabindex]')].filter(node=>node.getClientRects().length),first=items[0],last=items.at(-1);if(event.shiftKey&&document.activeElement===first){event.preventDefault();last?.focus();}else if(!event.shiftKey&&document.activeElement===last){event.preventDefault();first?.focus();}}
 });
 update({result:null,view:'pca'});
})();
