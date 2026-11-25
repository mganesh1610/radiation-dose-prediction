'use strict';
const $=id=>document.getElementById(id);
let bootstrap, jobs=[], selection=null, pending=null, busy=false;
const labels={};
async function api(path,body){
 const response=await fetch(path,{method:body?'POST':'GET',headers:body?{'Content-Type':'application/json','X-Defender-Request':'1'}:{},body:body?JSON.stringify(body):undefined});
 if(!response.headers.get('content-type')?.includes('application/json'))throw new Error('Google sign-in may have expired. Reload this page to sign in again');
 const data=await response.json();if(!response.ok){const error=new Error(data.error||`Request failed (${response.status})`);error.status=response.status;throw error;}return data;
}
function message(text,error=false){$('feedback').textContent=text;$('feedback').className=error?'error':'';}
function el(tag,text){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;return n;}
function preset() {
 const method = bootstrap.methods.find(item => item.id === $('method').value);
 const heldOut = bootstrap.studies[$('study').value];
 const total = bootstrap.training_pool.total;
 $('split-fit').textContent = (total - heldOut) + ' profiles to fit';
 $('split-test').textContent = heldOut + ' held out';
 $('split-study').textContent = $('study').value;
 $('split-progress').max = total;
 $('split-progress').value = total - heldOut;
 $('preset').textContent = Object.keys(method.parameters).length ?
  'Fixed settings · ' + JSON.stringify(method.parameters) : 'Training-fold median dose';
}
function controls(){
 const own=Math.max(0,bootstrap.access.job_limit-jobs.length),shared=Math.max(0,bootstrap.limits.jobs-bootstrap.jobs_used);
 const allowance = own+' of '+bootstrap.access.job_limit;
 const role = bootstrap.access.role === 'reviewer' ? 'Reviewer' : 'Researcher';
 $('allowance').textContent = allowance;
 $('allowance-role').textContent = role+' account · '+bootstrap.month+' (UTC)';
 $('shared-allowance').textContent = shared+' shared project runs available this month.';
 $('mobile-allowance').textContent = allowance+' monthly runs available · '+shared+' across the project.';
 $('request-usage').textContent=`Monthly requests · ${bootstrap.access.requests_used}/${bootstrap.access.request_limit} for you · ${bootstrap.requests_used}/${bootstrap.limits.api_requests} for the project. Cancelled and failed runs use the allowance.`;
 $('train').disabled=busy||!bootstrap.training_enabled||(!pending&&(own===0||shared===0));
 $('train').setAttribute('aria-busy',String(busy));
 $('method').disabled=$('study').disabled=busy||Boolean(pending);
 const w=bootstrap.worker,fast=w.demo_until*1000>Date.now(),minutes=fast?1:10;
 $('pickup').textContent=`The Mac checks the queue about every ${minutes} minute${minutes===1?'':'s'} while connected. Use Refresh status for updates.`;
 $('worker').textContent=w.checked_at?`Last worker check-in · ${new Date(w.checked_at*1000).toLocaleString()} · ${w.status.replaceAll('_',' ')}`:'Waiting for the first worker check-in.';
 if(!bootstrap.training_enabled)message('New training is paused. Existing results remain available.');
}
async function refresh(reload=true){
 $('refresh').disabled=true;
 try{
  if(reload){bootstrap=await api('/api/bootstrap');jobs=bootstrap.jobs;}
  controls();
  const target=jobs.slice(-12);const states=[];
  // Sequential requests keep the per-user request budget predictable.
  for(const id of target)states.push(await api('/api/jobs/'+id));
  $('jobs').replaceChildren();
  $('jobs-table-wrap').hidden = !states.length;
  $('jobs-status').textContent = states.length ? '' : 'No training runs this month.';
  for(const s of states.reverse()){
   const row=el('tr'),id=el('td',s.job_id.slice(-10));id.title=s.job_id;id.className='mono';
   const actions=el('td');
   if(s.status==='SUCCEEDED'){const b=el('button','View results');b.className='secondary';b.onclick=()=>showResult(s.job_id);actions.append(b);}
   else if(!['CANCELLED','FAILED'].includes(s.status)){const b=el('button','Cancel');b.className='secondary';b.onclick=async()=>{try{await api('/api/jobs/'+s.job_id+'/cancel',{});message('Cancellation recorded. Any unfinished publication will be rejected.');await refresh();}catch(e){message(e.message,true);}};actions.append(b);}
   row.append(id,el('td',s.method==='gradient_boosting'&&s.job_version===1?window.defenderModelNames.gradient_boosting:(labels[s.method]||s.method)),el('td',s.status+(s.stage?' · '+s.stage:'')),actions);$('jobs').append(row);
  }
 }catch(e){message(e.message,true);$('jobs-status').textContent='Unable to refresh your runs.';}finally{$('refresh').disabled=false;}
}
function scatter(rows){
 const svg=$('scatter');svg.replaceChildren();const ns='http://www.w3.org/2000/svg';
 const values=rows.flatMap(r=>[r.observed_gy,r.predicted_gy]);const low=Math.min(0,...values),high=Math.max(1,...values)*1.1;
 const narrow=matchMedia('(max-width:580px)').matches,W=narrow?360:700,L=narrow?44:60,R=W-25,B=300;
 svg.setAttribute('viewBox',`0 0 ${W} 350`);
 const x=v=>L+(v-low)/(high-low)*(R-L),y=v=>B-(v-low)/(high-low)*270;
 function node(tag,attrs,text){const n=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;svg.append(n);return n;}
 node('title',{},'Observed versus predicted dose in the held-out development study');
 for(let i=0;i<=4;i++){const v=low+(high-low)*i/4;node('line',{x1:L,y1:y(v),x2:R,y2:y(v),stroke:'#e1e8e3'});node('text',{x:L-10,y:y(v)+4,'text-anchor':'end'},v.toFixed(1));node('text',{x:x(v),y:319,'text-anchor':'middle'},v.toFixed(1));}
 node('line',{x1:x(low),y1:y(low),x2:x(high),y2:y(high),stroke:'#8da397','stroke-dasharray':'5 5'});
 for(const r of rows){const n=node('circle',{cx:x(r.observed_gy),cy:y(r.predicted_gy),r:4,fill:'#24775b',opacity:0.65});const t=document.createElementNS(ns,'title');t.textContent=`${r.sample_id}: observed ${r.observed_gy} Gy, predicted ${r.predicted_gy.toFixed(3)} Gy`;n.append(t);}
 node('text',{x:W/2,y:343,'text-anchor':'middle'},'Observed dose (Gy)');node('text',{x:L,y:15},'Predicted dose (Gy)');
}
async function showResult(id){try{
 selection=await api('/api/jobs/'+id+'/results');const r=selection.results;if(!r){message('Results are not ready yet.');return;}
 $('results').hidden=false;$('result-title').textContent=(r.method==='gradient_boosting'&&!r.refit_samples?window.defenderModelNames.gradient_boosting:(labels[r.method]||r.method))+' · development results';
 $('result-context').textContent=`${r.holdout_study} · ${r.train_samples} profiles to fit · ${r.validation_samples} held out for scoring.${r.refit_samples ? " Saved model refitted on all "+r.refit_samples+" profiles." : ""}`;
 $('metrics').replaceChildren();for(const [key,label]of [['mae','Donor-balanced MAE'],['rmse','Donor-balanced RMSE'],['bias','Donor-balanced bias']]){const block=el('div');block.append(el('span',label),el('strong',r.metrics[key].toFixed(3)+' Gy'));$('metrics').append(block);}
 scatter(r.predictions);$('predictions').replaceChildren();for(const row of r.predictions){const tr=el('tr');for(const v of [row.sample_id,row.group_id,row.observed_gy.toFixed(3),row.predicted_gy.toFixed(3)])tr.append(el('td',v));$('predictions').append(tr);}
 $('provenance').textContent=JSON.stringify({job_id:id,release:selection.release_id,model_manifest:selection.model_manifest,checks:r.checks,runtime:r.runtime,purpose:r.purpose},null,2);
 $('results').scrollIntoView({behavior:'auto',block:'start'});
 $('result-title').focus({preventScroll:true});
}catch(e){message(e.message,true);}}
matchMedia('(max-width:580px)').addEventListener('change',()=>{if(selection?.results)scatter(selection.results.predictions);});
$('download').onclick=()=>{if(!selection)return;const url=URL.createObjectURL(new Blob([JSON.stringify(selection,null,2)],{type:'application/json'}));const a=el('a');a.href=url;a.download=selection.job_id+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
$('refresh').onclick=()=>bootstrap ? refresh() : start();
$('train-form').onsubmit=async event=>{event.preventDefault();if(busy)return;busy=true;$('train').disabled=true;
 try{
  // Preserve pending ID and body on network failures. A retry cannot create a second job.
  if(!pending){pending={job_id:'web_'+bootstrap.month.replace('-','')+'_'+crypto.randomUUID().replaceAll('-',''),method:$('method').value,holdout_study:$('study').value};sessionStorage.setItem('defenderPending',JSON.stringify(pending));}
  const result=await api('/api/jobs',pending);if(!jobs.includes(result.job_id))jobs.push(result.job_id);pending=null;sessionStorage.removeItem('defenderPending');message('Job queued in Google Cloud. Your Mac will pick it up on its next check.');await refresh();
 }catch(e){if(e.status===400){pending=null;sessionStorage.removeItem('defenderPending');}message(e.message+(pending?' Retry reuses the same request.':''),true);}finally{busy=false;controls();}
};
async function start(){try{
 bootstrap=await api('/api/bootstrap');jobs=bootstrap.jobs;
 try{pending=JSON.parse(sessionStorage.getItem('defenderPending')||'null');}catch{pending=null;sessionStorage.removeItem('defenderPending');}
 if(pending&&!pending.job_id?.startsWith('web_'+bootstrap.month.replace('-','')+'_')){pending=null;sessionStorage.removeItem('defenderPending');}
 $('method').replaceChildren();for(const m of bootstrap.methods){labels[m.id]=m.id==='gradient_boosting'?window.defenderModelNames.gradient_boosting_wb_wbc:(window.defenderModelNames[m.id]||m.label);const o=el('option',labels[m.id]);o.value=m.id;$('method').append(o);}$('method').value='elastic_net';$('method').onchange=preset;$('study').onchange=preset;preset();$('train').disabled=false;
 if(pending){$('method').value=pending.method;$('study').value=pending.holdout_study;preset();message('A previous submission needs checking. Train model will retry the same request.');}
 await refresh(false);
}catch(e){message(e.message,true);$('worker').textContent='Worker status unavailable.';$('jobs-status').textContent='Unable to load runs. Use Refresh status to retry.';}}

let trainingStarted=false;window.defenderTraining={ensure(){if(!trainingStarted){trainingStarted=true;start();}}};
