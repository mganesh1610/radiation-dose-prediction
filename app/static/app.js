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
function preset(){const m=bootstrap.methods.find(m=>m.id===$('method').value);$('preset').textContent=Object.keys(m.parameters).length?JSON.stringify(m.parameters):'Training-fold median dose · no fitted gene features';}
function controls(){
 const own=Math.max(0,bootstrap.access.job_limit-jobs.length),shared=Math.max(0,bootstrap.limits.jobs-bootstrap.jobs_used);
 $('allowance').textContent=`${bootstrap.access.role==='reviewer'?'Reviewer demo':'Research workspace'} · ${own} of ${bootstrap.access.job_limit} runs remaining for ${bootstrap.month} (UTC). Shared project: ${shared} runs remaining. Cancelled and failed runs count toward these limits.`;
 $('request-usage').textContent=`Requests: ${bootstrap.access.requests_used}/${bootstrap.access.request_limit} for this account · ${bootstrap.requests_used}/${bootstrap.limits.api_requests} across the project this month. Refresh checks only your runs.`;
 $('train').disabled=busy||!bootstrap.training_enabled||(!pending&&(own===0||shared===0));
 $('train').setAttribute('aria-busy',String(busy));
 $('method').disabled=$('study').disabled=busy||Boolean(pending);
 const w=bootstrap.worker,fast=w.demo_until*1000>Date.now(),minutes=fast?1:10;
 $('pickup').textContent=`The project Mac checks for work about every ${minutes} minute${minutes===1?'':'s'} while awake, signed in and connected.${fast?' Demo pickup mode ends '+new Date(w.demo_until*1000).toLocaleTimeString()+'.':''} Use Refresh status to check progress; this page does not poll automatically.`;
 $('worker').textContent=w.checked_at?`Mac last checked in ${new Date(w.checked_at*1000).toLocaleString()} · ${w.status.replaceAll('_',' ')}. A past check-in does not guarantee current availability.`:'Mac has not checked in yet. Jobs can wait in the cloud.';
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
  if(!states.length){const row=el('tr'),cell=el('td','No runs yet. Choose a method and start your first engineering check.');cell.colSpan=4;row.append(cell);$('jobs').append(row);}
  for(const s of states.reverse()){
   const row=el('tr'),id=el('td',s.job_id.slice(-10));id.title=s.job_id;id.className='mono';
   const actions=el('td');
   if(s.status==='SUCCEEDED'){const b=el('button','View results');b.className='secondary';b.onclick=()=>showResult(s.job_id);actions.append(b);}
   else if(!['CANCELLED','FAILED'].includes(s.status)){const b=el('button','Cancel');b.className='secondary';b.onclick=async()=>{try{await api('/api/jobs/'+s.job_id+'/cancel',{});message('Cancellation recorded. Any unfinished publication will be rejected.');await refresh();}catch(e){message(e.message,true);}};actions.append(b);}
   row.append(id,el('td',labels[s.method]||s.method),el('td',s.status+(s.stage?' · '+s.stage:'')),actions);$('jobs').append(row);
  }
 }catch(e){message(e.message,true);}finally{$('refresh').disabled=false;}
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
 $('results').hidden=false;$('result-title').textContent=(labels[r.method]||r.method)+' · development results';
 $('result-context').textContent=`Held out ${r.holdout_study} · ${r.train_samples} training profiles · ${r.validation_samples} validation profiles. Engineering run; not a tuned comparison or a published predictor.`;
 $('metrics').replaceChildren();for(const [key,label]of [['mae','Donor-balanced MAE'],['rmse','Donor-balanced RMSE'],['bias','Donor-balanced bias']]){const block=el('div');block.append(el('span',label),el('strong',r.metrics[key].toFixed(3)+' Gy'));$('metrics').append(block);}
 scatter(r.predictions);$('predictions').replaceChildren();for(const row of r.predictions){const tr=el('tr');for(const v of [row.sample_id,row.group_id,row.observed_gy.toFixed(3),row.predicted_gy.toFixed(3)])tr.append(el('td',v));$('predictions').append(tr);}
 $('provenance').textContent=JSON.stringify({job_id:id,release:bootstrap.release_id,model_manifest:selection.model_manifest,checks:r.checks,runtime:r.runtime,purpose:r.purpose},null,2);
 $('results').scrollIntoView({behavior:'auto',block:'start'});
 $('result-title').focus({preventScroll:true});
}catch(e){message(e.message,true);}}
matchMedia('(max-width:580px)').addEventListener('change',()=>{if(selection?.results)scatter(selection.results.predictions);});
$('download').onclick=()=>{if(!selection)return;const url=URL.createObjectURL(new Blob([JSON.stringify(selection,null,2)],{type:'application/json'}));const a=el('a');a.href=url;a.download=selection.job_id+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
$('refresh').onclick=()=>refresh();
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
 $('method').replaceChildren();for(const m of bootstrap.methods){labels[m.id]=m.label;const o=el('option',m.label);o.value=m.id;$('method').append(o);}$('method').value='elastic_net';$('method').onchange=preset;preset();$('train').disabled=false;
 if(pending){$('method').value=pending.method;$('study').value=pending.holdout_study;preset();message('A previous submission needs checking. Train model will retry the same request.');}
 await refresh(false);
}catch(e){message(e.message,true);$('worker').textContent='Worker status unavailable.';$('jobs').textContent='Unable to load runs.';}}

let trainingStarted=false;window.defenderTraining={ensure(){if(!trainingStarted){trainingStarted=true;start();}}};
