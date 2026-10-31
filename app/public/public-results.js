'use strict';
(()=>{
const $=id=>document.getElementById(id);
const el=(tag,text)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;return n;};
let loaded=false, loading=false, selected=null;
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

function show(run){
 selected=run;const r=run.result;
 $('results').hidden=false;$('result-title').textContent=run.label+' · development results';
 $('result-context').textContent=`Held out ${r.holdout_study} · ${r.train_samples} training profiles · ${r.validation_samples} validation profiles. Engineering run; not a tuned comparison or a published predictor.`;
 $('metrics').replaceChildren();
 for(const [key,label]of [['mae','Donor-balanced MAE'],['rmse','Donor-balanced RMSE'],['bias','Donor-balanced bias']]){const block=el('div');block.append(el('span',label),el('strong',r.metrics[key].toFixed(3)+' Gy'));$('metrics').append(block);}
 scatter(r.predictions);$('predictions').replaceChildren();
 for(const row of r.predictions){const tr=el('tr');for(const v of [row.sample_id,row.group_id,row.observed_gy.toFixed(3),row.predicted_gy.toFixed(3)])tr.append(el('td',v));$('predictions').append(tr);}
 $('provenance').textContent=JSON.stringify({release_id:run.release_id,purpose:r.purpose,method:r.method,held_out_study:r.holdout_study,methods:run.methods,limitations:run.limitations},null,2);
}
async function ensure(){
 if(loaded||loading)return;loading=true;$('published-status').textContent='Loading published results…';$('published-runs').replaceChildren();
 try{const data=await window.defenderPublic.read('/published/experiments');
 for(const run of data.runs){const button=el('button',run.label+' · '+run.result.holdout_study);button.className='secondary';button.onclick=()=>{show(run);$('results').scrollIntoView({behavior:'auto',block:'start'});};$('published-runs').append(button);}
 if(data.runs.length)show(data.runs[0]);
 $('published-status').textContent=`${data.runs.length} published engineering run. This snapshot updates when reviewed results are republished.`;loaded=true;
 }catch(error){$('published-status').textContent=error.message;const retry=el('button','Retry loading results');retry.className='secondary';retry.onclick=ensure;$('published-runs').append(retry);}finally{loading=false;}
}
$('download').onclick=()=>{if(!selected)return;const url=URL.createObjectURL(new Blob([JSON.stringify(selected,null,2)],{type:'application/json'}));const a=el('a');a.href=url;a.download='defender-'+selected.run_id+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
matchMedia('(max-width:580px)').addEventListener('change',()=>{if(selected)scatter(selected.result.predictions);});
window.defenderTraining=Object.freeze({ensure});
})();
