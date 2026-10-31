'use strict';
// Deterministic approved-record lookup. It neither generates doses nor executes prompts.
(function(root){
 function answer(question,data){
  const q=String(question).trim().toLowerCase(),sources=[],unknown={answer:'Not recorded in the approved project records. Try a listed topic or an exact profile ID with “dose”, “time” or “study”.',sources:[],status:'not_recorded'};
  if(!q||q.length>500)return unknown;
  if(/clinical|diagnos|treatment|safe dose|patient/.test(q)){const f=data.facts.find(f=>f.id==='clinical');return {answer:f.answer,sources:f.sources,status:'recorded'};}
  const ids=[...new Set((q.match(/gsm\d+/g)||[]).map(i=>i.toUpperCase()))];
  if(ids.length){
   if(ids.length>3)return {answer:'Please look up at most three profile IDs at once.',sources:[],status:'not_recorded'};
   const rows=ids.map(id=>data.records.find(r=>r.profile_id===id));
   if(rows.some(r=>!r))return {answer:'Not recorded: at least one requested profile ID is absent from this approved development release. No dose was inferred.',sources:[],status:'not_recorded'};
   let key,label,unit;
   if(/dose|gray|\bgy\b/.test(q)){key='dose_gy';label='recorded exposure label';unit=' Gy (not a model prediction)';}
   else if(/time|hour|timing/.test(q)){key='time_hours';label='recorded time after exposure';unit=' hours';}
   else if(/study|cohort|source/.test(q)){key='study';label='source study';unit='';}else return unknown;
   return {answer:rows.map(r=>`${r.profile_id}: ${label} ${r[key]===null||r[key]===undefined?'not recorded':r[key]+unit}.`).join(' '),sources:rows.map(r=>({label:r.profile_id+' · GEO',url:'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc='+r.profile_id})),status:'recorded'};
  }
  if(/\bmae\b|\brmse\b|\bbias\b/.test(q)&&!/study bias/.test(q)){
   if(!/final|test/.test(q))return {answer:'Please specify “final test” and the model name. Development, selection and final-test errors use different evaluations.',sources:[{label:'Model comparisons',url:'#experiments'}],status:'clarification'};
   const ids={a:/\bmodel a\b/,b:/\bmodel b\b/,c:/\bmodel c\b/,d:/\bmodel d\b/,elastic_net:/elastic net/,pls:/\bpls\b|partial least/,gradient_boosting:/gradient boosting/,median_null:/median/};
   const match=Object.entries(ids).filter(([,r])=>r.test(q));if(match.length!==1)return {answer:'Please name one frozen model: A, B, C, D, Elastic Net, PLS, gradient boosting or median.',sources:[],status:'clarification'};
   const id=match[0][0].length===1?match[0][0].toUpperCase():match[0][0];const model=data.final_models.find(r=>r.id===id);if(!model)return unknown;
   const score=model.scores.find(s=>s.seed_role==='primary')||model.scores[0],metric=/rmse/.test(q)?'donor_rmse':/bias/.test(q)?'donor_bias':'donor_mae',label=metric.replace('donor_','').toUpperCase();
   return {answer:`${model.name}: final-test donor-balanced ${label} = ${score[metric].toFixed(3)} Gy, primary frozen model. Test: GSE90909, 52 profiles, 12 donor groups. This score is not an individual prediction interval or a deployment recommendation.`,sources:[{label:'Frozen model card',url:'?model='+id+'#experiments'}],status:'recorded'};
  }
  const matches=data.facts.map(f=>({f,score:f.terms.filter(t=>q.includes(t)).length})).filter(x=>x.score>0).sort((a,b)=>b.score-a.score);
  if(matches.length)return {answer:matches[0].f.answer,sources:matches[0].f.sources,status:'recorded'};
  return unknown;
 }
 if(typeof module!=='undefined'&&module.exports)module.exports={answer};else root.defenderAnswers=Object.freeze({answer});
})(typeof window==='undefined'?{}:window);
