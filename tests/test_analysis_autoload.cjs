const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
// Minimal DOM contract: exercise the actual route/request handlers without a browser dependency.
class Element {
 constructor(id='') { this.id=id; this.children=[]; this.dataset={}; this.style={}; this.value=''; this.classList={toggle(){}}; }
 append(...nodes){this.children.push(...nodes);}
 replaceChildren(...nodes){this.children=nodes;}
 setAttribute(key,value){this[key]=value;}
 removeAttribute(key){delete this[key];}
 addEventListener(){}
 querySelectorAll(){return [];}
 click(){}
 scrollIntoView(){}
 focus(){}
}
const tick=()=>new Promise(resolve=>setImmediate(resolve));
async function scenario(hash, query='', fail=false, viewBridge=undefined){
 const ids=new Map(),get=id=>{if(!ids.has(id))ids.set(id,new Element(id));return ids.get(id);};
 get('analysis-study').value='all';get('representation').value='quantile';get('gene-count').value='500';get('heat-scale').value='zscore';get('heat-cluster').value='both';
 const tabs=['pca','heatmap','correlation','qc','clustering'].map(view=>{const e=new Element();e.dataset.view=view;return e;});
 const downloads=[],requests=[],events={},warnings=[],location={hash,search:query};
 let release;
 const pending=new Promise(resolve=>release=resolve);
 const catalog={profiles:0,studies:[],groups:0,probes:0,annotation_genes:0,samples:[]};
 // Empty plotted vectors are sufficient here; numerical/rendering coverage lives in the heatmap and real-data checks.
 const result={heatmap:{gene_ids:['1','2'],zscore:[[-1,1],[-1,null]],log2:[[1,3],[2,null]]},samples:[{sample_id:'sample-a',study_id:'GSE102971',dose_gy:0},{sample_id:'sample-b',study_id:'GSE102971',dose_gy:2}],genes_used:500,options:{study:'all',representation:'quantile',genes:500},pca:{scores:[[0,0],[1,1]],explained_variance_ratio:[0,0]},provenance:{}};
 const context=vm.createContext({console:{...console,warn:(...args)=>warnings.push(args)},URLSearchParams,location,matchMedia:()=>({matches:false,addEventListener(){}}),setTimeout:()=>{},Blob,URL:{createObjectURL:blob=>{downloads.push(blob);return 'blob:test';},revokeObjectURL(){}},
  document:{getElementById:get,createElement:()=>new Element(),createElementNS:()=>new Element(),querySelector:()=>null,querySelectorAll:selector=>selector==='[data-view]'?tabs:[]},
  window:{defenderAnalysisView:viewBridge,defenderHeatmap:require('../app/static/heatmap-clustering.js'),addEventListener:(key,fn)=>events[key]=fn,scrollTo(){},defenderDevelopment:{ensureClusters(){}},defenderStatistics:{ensure(){}},defenderPublic:{read:async(path)=>{
   requests.push(path);if(path==='/api/catalog')return catalog;
   await pending;if(fail){fail=false;throw Error('Temporary failure');}return result;
  }}}
 });
 // Catalog study cards are unrelated; give the default study enough metadata for selection.
 catalog.studies=[{study_id:'GSE102971',profiles:0,groups:0,heading:'Study',subtitle:'Study'}];
 vm.runInContext(fs.readFileSync('app/static/explorer.js','utf8'),context);await tick();
 return {requests,events,location,get,tabs,release,downloads,result,warnings};
}
(async()=>{
 const a=await scenario('#analysis');
 assert.deepEqual(a.requests,['/api/catalog','/api/analysis'],'direct Analysis link starts one default request');
 a.events.hashchange();a.tabs[0].onclick();a.get('analysis-form').onsubmit({preventDefault(){}});
 assert.equal(a.requests.length,2,'re-entry and clicks cannot duplicate in-flight work');
 assert.equal(a.get('analysis-panel')['aria-busy'],'true');
 a.release();await tick();
 assert.equal(a.get('analysis-content').hidden,false);
 assert.equal(a.get('analysis-panel')['aria-busy'],'false');
 a.location.hash='#overview';a.events.hashchange();a.location.hash='#analysis';a.events.hashchange();
 a.get('analysis-form').onsubmit({preventDefault(){}});await tick();
 assert.equal(a.requests.length,2,'returning to Analysis and rerunning identical options reuse cache');
 const original=JSON.stringify(a.result);a.tabs[1].onclick();
 const layout=JSON.parse(a.get('analysis-provenance').textContent).heatmap_display;
 assert.equal(layout.distance_only_median_imputations,1);
 a.get('heat-scale').value='log2';a.get('heat-scale').onchange();
 assert.deepEqual(JSON.parse(a.get('analysis-provenance').textContent).heatmap_display.profiles,layout.profiles,'color scale does not change clustering');
 a.get('download-analysis').onclick();const exported=JSON.parse(await a.downloads[0].text());
 assert.deepEqual(exported.heatmap,a.result.heatmap);assert.equal(exported.heatmap_display.scale,'log2');
 assert.equal(JSON.stringify(a.result),original,'display and export never mutate source data');
 assert.equal(a.requests.length,2,'heatmap controls need no extra requests');
 const b=await scenario('#overview');assert.deepEqual(b.requests,['/api/catalog'],'other pages do not compute an analysis');
 b.location.hash='#analysis';b.events.hashchange();assert.equal(b.requests.length,2);b.release();await tick();
 const c=await scenario('#analysis','?analysis=clustering');assert.equal(c.requests.length,1,'clustering deep link does not spend core-analysis request');
 c.tabs[0].onclick();assert.equal(c.requests.length,2);c.release();await tick();
 const retired=await scenario('#analysis','?analysis=statistics');assert.deepEqual(retired.requests,['/api/catalog','/api/analysis'],'an old dose-response link safely opens the retained PCA view');retired.release();await tick();
 const d=await scenario('#analysis','',true);d.release();await tick();
 assert(d.get('analysis-status').textContent.includes('retry'));d.events.hashchange();assert.equal(d.requests.length,2,'failure does not loop automatically');
 d.get('analysis-form').onsubmit({preventDefault(){}});await tick();assert.equal(d.requests.length,3);assert.equal(d.get('analysis-content').hidden,false,'explicit retry recovers');
 const updates=[];const e=await scenario('#analysis','',false,{update:state=>updates.push(state),selectProfile:()=>{e.get('point-detail').textContent='Profile inspector updated';}});e.release();await tick();
 assert(updates.some(state=>state.result===e.result&&state.view==='pca'),'presentation receives the exact existing result, without another request');
 assert.equal(e.requests.length,2);
 const saved=JSON.stringify(e.result);e.tabs[1].onclick();assert.equal(JSON.stringify(e.result),saved,'presentation notifications preserve all measured values');
 const heatCell=e.get('analysis-chart').children.find(node=>node['data-gene']);heatCell.onclick();
 assert.equal(e.get('point-detail').textContent,heatCell['aria-label'],'the inspector cannot overwrite the selected gene expression value');
 const f=await scenario('#analysis','',false,{update(){throw Error('Optional renderer unavailable');}});f.release();await tick();
 assert.equal(f.get('analysis-content').hidden,false,'the existing 2D chart survives presentation failure');
 assert.equal(f.requests.length,2,'presentation failure cannot create an API retry loop');
 assert(f.warnings.length>=1,'optional presentation failures are surfaced rather than silently swallowed');
 f.get('download-analysis').onclick();assert.deepEqual(JSON.parse(await f.downloads[0].text()).pca,JSON.parse(JSON.stringify(f.result.pca)),'original snapshot export survives presentation failure');
 console.log('Analysis autoload passed: direct link, deferred entry, no background computation, one in-flight request, cache, clustering entry, error and explicit retry.');
})().catch(error=>{console.error(error);process.exitCode=1;});
