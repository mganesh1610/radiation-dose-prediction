'use strict';
// Public requests only resolve to the static, versioned export. No API fallback.
(()=>{
 const requests=new Map();
 async function json(path){
  if(!/^\/data\/(?:publication\.json|[a-f0-9]{20}\/[A-Za-z0-9_-]+\.json)$/.test(path))throw Error('Unknown published resource.');
  if(!requests.has(path))requests.set(path,(async()=>{
   const response=await fetch(path,{credentials:'omit',redirect:'error'});
   if(!response.ok)throw Error('Published data could not be loaded. Please retry.');
   return response.json();
  })().catch(error=>{requests.delete(path);throw error;}));
  return requests.get(path);
 }
 let publication;
 async function snapshot(){
  publication=await json('/data/publication.json');
  if(publication.schema_version!==1||!/^\/[a-z]+\/[a-f0-9]{20}$/.test(publication.data_path))throw Error('Unsupported published snapshot.');
  document.getElementById('snapshot-note').textContent=`Public research explorer · Snapshot ${publication.published_date} · ${publication.profiles} core profiles`;
  return publication;
 }
 async function read(path,options){
  let filename;
  if(path==='/api/catalog'&&options===undefined)filename='catalog.json';
  else if(path==='/published/statistical-results'&&options===undefined)filename='statistical-results.json';
  else if(path==='/published/assistant-records'&&options===undefined)filename='assistant-records.json';
  else if(path==='/published/development-results'&&options===undefined)filename='development-results.json';
  else if(path==='/published/final-results'&&options===undefined)filename='research-results.json';
  else if(path==='/published/experiments'&&options===undefined)filename='experiments.json';
  else if(path==='/api/analysis'&&options&&Object.keys(options).sort().join(',')==='genes,representation,study'
   &&['all','GSE102971','GSE8917','GSE65292'].includes(options.study)
   &&['log2','quantile'].includes(options.representation)&&[100,500,2000].includes(options.genes)){
    filename=`analysis-${options.study}-${options.representation}-${options.genes}.json`;
  }else throw Error('This action is unavailable in the public explorer.');
  const info=await snapshot();
  return json(info.data_path+'/'+filename);
 }
 window.defenderPublic=Object.freeze({read});
})();
