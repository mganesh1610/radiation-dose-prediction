const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.join(__dirname, '..', 'data', 'public_web');
if (!fs.existsSync(path.join(root, 'data/publication.json'))) {
  if (process.env.DEFENDER_REQUIRE_DATA_TESTS === '1') throw new Error('Reviewed public export is required');
  console.log('SKIP public adapter: separately provisioned reviewed public export is absent.');
  process.exit(0);
}

const requests = [], note = {};
let failNext = false;
const context = vm.createContext({window:{}, document:{getElementById:()=>note}, fetch:async(url,options)=>{
  requests.push({url,options});
  if(failNext){failNext=false;return {ok:false};}
  return {ok:true,json:async()=>JSON.parse(fs.readFileSync(path.join(root,url),'utf8'))};
}});
vm.runInContext(fs.readFileSync(path.join(root,'static/public-data.js'),'utf8'),context);
const read=context.window.defenderPublic.read;
(async()=>{
 await Promise.all([read('/api/catalog'),read('/api/catalog')]);
 assert.equal(requests.length,2,'concurrent readers share publication/catalog fetches');
 await read('/api/analysis',{study:'all',representation:'quantile',genes:500});
 await read('/api/analysis',{study:'all',representation:'quantile',genes:500});
 assert.equal(requests.length,3,'repeat chart settings reuse memory');
 for(const route of ['/api/jobs','/api/predict','/api/jobs/example/cancel','https://other.test/data'])await assert.rejects(read(route));
 await assert.rejects(read('/api/analysis',{study:'GSE90909',representation:'log2',genes:500}));
 assert.equal(requests.length,3,'unapproved actions never reach network');
 failNext=true;
 await assert.rejects(read('/api/analysis',{study:'GSE8917',representation:'log2',genes:100}));
 await read('/api/analysis',{study:'GSE8917',representation:'log2',genes:100});
 assert.equal(requests.length,5,'failed static reads can be retried');
 const final=await read('/published/final-results');
 assert.equal(final.model_cards.length,8);
 assert.equal(final.status,'completed_consumed');
 assert.equal(final.live_prediction_enabled,false);
 assert.equal(requests.length,6);
 const development=await read('/published/development-results');
 assert.equal(development.regression.length,20);
 assert.equal(development.final_test_included,false);
 await read('/published/development-results');
 assert.equal(requests.length,7,'development views share a cached static snapshot');
 assert(requests.every(r=>r.url.startsWith('/data/')&&r.options.credentials==='omit'&&r.options.redirect==='error'&&!r.options.method&&!r.options.body));
 assert(note.textContent.includes('145 core profiles'));
 console.log('Public adapter passed: cache, concurrent reads, retry, action rejection, same-origin static GET-only requests.');
})().catch(error=>{console.error(error);process.exitCode=1;});
