import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {spatialFrame,clusterSpatialFrame} from '../app/static/spatial-model.mjs';
import outliers from '../app/static/pca-outliers.js';

const measured=()=>({samples:[{sample_id:'control',dose_gy:0},{sample_id:'exposed',dose_gy:8}],pca:{scores:[[-10,5],[10,-5]],explained_variance_ratio:[.4,.1]}});
test('the PCA ellipse uses the same study covariance for its boundary and review flags',()=>{
 const scores=Array.from({length:24},(_,i)=>[Math.cos(i*Math.PI/12),Math.sin(i*Math.PI/12)]);
 scores.push([6,4]);
 const other=scores.slice(0,-1).map(([x,y])=>[50+x,x+2*y]);
 const data={samples:[...scores,...other].map((_,i)=>({study_id:i<scores.length?'A':'B',dose_gy:i===6?99:0})),
   pca:{scores:scores.concat(other)}};
 const original=JSON.stringify(data);
 const guide=outliers.forResult(data);
 assert.deepEqual(guide.flagged,[24]);
 assert.equal(guide.outlines.size,2);
 assert.equal(outliers.status(guide.distances[6]),'Within guide');
 assert.equal(outliers.status(guide.distances[24]),'Flagged');
 // The second study is a known affine transform of a unit circle with sample variance 12/23.
 for(const [x,y] of guide.outlines.get('B')){
  const u=x-50,v=(y-u)/2;
  assert(Math.abs((u*u+v*v)*23/12-outliers.cutoff)<1e-9);
 }
 const frame=spatialFrame(data,guide.outlines);
 for(const outline of guide.outlines.values())for(const pair of outline){
  const [x,y,z]=frame.floorPosition(pair);
  assert(x>=frame.bounds.left&&x<=frame.bounds.right&&y===frame.bounds.floor&&z>=frame.bounds.back&&z<=frame.bounds.front);
 }
 assert.deepEqual(outliers.forResult({...data,samples:data.samples.map(s=>({...s,dose_gy:1000}))}).distances,guide.distances);
 assert.equal(JSON.stringify(data),original);
});
test('small or singular studies stay unscored and the guide can genuinely flag zero profiles',()=>{
 const scores=[[0,0],[1,0],[1,1],[0,1],[0,0],[1,1],[2,2],[0,0],[1,2]];
 const guide=outliers.forResult({samples:['A','A','A','A','B','B','B','C','C'].map(study_id=>({study_id})),pca:{scores}});
 assert.deepEqual([...guide.outlines.keys()],['A']);
 assert.deepEqual(guide.distances.slice(4),[null,null,null,null,null]);
 assert.deepEqual(guide.flagged,[]);
 assert.equal(outliers.status(null),'Unscored');
});
test('clustering uses signed PC3 and fixed axes when the displayed study changes',()=>{
 const reference=[{x:-8,y:0,z:-4,dose_gy:2},{x:0,y:4,z:0,dose_gy:2},{x:8,y:8,z:4,dose_gy:2}];
 const all=clusterSpatialFrame(reference,reference);
 const subset=clusterSpatialFrame(reference,[reference[1],reference[2]]);
 assert.deepEqual(subset.bounds,all.bounds);
 assert.deepEqual(subset.ticks,all.ticks);
 assert.deepEqual(subset.positions,all.positions.slice(1).map((p,index)=>({...p,index})));
 const [low,mid,high]=all.positions;
 assert(low.y<0&&high.y>0);
 assert.equal(high.x-mid.x,2*(high.y-mid.y));
 assert.equal(high.z-mid.z,high.y-mid.y);
 assert.equal(clusterSpatialFrame(reference,[]).positions.length,0);
 assert.throws(()=>clusterSpatialFrame([{x:1,y:2,dose_gy:0}],[]),/Three saved principal/);
});
test('spatial height is the recorded dose; a third PCA component is neither required nor invented',()=>{
 const data=measured(),before=JSON.stringify(data),frame=spatialFrame(data);
 assert.equal(frame.positions[0].y,frame.bounds.floor);
 assert.equal(frame.positions[1].y,frame.bounds.ceiling);
 assert.equal(frame.axes.height,'recorded dose (Gy), not PC3');
 assert.equal(JSON.stringify(data),before);
});
test('PC1 and PC2 retain the same units and every profile retains its source order',()=>{
 const frame=spatialFrame(measured());
 assert.deepEqual(frame.positions.map(p=>p.index),[0,1]);
 const pc1Change=frame.positions[1].x-frame.positions[0].x;
 const pc2Change=frame.positions[0].z-frame.positions[1].z;
 assert.equal(pc1Change/pc2Change,2);
});
test('a zero-dose or constant-coordinate cohort still has finite display geometry',()=>{
 const data=measured();data.samples.forEach(s=>s.dose_gy=0);data.pca.scores=[[0,0],[0,0]];
 const frame=spatialFrame(data);
 assert(frame.positions.every(p=>[p.x,p.y,p.z].every(Number.isFinite)));
 assert(frame.positions.every(p=>p.y===frame.bounds.floor));
});
test('missing or invalid measured values fail closed to the existing 2D view',()=>{
 for(const dose of [null,undefined,NaN,Infinity,-1]){const data=measured();data.samples[0].dose_gy=dose;assert.throws(()=>spatialFrame(data),/recorded numerical dose/);}
 const data=measured();data.pca.scores[0][0]=NaN;assert.throws(()=>spatialFrame(data),/finite PCA/);
 data.pca.scores=[];assert.throws(()=>spatialFrame(data),/complete measured PCA/);
});
test('both HTML entry points preserve the original controls and load enhancement before the data controller',()=>{
 const required=['analysis-study','representation','gene-count','run-analysis','analysis-chart','analysis-table','analysis-provenance','heat-scale','heat-cluster','tab-clustering','clustering-explorer','workspace-methods-dialog','workspace-methods-close','workspace-methods-title','workspace-methods-content'];
 for(const name of ['app/static/index.html','app/public/index.html']){
  const html=fs.readFileSync(new URL('../'+name,import.meta.url),'utf8');
  for(const id of required)assert.equal((html.match(new RegExp(`id="${id}"`,'g'))||[]).length,1,`${name}: ${id}`);
  assert(html.indexOf('/static/analysis-workspace.js')<html.indexOf('/static/explorer.js'));
  assert(html.includes('id="workspace-outlier-toggle"')&&html.includes('id="workspace-review-outlier"'));
  assert(html.indexOf('/static/pca-outliers.js')>=0&&html.indexOf('/static/pca-outliers.js')<html.indexOf('/static/analysis-workspace.js'));
  assert(html.includes('value="100"')&&html.includes('value="2000"')&&html.includes('value="log2"'));
  assert(!html.includes('tab-statistics')&&!html.includes('/static/statistical-explorer.js'),'retired dose-response UI is not loaded');
  assert(!html.includes('Expression in perspective.'),'the removed hero heading cannot return');
  assert(html.includes('class="analysis-toolbar"'),'actions and remaining tabs share a toolbar');
  assert(html.includes('<dialog id="workspace-methods-dialog" aria-labelledby="workspace-methods-title">'),'the presentation adapter requires its shared Methods dialog');
 }
});
test('renderer modules resolve locally without a CDN or inline import-map exception',()=>{
 const scene=fs.readFileSync(new URL('../app/static/analysis-scene.mjs',import.meta.url),'utf8');
 const orbit=fs.readFileSync(new URL('../app/static/vendor/three/OrbitControls.js',import.meta.url),'utf8');
 assert(scene.includes("from './vendor/three/three.module.js'"));
 assert(orbit.includes("from './three.module.js'"));
 assert(!scene.includes('https://'));
});
test('presentation state cannot register the page as an original analysis tab',()=>{
 const adapter=fs.readFileSync(new URL('../app/static/analysis-workspace.js',import.meta.url),'utf8');
 // The existing explorer binds click/keyboard handlers to every [data-view] node.
 assert(!/\.dataset\.view\s*=/.test(adapter),'data-view is reserved for original tab controls');
 for(const name of ['app/static/index.html','app/public/index.html']){
  const html=fs.readFileSync(new URL('../'+name,import.meta.url),'utf8');
  assert.equal((html.match(/data-view=/g)||[]).length,5);
 }
});
