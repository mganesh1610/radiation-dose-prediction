import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {spatialFrame} from '../app/static/spatial-model.mjs';

const measured=()=>({samples:[{sample_id:'control',dose_gy:0},{sample_id:'exposed',dose_gy:8}],pca:{scores:[[-10,5],[10,-5]],explained_variance_ratio:[.4,.1]}});
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
