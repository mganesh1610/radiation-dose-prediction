import * as THREE from './vendor/three/three.module.js';
import {OrbitControls} from './vendor/three/OrbitControls.js';
import {spatialFrame,STUDY_STYLES} from './spatial-model.mjs';

/** Optional, demand-rendered visualization of the analysis service's existing result. */
export function createAnalysisScene(container, callbacks) {
  const renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true,powerPreference:'low-power'});
  renderer.setPixelRatio(Math.min(devicePixelRatio||1,2));
  renderer.outputColorSpace=THREE.SRGBColorSpace;
  const canvas=renderer.domElement;
  canvas.tabIndex=0;
  canvas.setAttribute('role','img');
  canvas.setAttribute('aria-label','Spatial PCA: PC1 and PC2 on the floor, recorded dose in Gy vertically. Drag to rotate, scroll to zoom. Arrow keys select profiles; Home resets the camera.');
  container.replaceChildren(canvas);
  const world=new THREE.Scene();world.background=new THREE.Color('#10261f');
  const camera=new THREE.OrthographicCamera(-8,8,6,-6,.1,120);
  const controls=new OrbitControls(camera,canvas);
  controls.enableDamping=false;controls.rotateSpeed=.7;controls.zoomSpeed=.8;
  controls.minZoom=.4;controls.maxZoom=4;controls.minPolarAngle=.12;controls.maxPolarAngle=Math.PI*.49;
  world.add(new THREE.AmbientLight('#e3f4df',2.1));
  const key=new THREE.DirectionalLight('#fff',2.4);key.position.set(-6,13,10);world.add(key);
  const fill=new THREE.DirectionalLight('#9ed7bc',1.1);fill.position.set(8,4,-9);world.add(fill);
  let group=null,points=[],ring=null,current=null,active=false,chosen=-1,highlight=null,rotating=false,rotationHandle=0;
  const draw=()=>{if(active&&!document.hidden)renderer.render(world,camera);};
  controls.addEventListener('change',draw);
  const motion=matchMedia('(prefers-reduced-motion:reduce)');
  function spin(enabled){
    rotating=Boolean(enabled)&&!motion.matches;controls.autoRotate=rotating;cancelAnimationFrame(rotationHandle);
    callbacks.onRotation?.(rotating);
    if(rotating){let last=performance.now();const frame=now=>{if(!rotating)return;const delta=Math.min((now-last)/1000,.1);last=now;if(active&&!document.hidden)controls.update(delta);rotationHandle=requestAnimationFrame(frame);};controls.autoRotateSpeed=.7;rotationHandle=requestAnimationFrame(frame);}
    return rotating;
  }
  motion.addEventListener('change',()=>{if(motion.matches)spin(false);});
  document.addEventListener('visibilitychange',()=>{if(document.hidden)spin(false);else draw();});
  function reset(){camera.position.set(11,9.5,16);camera.zoom=1;controls.target.set(0,-.1,0);camera.updateProjectionMatrix();controls.update();draw();}
  function resize(){
    const r=container.getBoundingClientRect();if(!r.width||!r.height)return;
    const aspect=r.width/r.height,height=aspect<1?17/Math.sqrt(aspect):12.8;
    camera.left=-height*aspect/2;camera.right=height*aspect/2;camera.top=height/2;camera.bottom=-height/2;
    camera.updateProjectionMatrix();renderer.setSize(r.width,r.height,false);
    container.classList.toggle('compact',r.width<450);
    group?.traverse(object=>{if(object.isSprite&&object.userData.axisTitle)object.visible=r.width>=450;});draw();
  }
  new ResizeObserver(resize).observe(container);
  function stroke(a,b,opacity=.23,color='#5a8163'){
    const geometry=new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...a),new THREE.Vector3(...b)]);
    group.add(new THREE.Line(geometry,new THREE.LineBasicMaterial({color,transparent:true,opacity})));
  }
  function text(value,position,heading=false){
    const bitmap=document.createElement('canvas');bitmap.width=512;bitmap.height=64;
    const ctx=bitmap.getContext('2d');ctx.font='400 28px Geist, sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillStyle=heading?'#c6dccb':'#a3bfac';ctx.fillText(String(value),256,32);
    const texture=new THREE.CanvasTexture(bitmap);texture.colorSpace=THREE.SRGBColorSpace;
    const label=new THREE.Sprite(new THREE.SpriteMaterial({map:texture,transparent:true,depthTest:false,depthWrite:false}));
    label.position.set(...position);const height=heading?.68:.52;label.scale.set(height*8,height,1);label.userData.axisTitle=heading;group.add(label);
  }
  function release(){
    if(!group)return;world.remove(group);const geometries=new Set(),materials=new Set();
    group.traverse(object=>{if(object.geometry)geometries.add(object.geometry);if(object.material)materials.add(object.material);});
    geometries.forEach(g=>g.dispose());materials.forEach(m=>{m.map?.dispose();m.dispose();});
  }
  function setResult(result){
    if(result===current)return;
    const frame=spatialFrame(result);release();current=result;points=[];chosen=-1;highlight=null;
    group=new THREE.Group();world.add(group);
    const b=frame.bounds;
    for(const tick of frame.ticks.pc1){stroke([tick.position,b.floor,b.back],[tick.position,b.floor,b.front]);stroke([tick.position,b.floor,b.back],[tick.position,b.ceiling,b.back],.13);text(tick.value,[tick.position,b.floor-.3,b.front+.23]);}
    for(const tick of frame.ticks.pc2){stroke([b.left,b.floor,tick.position],[b.right,b.floor,tick.position]);stroke([b.left,b.floor,tick.position],[b.left,b.ceiling,tick.position],.13);text(tick.value,[b.left-.67,b.floor-.24,tick.position]);}
    for(const tick of frame.ticks.dose){stroke([b.left,tick.position,b.back],[b.right,tick.position,b.back],.15);stroke([b.left,tick.position,b.back],[b.left,tick.position,b.front],.15);text(Number(tick.value.toFixed(2)),[b.left-.35,tick.position,b.back]);}
    stroke([b.left,b.floor,b.front],[b.right,b.floor,b.front],.5,'#b7d4bf');stroke([b.left,b.floor,b.back],[b.left,b.floor,b.front],.5,'#b7d4bf');stroke([b.left,b.floor,b.back],[b.left,b.ceiling,b.back],.5,'#b7d4bf');
    const explained=result.pca.explained_variance_ratio;
    text(`PC1 · ${(explained[0]*100).toFixed(1)}%`,[0,b.floor-.8,b.front+.6],true);
    text(`PC2 · ${(explained[1]*100).toFixed(1)}%`,[b.left-3.6,b.floor-.8,(b.back+b.front)/2],true);
    text('Recorded dose · Gy',[b.left-.1,b.ceiling+.55,b.back],true);
    const shapes={circle:new THREE.SphereGeometry(.105,18,12),diamond:new THREE.OctahedronGeometry(.135),square:new THREE.BoxGeometry(.17,.17,.17)};
    for(const p of frame.positions){
      const sample=result.samples[p.index],style=STUDY_STYLES[sample.study_id]||{color:'#becfba',shape:'circle'};
      const material=new THREE.MeshStandardMaterial({color:style.color,emissive:style.color,emissiveIntensity:.08,metalness:.08,roughness:.35,transparent:true});
      const dot=new THREE.Mesh(shapes[style.shape],material);dot.position.set(p.x,p.y,p.z);dot.userData={index:p.index,study:sample.study_id};group.add(dot);points.push(dot);
    }
    ring=new THREE.Mesh(new THREE.SphereGeometry(.16,20,12),new THREE.MeshBasicMaterial({color:'#edf1dc',wireframe:true,transparent:true,opacity:.6}));ring.visible=false;group.add(ring);
    container.dataset.renderer='threejs';container.dataset.profiles=String(points.length);container.dataset.analysisId=result.analysis_id;container.dataset.heightAxis='recorded-dose-gy';
    reset();resize();
  }
  function choose(index){chosen=index;if(!ring)return;ring.visible=Boolean(points[index]);if(points[index])ring.position.copy(points[index].position);draw();}
  function emphasize(study){highlight=study;for(const p of points){p.material.opacity=!study||p.userData.study===study?1:.14;p.material.depthWrite=p.material.opacity===1;}draw();}
  const picker=new THREE.Raycaster(),pointer=new THREE.Vector2();
  function under(event){const r=canvas.getBoundingClientRect();pointer.set((event.clientX-r.left)/r.width*2-1,1-(event.clientY-r.top)/r.height*2);picker.setFromCamera(pointer,camera);return picker.intersectObjects(points).find(hit=>hit.object.material.opacity>.5)?.object;}
  let down;
  canvas.addEventListener('pointerdown',event=>{down=[event.clientX,event.clientY];});
  canvas.addEventListener('pointerup',event=>{if(!down||Math.hypot(event.clientX-down[0],event.clientY-down[1])>6)return;const dot=under(event);if(dot)callbacks.onSelect(dot.userData.index);});
  canvas.addEventListener('pointermove',event=>{if(event.buttons){callbacks.onHover(null);return;}const dot=under(event);canvas.style.cursor=dot?'pointer':'grab';callbacks.onHover(dot?{index:dot.userData.index,x:event.clientX,y:event.clientY}:null);});
  canvas.addEventListener('pointerleave',()=>callbacks.onHover(null));
  canvas.addEventListener('keydown',event=>{
    const direction=['ArrowRight','ArrowDown'].includes(event.key)?1:['ArrowLeft','ArrowUp'].includes(event.key)?-1:0;
    if(direction&&points.length){event.preventDefault();callbacks.onSelect(chosen<0?(direction<0?points.length-1:0):(chosen+direction+points.length)%points.length);}
    else if(event.key==='Home'){event.preventDefault();reset();}
    else if(['+','=','-'].includes(event.key)){event.preventDefault();camera.zoom=THREE.MathUtils.clamp(camera.zoom*(event.key==='-'?1/1.15:1.15),.4,4);camera.updateProjectionMatrix();draw();}
  });
  canvas.addEventListener('webglcontextlost',event=>{event.preventDefault();spin(false);active=false;callbacks.onUnavailable();});
  reset();resize();
  return Object.freeze({setResult,select:choose,highlight:emphasize,reset,resize,rotate:spin,
    setActive(value){active=Boolean(value);if(!active)spin(false);else resize();},
    image(){draw();return new Promise((resolve,reject)=>canvas.toBlob(blob=>blob?resolve(blob):reject(new Error('Image export unavailable.')),'image/png'));}
  });
}
