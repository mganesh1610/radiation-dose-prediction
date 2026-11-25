/** Coordinate preparation for display only. Never fit PCA or modify analysis records. */
export const STUDY_STYLES = Object.freeze({
  GSE102971: Object.freeze({color:'#77d8a9',shape:'circle'}),
  GSE8917: Object.freeze({color:'#edb45e',shape:'diamond'}),
  GSE65292: Object.freeze({color:'#9cacf8',shape:'square'})
});

/** All three axes use the full reference panel and one common display scale. */
export function clusterSpatialFrame(reference, points) {
  const finitePoint = p => p && [p.x, p.y, p.z].every(Number.isFinite);
  if (!reference?.length || !reference.every(finitePoint) || !points.every(finitePoint)) {
    throw new Error('Three saved principal components are required for the 3D clustering view.');
  }
  function extent(key) {
    const values = reference.map(p => p[key]);
    let low = Math.min(...values), high = Math.max(...values);
    if (low === high) { low -= 1; high += 1; }
    return {low, high, center: (low + high) / 2};
  }
  const pc1 = extent('x'), pc2 = extent('y'), pc3 = extent('z');
  const scale = 10 / Math.max(...[pc1, pc2, pc3].map(a => a.high - a.low));
  const position = (axis, value) => (value - axis.center) * scale;
  function ticks(axis) {
    return Array.from({length: 5}, (_, i) => {
      const value = axis.low + (axis.high - axis.low) * i / 4;
      return {value: Number(value.toFixed(1)), position: position(axis, value)};
    });
  }
  return {
    bounds: {left: position(pc1, pc1.low), right: position(pc1, pc1.high),
      back: position(pc2, pc2.low), front: position(pc2, pc2.high),
      floor: position(pc3, pc3.low), ceiling: position(pc3, pc3.high)},
    ticks: {pc1: ticks(pc1), pc2: ticks(pc2), pc3: ticks(pc3)},
    positions: points.map((p, index) => ({
      index, x: position(pc1, p.x), y: position(pc3, p.z), z: position(pc2, p.y)
    }))
  };
}

export function spatialFrame(analysis, outlines = new Map()) {
  const samples=analysis?.samples, scores=analysis?.pca?.scores;
  if(!Array.isArray(samples)||!samples.length||!Array.isArray(scores)||scores.length!==samples.length)
    throw new Error('A complete measured PCA is needed for the spatial view.');
  for(let i=0;i<samples.length;i++) {
    if(!Array.isArray(scores[i])||scores[i].length<2||!scores[i].slice(0,2).every(Number.isFinite))
      throw new Error('The spatial view requires two finite PCA coordinates per profile.');
    if(!Number.isFinite(samples[i].dose_gy)||samples[i].dose_gy<0)
      throw new Error('A recorded numerical dose is required for every spatial height. Use the 2D view.');
  }
  const limits = scores.concat(...outlines.values());
  const extent=axis=>{
    const values=limits.map(row=>row[axis]);
    let low=Math.floor(Math.min(...values)/10)*10, high=Math.ceil(Math.max(...values)/10)*10;
    if(low===high){low-=10;high+=10;}
    return {low,high,center:(low+high)/2};
  };
  const pc1=extent(0),pc2=extent(1),units=11/Math.max(pc1.high-pc1.low,pc2.high-pc2.low);
  const doseMaximum=Math.max(2,Math.ceil(Math.max(...samples.map(s=>s.dose_gy))/2)*2);
  const x=value=>(value-pc1.center)*units,z=value=>(value-pc2.center)*units,y=value=>-2.5+value/doseMaximum*5;
  return {
    axes:{pc1,pc2,dose:{low:0,high:doseMaximum},height:'recorded dose (Gy), not PC3'},
    bounds:{left:x(pc1.low),right:x(pc1.high),back:z(pc2.low),front:z(pc2.high),floor:-2.5,ceiling:2.5},
    floorPosition: ([pc1,pc2]) => [x(pc1), -2.5, z(pc2)],
    positions:samples.map((sample,index)=>({index,x:x(scores[index][0]),y:y(sample.dose_gy),z:z(scores[index][1])})),
    ticks:{pc1:Array.from({length:(pc1.high-pc1.low)/10+1},(_,i)=>({value:pc1.low+i*10,position:x(pc1.low+i*10)})),
      pc2:Array.from({length:(pc2.high-pc2.low)/10+1},(_,i)=>({value:pc2.low+i*10,position:z(pc2.low+i*10)})),
      dose:Array.from({length:5},(_,i)=>({value:doseMaximum*i/4,position:y(doseMaximum*i/4)}))}
  };
}
