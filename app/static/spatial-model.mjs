/** Coordinate preparation for display only. Never fit PCA or modify analysis records. */
export const STUDY_STYLES = Object.freeze({
  GSE102971: Object.freeze({color:'#77d8a9',shape:'circle'}),
  GSE8917: Object.freeze({color:'#edb45e',shape:'diamond'}),
  GSE65292: Object.freeze({color:'#9cacf8',shape:'square'})
});

export function spatialFrame(analysis) {
  const samples=analysis?.samples, scores=analysis?.pca?.scores;
  if(!Array.isArray(samples)||!samples.length||!Array.isArray(scores)||scores.length!==samples.length)
    throw new Error('A complete measured PCA is needed for the spatial view.');
  for(let i=0;i<samples.length;i++) {
    if(!Array.isArray(scores[i])||scores[i].length<2||!scores[i].slice(0,2).every(Number.isFinite))
      throw new Error('The spatial view requires two finite PCA coordinates per profile.');
    if(!Number.isFinite(samples[i].dose_gy)||samples[i].dose_gy<0)
      throw new Error('A recorded numerical dose is required for every spatial height. Use the 2D view.');
  }
  const extent=axis=>{
    const values=scores.map(row=>row[axis]);
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
    positions:samples.map((sample,index)=>({index,x:x(scores[index][0]),y:y(sample.dose_gy),z:z(scores[index][1])})),
    ticks:{pc1:Array.from({length:(pc1.high-pc1.low)/10+1},(_,i)=>({value:pc1.low+i*10,position:x(pc1.low+i*10)})),
      pc2:Array.from({length:(pc2.high-pc2.low)/10+1},(_,i)=>({value:pc2.low+i*10,position:z(pc2.low+i*10)})),
      dose:Array.from({length:5},(_,i)=>({value:doseMaximum*i/4,position:y(doseMaximum*i/4)}))}
  };
}
