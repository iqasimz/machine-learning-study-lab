'use strict';
const percent = n => (100*n).toFixed(1) + '%';
const sigmoid = z => 1/(1+Math.exp(-z));
const labDefinitions = {
  'line-fit': {
    title:'Fit the original line by hand',
    inputs:[['Weight',-.5,2.5,.05,1],['Bias',-1,4,.05,0]],
    run:([w,b])=>{
      const xs=[1,2,3,4], ys=[2,3,5,4], fit=x=>w*x+b;
      const mse=xs.reduce((s,x,i)=>s+(fit(x)-ys[i])**2,0)/4;
      return `ŷ = ${w.toFixed(2)}x ${b<0?'−':'+'} ${Math.abs(b).toFixed(2)} · MSE ${mse.toFixed(3)}`;
    }, reset:['Optimal fit',[.8,1.5]], plot:true
  },
  'sigmoid':{
    title:'Change the logit and decision threshold',
    inputs:[['Logit',-6,6,.1,1],['Threshold',.1,.9,.01,.5]],
    run:([z,t])=>`P(spam) ${percent(sigmoid(z))} · ${sigmoid(z)>=t?'Spam (1)':'Not spam (0)'}`
  },
  'softmax':{
    title:'Turn class scores into probabilities',
    inputs:[['Sports',-3,5,.1,2],['Technology',-3,5,.1,1],['Politics',-3,5,.1,0]],
    run:zs=>{const m=Math.max(...zs), e=zs.map(z=>Math.exp(z-m)), total=e.reduce((a,b)=>a+b);return e.map((v,i)=>['Sports','Technology','Politics'][i]+' '+percent(v/total)).join(' · ');}
  },
  'prior':{
    title:'Update the probability of spam after seeing “free”',
    inputs:[['Spam prior (%)',1,50,1,10]],
    run:([v])=>`P(spam | “free”) ${percent(.8*v/(.8*v+.1*(100-v)))}`
  },
  'leakage':{
    title:'Move the held-out observation',
    inputs:[['Validation spending',10,150,1,100]],
    run:([v])=>`Training-only mean 13.0 · Leaked mean ${((52+v)/5).toFixed(1)}`
  },
  'outreach':{
    title:'Choose the outreach threshold',
    inputs:[['Threshold (%)',20,80,5,50]],
    run:([v])=>{
      const [tp,fp,fn,tn]=[
        [8,22,0,10],[8,18,0,14],[7,16,1,16],[7,14,1,18],[7,12,1,20],[6,10,2,22],
        [6,8,2,24],[5,6,3,26],[4,5,4,27],[4,4,4,28],[3,3,5,29],[3,2,5,30],[2,1,6,31]
      ][(v-20)/5];
      return `TP ${tp} · FP ${fp} · FN ${fn} · TN ${tn} · Precision ${percent(tp/(tp+fp))} · Recall ${percent(tp/(tp+fn))} · Error cost €${15*fp+80*fn}`;
    }, reset:['Minimize toy cost',[40]]
  }
};
function drawFit(svg, w, b) {
  const tx=x=>50+82*x, ty=y=>194-24*y;
  const grid=[0,2,4,6].map(y=>`<line x1="45" x2="386" y1="${ty(y)}" y2="${ty(y)}" stroke="#d2dfdf"/><text x="39" y="${ty(y)+4}" text-anchor="end">${y}</text>`).join('');
  const dots=[2,3,5,4].map((y,i)=>{
    const x=i+1, yp=w*x+b;
    return `<line x1="${tx(x)}" x2="${tx(x)}" y1="${ty(y)}" y2="${ty(yp)}" stroke="#b97029" stroke-dasharray="4 3"/><circle cx="${tx(x)}" cy="${ty(y)}" r="5" fill="#286390"/>`;
  }).join('');
  svg.innerHTML=`<defs><clipPath id="fit-clip"><rect x="45" y="20" width="341" height="175"/></clipPath></defs>${grid}<path d="M45 20V194H386" stroke="#657789" fill="none"/><text x="47" y="15">Target y</text><text x="421" y="222" text-anchor="end">Feature x</text><g clip-path="url(#fit-clip)"><line x1="${tx(0)}" y1="${ty(b)}" x2="${tx(4.1)}" y2="${ty(w*4.1+b)}" stroke="#157c64" stroke-width="3"/>${dots}</g>${[1,2,3,4].map(x=>`<text x="${tx(x)}" y="211" text-anchor="middle">${x}</text>`).join('')}`;
}
for (const lab of document.querySelectorAll('[data-lab]')) {
  const definition=labDefinitions[lab.dataset.lab];
  const title=document.createElement('h4');
  title.textContent=definition.title;
  lab.append(title);
  const controls=definition.inputs.map(([name,min,max,step,value])=>{
    const label=document.createElement('label'), input=document.createElement('input'), number=document.createElement('output');
    const field=document.createElement('span');
    field.textContent=name;
    input.type='range';
    Object.assign(input,{min,max,step,value});
    label.append(field,input,number);
    lab.append(label);
    return {input,number};
  });
  let plot;
  if(definition.plot){
    plot=document.createElementNS('http://www.w3.org/2000/svg','svg');
    plot.setAttribute('viewBox','0 0 425 225');
    plot.setAttribute('role','img');
    plot.setAttribute('aria-label','Original data points, fitted line, and prediction errors');
    lab.append(plot);
  }
  const output=document.createElement('p');
  output.className='result';
  output.setAttribute('aria-live','polite');
  lab.append(output);
  function update(){
    const values=controls.map(({input,number})=>{number.value=input.value;return Number(input.value);});
    output.textContent=definition.run(values);
    if(plot)drawFit(plot,...values);
  }
  controls.forEach(({input})=>input.addEventListener('input',update));
  if(definition.reset){
    const [label,values]=definition.reset;
    const button=document.createElement('button');
    button.type='button';
    button.textContent=label;
    button.addEventListener('click',()=>{controls.forEach(({input},i)=>input.value=values[i]);update();});
    lab.append(button);
  }
  update();
}
for(const pre of document.querySelectorAll('pre')){
  const bar=document.createElement('div'), button=document.createElement('button');
  bar.className='code-toolbar';
  button.type='button';
  button.textContent='Copy block';
  button.addEventListener('click',async()=>{
    try{await navigator.clipboard.writeText(pre.innerText);button.textContent='Copied';}
    catch{
      const range=document.createRange();
      range.selectNodeContents(pre);
      const selection=window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      button.textContent='Selected for copying';
    }
    setTimeout(()=>button.textContent='Copy block',1800);
  });
  bar.append(button);
  pre.before(bar);
}

;

'use strict';
(function () {
  var pts = [[1, 2], [2, 3], [3, 5], [4, 4]];
  var wEl = document.getElementById('fit-w');
  var bEl = document.getElementById('fit-b');
  var svg = document.getElementById('fit-svg');
  if (!wEl || !bEl || !svg) return;
  function draw() {
    var w = Number(wEl.value);
    var b = Number(bEl.value);
    var loss = 0;
    pts.forEach(function (p) { loss += Math.pow(w * p[0] + b - p[1], 2); });
    loss /= pts.length;
    document.getElementById('fit-w-out').textContent = w.toFixed(1);
    document.getElementById('fit-b-out').textContent = b.toFixed(1);
    var sign = b >= 0 ? '+' : '−';
    document.getElementById('fit-readout').textContent = 'ŷ = ' + w.toFixed(1) + 'x ' + sign + ' ' + Math.abs(b).toFixed(1) + ' · MSE ' + loss.toFixed(3);
    var y1 = 170 - b * 22;
    var y2 = 170 - (w * 286 / 53 + b) * 22;
    var marks = '';
    pts.forEach(function (p) {
      var x = 38 + p[0] * 53;
      var ya = 170 - p[1] * 22;
      var yp = 170 - (w * p[0] + b) * 22;
      marks += '<line x1="' + x + '" x2="' + x + '" y1="' + ya + '" y2="' + yp + '" stroke="#D69D3B" stroke-width="2" stroke-dasharray="4 3"/>';
      marks += '<circle cx="' + x + '" cy="' + ya + '" r="5.5" fill="#5199E8"/>';
    });
    var grid = '';
    [0, 2, 4, 6].forEach(function (y) {
      grid += '<line x1="38" y1="' + (170 - y * 22) + '" x2="324" y2="' + (170 - y * 22) + '" stroke="currentColor" stroke-opacity=".12"/>';
      grid += '<text x="30" y="' + (174 - y * 22) + '" font-size="10" text-anchor="end" fill="currentColor">' + y + '</text>';
    });
    [1, 2, 3, 4, 5].forEach(function (x) {
      grid += '<text x="' + (38 + x * 53) + '" y="190" font-size="10" text-anchor="middle" fill="currentColor">' + x + '</text>';
    });
    svg.innerHTML = grid + '<text x="180" y="206" font-size="10" text-anchor="middle" fill="currentColor">House size</text><line x1="38" y1="' + y1 + '" x2="324" y2="' + y2 + '" stroke="#159A80" stroke-width="2.7"/>' + marks;
  }
  wEl.addEventListener('input', draw);
  bEl.addEventListener('input', draw);
  document.getElementById('fit-reset').addEventListener('click', function () { wEl.value = '1'; bEl.value = '0'; draw(); });
  document.getElementById('fit-best').addEventListener('click', function () { wEl.value = '0.8'; bEl.value = '1.5'; draw(); });
  draw();
})();

;

'use strict';
function m2Truth(x){return 1+0.5*x+1.5*x*x}
function m2Eval(w,x){let v=0;for(let j=w.length-1;j>=0;j--)v=v*x+w[j];return v}
function m2Fit(xs,ys,d,l){
  const n=d+1,A=Array.from({length:n},()=>Array(n).fill(0)),b=Array(n).fill(0);
  for(let i=0;i<xs.length;i++){
    const p=Array.from({length:n},(_,j)=>Math.pow(xs[i],j));
    for(let j=0;j<n;j++){b[j]+=p[j]*ys[i];for(let k=0;k<n;k++)A[j][k]+=p[j]*p[k]}
  }
  for(let j=0;j<n;j++)A[j][j]+=(j===0?0:xs.length*l)+1e-12;
  const M=A.map((r,i)=>r.concat([b[i]]));
  for(let j=0;j<n;j++){
    let h=j;for(let i=j+1;i<n;i++)if(Math.abs(M[i][j])>Math.abs(M[h][j]))h=i;
    const swap=M[j];M[j]=M[h];M[h]=swap;
    const div=M[j][j]||1e-15;
    for(let k=j;k<=n;k++)M[j][k]/=div;
    for(let i=0;i<n;i++)if(i!==j){const f=M[i][j];for(let k=j;k<=n;k++)M[i][k]-=f*M[j][k]}
  }
  return M.map(r=>r[n]);
}
function m2Loss(xs,ys,w){return xs.reduce((s,x,i)=>s+(m2Eval(w,x)-ys[i])**2,0)/xs.length}
function pct(n){return (100*n).toFixed(1)+'%'}
function sig(z){return 1/(1+Math.exp(-z))}
const evalFraudScores=[0.97,0.91,0.87,0.79,0.73,0.69,0.63,0.60,0.57,0.51,0.48,0.46,0.43,0.38,0.36,0.31,0.27,0.21,0.15,0.08];
const evalNormalScores=Array.from({length:180},(_,i)=>i<140?0.01+i*0.24/139:i<170?0.27+(i-140)*0.31/29:0.60+(i-170)*0.30/9);
function fraudAt(t){
  const tp=evalFraudScores.filter(v=>v>=t).length, fn=20-tp;
  const fp=evalNormalScores.filter(v=>v>=t).length, tn=180-fp;
  return {tp,fp,fn,tn};
}
function knnDist(a,b){return (a.x-b.x)**2+(a.y-b.y)**2}
function knnNearest(pts,q,k){return pts.map((p,i)=>({x:p.x,y:p.y,cls:p.cls,i,d:knnDist(p,q)})).sort((a,b)=>a.d-b.d).slice(0,k)}
function knnClass(pts,q,k){const n=knnNearest(pts,q,k);return n.filter(p=>p.cls===1).length>k/2?1:0}
function knnPoints(){
  const a=Array.from({length:13},(_,i)=>{const ang=Math.PI*i/12;return {x:0.12+0.62*i/12+0.009*Math.sin(i*5),y:0.47-0.27*Math.sin(ang)+0.012*Math.cos(i*3),cls:0}});
  const b=Array.from({length:13},(_,i)=>{const ang=Math.PI*i/12;return {x:0.36+0.61*i/12+0.009*Math.cos(i*4),y:0.53+0.27*Math.sin(ang)+0.012*Math.sin(i*4),cls:1}});
  return a.concat(b);
}
const svm5Configs={
  linear:{bias:-0.9999774,w:[0.000000083,-0.000474223],idx:[10,11,14,15,16,17,18,19,21,22,23,24,25,0,1,2,3,4,5,6,7,8,9]},
  broad:{gamma:0.25,bias:-4.013508,idx:[10,12,13,14,15,16,17,18,20,21,22,23,25,0,1,2,3,4,5,6,7,8,9],coef:[-1.10362,-3,-3,-0.66016,-2.3176,-3,-3,-0.24113,-1.77433,-3,-3,-2.90317,-3,3,3,3,3,3,3,3,3,3,3]},
  medium:{gamma:1,bias:-1.7005663,idx:[12,13,16,17,19,21,22,25,1,6,8],coef:[-0.6343,-1.18562,-0.79063,-0.23209,-0.43798,-0.55737,-1.86217,-1.59795,2.83682,1.64917,2.81212]},
  narrow:{gamma:5,bias:-0.6632317,idx:[10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,1,3,5,6,8],coef:[-0.19948,-0.12967,-0.29175,-0.2925,-0.19879,-0.20089,-0.21607,-0.1583,-0.18527,-0.19949,-0.21798,-0.24746,-0.35077,-0.19481,-0.20564,-0.21438,1.02201,0.63355,0.0553,0.84767,0.94472]}
};
function svm5Points(){
  const inner=Array.from({length:10},(_,i)=>{const a=2*Math.PI*i/10;return {x:.26*Math.cos(a)+.04*Math.sin(3*a),y:.26*Math.sin(a)+.03*Math.cos(2*a),cls:1}});
  const outer=Array.from({length:16},(_,i)=>{const a=2*Math.PI*i/16;return {x:.9*Math.cos(a)+.03*Math.sin(3*a),y:.82*Math.sin(a)+.02*Math.cos(4*a),cls:0}});
  return inner.concat(outer);
}
function svm5Score(pts,model,x,y){
  if(model.w)return model.bias+model.w[0]*x+model.w[1]*y;
  let sum=model.bias;
  for(let i=0;i<model.idx.length;i++){
    const p=pts[model.idx[i]];
    sum+=model.coef[i]*Math.exp(-model.gamma*((x-p.x)**2+(y-p.y)**2));
  }
  return sum;
}
const treeData=[{x:20,r:1},{x:25,r:1},{x:30,r:1},{x:35,r:0},{x:40,r:1},{x:45,r:0},{x:50,r:0},{x:55,r:0},{x:60,r:0},{x:65,r:0}];
const treeCuts=[22.5,27.5,32.5,37.5,42.5,47.5,52.5,57.5,62.5];
function treeGini(arr){if(!arr.length)return 0;const p=arr.filter(v=>v.r===1).length/arr.length;return 2*p*(1-p)}
function boost7MSE(y,p){return y.reduce((s,v,i)=>s+(v-p[i])**2,0)/y.length}
function boost7Fit(y,eta,rounds){
  const n=y.length, mean=y.reduce((a,b)=>a+b,0)/n;
  let pred=Array(n).fill(mean);
  const history=[{pred:pred.slice(),mse:boost7MSE(y,pred),cut:null,left:0,right:0}];
  for(let t=0;t<rounds;t++){
    const residual=y.map((v,i)=>v-pred[i]);
    let best={sse:Infinity,cut:0,left:0,right:0};
    for(let cut=0;cut<n-1;cut++){
      const left=residual.slice(0,cut+1), right=residual.slice(cut+1);
      const lv=left.reduce((a,b)=>a+b,0)/left.length;
      const rv=right.reduce((a,b)=>a+b,0)/right.length;
      const sse=left.reduce((s,v)=>s+(v-lv)**2,0)+right.reduce((s,v)=>s+(v-rv)**2,0);
      if(sse<best.sse)best={sse,cut,left:lv,right:rv};
    }
    pred=pred.map((p,i)=>p+eta*(i<=best.cut?best.left:best.right));
    history.push(Object.assign({pred:pred.slice(),mse:boost7MSE(y,pred)}, best));
  }
  return history;
}
const km8Points=[[.13,.20],[.19,.25],[.23,.15],[.26,.31],[.12,.34],[.29,.23],[.68,.15],[.76,.19],[.82,.29],[.66,.30],[.88,.16],[.78,.36],[.42,.70],[.51,.78],[.57,.64],[.49,.88],[.35,.82],[.61,.83]];
const km8Seeds={2:[[.17,.12],[.84,.23]],3:[[.12,.12],[.84,.13],[.32,.56]],4:[[.12,.12],[.84,.13],[.32,.56],[.65,.87]]};
function km8Assign(points,centers){
  return points.map(p=>{let best=0,d=Infinity;for(let j=0;j<centers.length;j++){const nd=(p[0]-centers[j][0])**2+(p[1]-centers[j][1])**2;if(nd<d){d=nd;best=j}}return best});
}
function km8State(k,steps){
  const pts=km8Points; let c=km8Seeds[k].map(p=>p.slice());
  for(let s=0;s<steps;s++){
    const a=km8Assign(pts,c);
    c=c.map((v,j)=>{const members=pts.filter((_,i)=>a[i]===j);return members.length?[members.reduce((t,p)=>t+p[0],0)/members.length,members.reduce((t,p)=>t+p[1],0)/members.length]:v});
  }
  const labels=km8Assign(pts,c);
  const sse=pts.reduce((sum,p,i)=>sum+(p[0]-c[labels[i]][0])**2+(p[1]-c[labels[i]][1])**2,0);
  return {centers:c,labels,sse};
}
const db8Points=[].concat(
  Array.from({length:18},(_,i)=>{const a=Math.PI*i/17;return [.12+.30*(1+Math.cos(a)),.32+.20*Math.sin(a)]}),
  Array.from({length:18},(_,i)=>{const a=Math.PI*i/17;return [.38+.30*(1-Math.cos(a)),.73-.20*Math.sin(a)]}),
  [[.08,.84],[.88,.09],[.17,.68],[.76,.91]]
);
function db8Cluster(pts,eps,minpts){
  const n=pts.length, eps2=eps*eps;
  const neighbors=pts.map((_,i)=>pts.map((p,j)=>((p[0]-pts[i][0])**2+(p[1]-pts[i][1])**2<=eps2)?j:-1).filter(j=>j>=0));
  const core=neighbors.map(v=>v.length>=minpts);
  const labels=Array(n).fill(-2); let c=0;
  for(let i=0;i<n;i++){
    if(labels[i]!==-2)continue;
    if(!core[i]){labels[i]=-1;continue}
    labels[i]=c;
    const queue=neighbors[i].slice(), seen=new Set(queue);
    for(let q=0;q<queue.length;q++){
      const j=queue[q];
      if(labels[j]===-1)labels[j]=c;
      if(labels[j]!==-2)continue;
      labels[j]=c;
      if(core[j])for(const k of neighbors[j])if(!seen.has(k)){seen.add(k);queue.push(k)}
    }
    c++;
  }
  return {labels,core,clusters:c,noise:labels.filter(x=>x===-1).length};
}
function gmmShare(x){
  const dens=(v,m)=>Math.exp(-0.5*((v-m)/1.5)**2);
  const a=dens(x,3), b=dens(x,7);
  return {a:a/(a+b), b:b/(a+b)};
}
const pca9Points=[[-2.8,-1.6],[-2.4,-.9],[-2.1,-1.5],[-1.5,-.6],[-1,-.25],[-.6,-.7],[0,.35],[.5,-.05],[.9,.7],[1.5,.75],[1.9,1.55],[2.5,1.4],[2.8,2.1]];
const pca9Mean=[pca9Points.reduce((s,p)=>s+p[0],0)/pca9Points.length,pca9Points.reduce((s,p)=>s+p[1],0)/pca9Points.length];
const pca9Centered=pca9Points.map(p=>[p[0]-pca9Mean[0],p[1]-pca9Mean[1]]);
const pca9Sxx=pca9Centered.reduce((s,p)=>s+p[0]*p[0],0);
const pca9Syy=pca9Centered.reduce((s,p)=>s+p[1]*p[1],0);
const pca9Sxy=pca9Centered.reduce((s,p)=>s+p[0]*p[1],0);
const pca9BestAngle=0.5*Math.atan2(2*pca9Sxy,pca9Sxx-pca9Syy)*180/Math.PI;
function pca9Variance(deg){const a=deg*Math.PI/180,c=Math.cos(a),s=Math.sin(a);return (pca9Sxx*c*c+2*pca9Sxy*c*s+pca9Syy*s*s)/(pca9Sxx+pca9Syy)}
function pca9Project(point,deg){
  const a=deg*Math.PI/180,c=Math.cos(a),s=Math.sin(a);
  const px=point[0]-pca9Mean[0], py=point[1]-pca9Mean[1], z=px*c+py*s;
  return {x:pca9Mean[0]+z*c,y:pca9Mean[1]+z*s};
}
function anomalyError(y){const x=1.5;return (y-0.7*x)**2/(1+0.7**2)}
const outreachRows=[[8,22,0,10],[8,18,0,14],[7,16,1,16],[7,14,1,18],[7,12,1,20],[6,10,2,22],[6,8,2,24],[5,6,3,26],[4,5,4,27],[4,4,4,28],[3,3,5,29],[3,2,5,30],[2,1,6,31]];
function outreachAt(v){const row=outreachRows[(v-20)/5];return {tp:row[0],fp:row[1],fn:row[2],tn:row[3],cost:15*row[1]+80*row[2]}}

function checkDefaults(){
  const xs=[-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1];
  const noise=[0.04,0.25,-0.20,0.18,-0.15,0.10,-0.19,0.23,-0.12];
  const ys=xs.map((x,i)=>m2Truth(x)+noise[i]);
  const w=m2Fit(xs,ys,1,0);
  const vx=Array.from({length:31},(_,i)=>-0.95+i*1.9/30);
  const vy=vx.map((x,i)=>m2Truth(x)+0.07*Math.sin(i*2.2+0.7));
  const eq=(a,b,msg)=>{if(a!==b)throw new Error(msg+': '+a+' vs '+b)};
  eq(m2Loss(xs,ys,w).toFixed(4),'0.3470','train');
  eq(m2Loss(vx,vy,w).toFixed(4),'0.2144','val');
  eq(pct(sig(1)),'73.1%','sigmoid');
  const scores=[2,1,0], m=Math.max(...scores), e=scores.map(z=>Math.exp(z-m)), tot=e.reduce((a,b)=>a+b);
  eq(e.map(v=>pct(v/tot)).join(','),'66.5%,24.5%,9.0%','softmax');
  const sp=0.8*0.10, np=0.1*0.90; eq(pct(sp/(sp+np)),'47.1%','prior');
  const fr=fraudAt(0.5); eq(fr.tn+','+fr.fp+','+fr.fn+','+fr.tp,'162,18,10,10','fraud');
  const pts=knnPoints(), near=knnNearest(pts,{x:0.55,y:0.48},3);
  eq(knnClass(pts,{x:0.55,y:0.48},3)+':'+near.filter(p=>p.cls===0).length+':'+near.filter(p=>p.cls===1).length,'0:2:1','knn');
  const spts=svm5Points(), model=svm5Configs.linear;
  const acc=spts.filter(p=>(svm5Score(spts,model,p.x,p.y)>=0?1:0)===p.cls).length;
  eq(acc+','+model.idx.length,'16,23','svm');
  const cut=42.5, left=treeData.filter(p=>p.x<=cut), right=treeData.filter(p=>p.x>cut);
  const after=left.length/10*treeGini(left)+right.length/10*treeGini(right);
  eq(treeGini(left).toFixed(3)+','+treeGini(right).toFixed(3)+','+after.toFixed(3),'0.320,0.000,0.160','tree');
  const hist=boost7Fit([2,2,4,8,8,10],0.5,8);
  eq(hist[0].mse.toFixed(3),'9.889','boost');
  eq(km8State(3,0).sse.toFixed(3),'0.908','kmeans');
  const db=db8Cluster(db8Points,0.11,4);
  eq(db.clusters+','+db.core.filter(Boolean).length+','+db.noise,'1,32,4','dbscan');
  const g=gmmShare(5.5); eq(pct(g.a)+','+pct(g.b),'29.1%,70.9%','gmm');
  eq(pct(pca9Variance(0)),'72.3%','pca');
  eq((100*pca9Variance(pca9BestAngle)).toFixed(1)+'%','98.5%','pca-max');
  eq(anomalyError(1.1).toFixed(3),'0.002','anomaly');
  eq(((52+100)/5).toFixed(1),'30.4','leak');
  eq(pct(sig(-1.6+0.62*4)),'70.7%','drift');
  const o=outreachAt(50); eq(o.tp+','+o.fp+','+o.fn+','+o.tn+','+o.cost,'6,8,2,24,280','outreach');
}

function choices(name, current, options, also, alsoVal){
  return '<div class="choices">'+options.map(o=>{
    const on=String(o[0])===String(current);
    const extra=also?` data-also="${also}" data-also-val="${alsoVal}"`:'';
    return `<button type="button" data-set="${name}" data-val="${o[0]}"${/^-?\d+(\.\d+)?$/.test(String(o[0]))?' data-num="1"':''}${extra} aria-pressed="${on}">${o[1]}</button>`;
  }).join('')+'</div>';
}
function range(name, label, min, max, step, value, digits){
  const shown=Number(value).toFixed(digits);
  return `<label><span>${label}</span><input type="range" data-set="${name}" min="${min}" max="${max}" step="${step}" value="${value}"><output>${shown}</output></label>`;
}
function stats(pairs){
  return '<div class="stats">'+pairs.map(p=>`<div class="stat"><span>${p[0]}</span><b>${p[1]}</b></div>`).join('')+'</div>';
}
function bind(root, state, draw){
  function paint(rebuild){
    const html=draw(state);
    if(rebuild||!root.querySelector('.view')){
      root.innerHTML=html;
      root.querySelectorAll('input[data-set]').forEach(node=>{
        node.addEventListener('input',()=>{state[node.dataset.set]=Number(node.value);paint(false)});
      });
      root.querySelectorAll('button[data-set]').forEach(node=>{
        node.addEventListener('click',()=>{
          state[node.dataset.set]=node.dataset.num==='1'?Number(node.dataset.val):node.dataset.val;
          if(node.dataset.also)state[node.dataset.also]=Number(node.dataset.alsoVal);
          paint(true);
        });
      });
      return;
    }
    const tmp=document.createElement('div');
    tmp.innerHTML=html;
    root.querySelector('.view').innerHTML=tmp.querySelector('.view').innerHTML;
    root.querySelectorAll('label').forEach(label=>{
      const input=label.querySelector('input'), out=label.querySelector('output');
      if(!input||!out)return;
      const step=String(input.step);
      const digits=step.indexOf('.')<0?0:step.split('.')[1].length;
      out.textContent=Number(state[input.dataset.set]).toFixed(digits);
    });
  }
  paint(true);
}

const drawers={
  poly(s){
    const xs=[-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1];
    const noise=[0.04,0.25,-0.20,0.18,-0.15,0.10,-0.19,0.23,-0.12];
    const ys=xs.map((x,i)=>m2Truth(x)+noise[i]);
    const lam=[0,0.0001,0.001,0.01,0.1][s.penalty];
    const w=m2Fit(xs,ys,s.degree,lam);
    const vx=Array.from({length:31},(_,i)=>-0.95+i*1.9/30);
    const vy=vx.map((x,i)=>m2Truth(x)+0.07*Math.sin(i*2.2+0.7));
    const px=x=>40+(x+1)*142, py=y=>176-(y-0.5)*42;
    const curve=Array.from({length:151},(_,i)=>-1+i/75);
    const truth=curve.map(x=>px(x)+','+py(m2Truth(x))).join(' ');
    const fit=curve.map(x=>px(x)+','+py(m2Eval(w,x))).join(' ');
    const dots=xs.map((x,i)=>`<circle cx="${px(x)}" cy="${py(ys[i])}" r="4.7" fill="#529BE8"/>`).join('');
    const note=s.degree===1?'A straight line misses the curvature. The model is underfitting.':s.degree===2?'A quadratic curve follows the underlying pattern without needing to match every noisy observation.':'The degree-8 model has enough flexibility to interpolate the training examples. Notice how its validation error behaves.';
    return `<h4>Choose model complexity</h4><p class="note">${s.degree+1} coefficients</p>`+
      choices('degree',s.degree,[[1,'Degree 1'],[2,'Degree 2'],[8,'Degree 8']],'penalty',0)+
      `<div class="view"><svg viewBox="0 0 340 206" role="img" aria-label="Training points, true curve, and fitted polynomial"><polyline points="${truth}" fill="none" stroke="currentColor" stroke-dasharray="5 5" opacity=".45"/><polyline points="${fit}" fill="none" stroke="#159B82" stroke-width="2.7"/>${dots}</svg>`+
      stats([['Training error (MSE)', m2Loss(xs,ys,w).toFixed(4)],['Validation error (MSE)', m2Loss(vx,vy,w).toFixed(4)]])+
      `<p>${note}</p><p class="note">λ = ${lam}</p></div><h4>Now try regularization</h4>`+
      range('penalty','Penalty strength (λ)',0,4,1,s.penalty,0)+
      `<p class="note">No penalty at the left, stronger penalty at the right.</p>`+
      `<p><button type="button" data-set="degree" data-val="8" data-num="1" data-also="penalty" data-also-val="3">Try degree 8 with regularization</button></p>`+
      `<p class="note">Calculated from a fixed synthetic dataset. Validation uses separate noisy observations. This is L2 regularization; other penalty types will behave differently.</p>`;
  },
  sigmoid(s){
    const p=sig(s.z);
    const px=z=>37+(z+6)*23.75, py=v=>164-v*130;
    const curve=Array.from({length:121},(_,i)=>-6+i/10);
    const line=curve.map(z=>px(z)+','+py(sig(z))).join(' ');
    const klass=p>=s.t?'Spam (1)':'Not spam (0)';
    return `<h4>See how sigmoid works</h4><div class="view">`+
      stats([['Model score', s.z.toFixed(1)],['P(spam)', pct(p)],['Predicted class', klass]])+
      `<svg viewBox="0 0 340 206" role="img" aria-label="Sigmoid curve with the current score and threshold"><line x1="37" x2="322" y1="${py(s.t)}" y2="${py(s.t)}" stroke="#D7A044" stroke-dasharray="5 4"/><polyline points="${line}" fill="none" stroke="#159B82" stroke-width="3"/><circle cx="${px(s.z)}" cy="${py(p)}" r="6" fill="#529BE8"/></svg></div>`+
      range('z','Change the model score',-6,6,0.2,s.z,1)+
      range('t','Classification threshold',0.1,0.9,0.05,s.t,2)+
      `<p class="note">The gold dashed line is the decision threshold. Changing it does not change the predicted probability.</p>`;
  },
  softmax(s){
    const m=Math.max(s.a,s.b,s.c), raw=[s.a,s.b,s.c].map(z=>Math.exp(z-m)), tot=raw.reduce((a,b)=>a+b);
    const names=['Sports','Technology','Politics'], colors=['#529BE8','#159B82','#D7A044'];
    const bars=raw.map((v,i)=>`<p><strong>${names[i]}</strong> ${pct(v/tot)}</p><div class="bar"><span style="width:${100*v/tot}%;background:${colors[i]}"></span></div>`).join('');
    return `<h4>Convert scores into probabilities</h4>`+
      range('a','Sports',-3,5,0.5,s.a,1)+range('b','Technology',-3,5,0.5,s.b,1)+range('c','Politics',-3,5,0.5,s.c,1)+
      `<div class="view"><h4>Softmax probabilities</h4>${bars}<p class="note">The three probabilities add up to 100%.</p></div>`;
  },
  prior(s){
    const spam=0.8*s.prior/100, normal=0.1*(1-s.prior/100), post=spam/(spam+normal);
    const left=100*spam/(spam+normal);
    return `<h4>Change the prior</h4>`+range('prior','Percentage of emails that are spam',1,50,1,s.prior,0)+
      `<div class="view"><p class="note">“free” still appears in 80% of spam and 10% of normal email.</p>`+
      stats([['P(spam | “free”)', pct(post)]])+
      `<div class="bar"><span style="width:${left}%;background:#D7A044"></span><span style="width:${100-left}%;background:#529BE8"></span></div>`+
      `<p class="note">Gold is spam containing “free”. Blue is normal email containing “free”.</p></div>`;
  },
  fraud(s){
    const r=fraudAt(s.t);
    const precision=r.tp+r.fp?r.tp/(r.tp+r.fp):0, recall=r.tp/20;
    return `<h4>Fraud classification threshold</h4><p class="note">200 transactions: 20 fraud and 180 normal.</p>`+
      range('t','Threshold',0.1,0.9,0.05,s.t,2)+
      '<div class="view">'+stats([['True negatives', r.tn],['False positives', r.fp],['False negatives', r.fn],['True positives', r.tp],['Precision', pct(precision)],['Recall', pct(recall)]])+'</div><p>A threshold changes decisions, not the underlying scores. Precision may fluctuate rather than increase monotonically as the threshold rises.</p>';
  },
  knn(s){
    const pts=knnPoints(), q={x:s.x,y:s.y}, k=s.k;
    const near=knnNearest(pts,q,k), vote=knnClass(pts,q,k);
    const blue=near.filter(p=>p.cls===0).length, orange=near.filter(p=>p.cls===1).length;
    const px=v=>18+v*300, py=v=>188-v*170;
    const cells=[];
    for(let row=0;row<15;row++)for(let col=0;col<23;col++){
      const qq={x:(col+0.5)/23,y:(row+0.5)/15};
      const cls=knnClass(pts,qq,k);
      cells.push(`<rect x="${18+col*300/23}" y="${18+row*170/15}" width="${300/23+0.4}" height="${170/15+0.4}" fill="${cls?'#D49B3C':'#529BE8'}" opacity=".22"/>`);
    }
    const dots=pts.map(p=>`<circle cx="${px(p.x)}" cy="${py(p.y)}" r="4.2" fill="${p.cls?'#D49B3C':'#529BE8'}"/>`).join('');
    const links=near.map(p=>`<line x1="${px(q.x)}" y1="${py(q.y)}" x2="${px(p.x)}" y2="${py(p.y)}" stroke="#157c64" stroke-dasharray="3 3"/>`).join('');
    return `<h4>Build your own nonlinear classifier</h4><p>Number of neighbors (k)</p>`+
      choices('k',k,[[1,'1'],[3,'3'],[7,'7'],[15,'15']])+
      `<div class="view"><svg viewBox="0 0 340 210" role="img" aria-label="k nearest neighbors on two moons">${cells.join('')}${dots}${links}<circle cx="${px(q.x)}" cy="${py(q.y)}" r="6" fill="#111"/></svg>`+
      stats([['Predicted class', vote?'Orange':'Blue'],['Votes', blue+' blue · '+orange+' orange']])+
      `</div>`+
      range('x','Query feature 1',0.02,0.98,0.02,s.x,2)+
      range('y','Query feature 2',0.02,0.98,0.02,s.y,2)+
      `<p class="note">The highlighted points are the nearest neighbors; background cells approximate the learned decision regions.</p>`;
  },
  svm(s){
    const pts=svm5Points(), model=svm5Configs[s.choice];
    const acc=pts.filter(p=>(svm5Score(pts,model,p.x,p.y)>=0?1:0)===p.cls).length;
    const cells=[];
    for(let r=0;r<18;r++)for(let c=0;c<28;c++){
      const x=-1.2+(c+0.5)*2.4/28, y=1.2-(r+0.5)*2.4/18;
      const cls=svm5Score(pts,model,x,y)>=0?1:0;
      cells.push(`<rect x="${29+c*280/28}" y="${12+r*178/18}" width="${280/28+0.3}" height="${178/18+0.3}" fill="${cls?'#D79C41':'#529BE8'}" opacity=".28"/>`);
    }
    const mapx=x=>29+(x+1.2)*280/2.4, mapy=y=>12+(1.2-y)*178/2.4;
    const sv=new Set(model.idx);
    const dots=pts.map((p,i)=>`<circle cx="${mapx(p.x)}" cy="${mapy(p.y)}" r="${sv.has(i)?6.5:4}" fill="${p.cls?'#D79C41':'#529BE8'}" stroke="${sv.has(i)?'#111':'none'}" stroke-width="1.4"/>`).join('');
    const note=s.choice==='linear'?'The linear SVM cannot separate the circular classes. Here it predicts the outer class throughout the displayed region.':s.choice==='broad'?'Low gamma creates a broad, smooth region of influence, allowing the classifier to surround the inner class.':s.choice==='medium'?'Moderate gamma also separates the training circles, with a somewhat more localized decision function.':'High gamma makes individual support vectors more locally influential. On noisier datasets, this can lead to overfitting.';
    return `<h4>Linear SVM versus RBF SVM</h4>`+
      choices('choice',s.choice,[['linear','Linear SVM'],['broad','RBF · gamma 0.25'],['medium','RBF · gamma 1'],['narrow','RBF · gamma 5']])+
      `<div class="view"><svg viewBox="0 0 340 214" role="img" aria-label="SVM decision region on two circles">${cells.join('')}${dots}</svg>`+
      stats([['Training accuracy', acc+' / 26'],['Support vectors', String(model.idx.length)]])+
      `<p>${note}</p></div><p class="note">Fitted scikit-learn SVC models on 26 fixed synthetic training points, all with C = 3. Ringed points are support vectors. Blue is the outer class and gold is the inner class.</p>`;
  },
  tree(s){
    const cut=treeCuts[s.cut], left=treeData.filter(p=>p.x<=cut), right=treeData.filter(p=>p.x>cut);
    const lg=treeGini(left), rg=treeGini(right), after=left.length/10*lg+right.length/10*rg;
    const scale=x=>37+(x-20)/45*274;
    const dots=treeData.map(p=>`<circle cx="${scale(p.x)}" cy="${p.r?46:92}" r="7" fill="${p.r?'#159B82':'#529BE8'}"/>`).join('');
    return `<h4>Move the split threshold</h4><div class="view"><p>Current question: income ≤ €${cut}k?</p>`+
      `<svg viewBox="0 0 340 120" role="img" aria-label="Ten customers split by income"><line x1="${scale(cut)}" y1="20" x2="${scale(cut)}" y2="110" stroke="#D49B3C" stroke-width="2"/>${dots}</svg>`+
      stats([['Left Gini', lg.toFixed(3)],['Right Gini', rg.toFixed(3)],['Before', '0.480'],['Weighted after', after.toFixed(3)],['Left size', String(left.length)],['Right size', String(right.length)]])+
      `</div>`+range('cut','Split index',0,8,1,s.cut,0)+
      `<p class="note">Green points churn and blue points stay. Gini is 2p(1−p) inside each side. The split shown is one of the nine midpoints between these incomes. A real tree also searches across the other available features.</p>`;
  },
  boost(s){
    const y=[2,2,4,8,8,10], hist=boost7Fit(y,s.rate,8), cur=hist[s.count];
    const bars=y.map((v,i)=>{
      const h=v*12, ph=Math.max(0,cur.pred[i]*12);
      return `<rect x="${30+i*50}" y="${150-h}" width="16" height="${h}" fill="#529BE8"/><rect x="${48+i*50}" y="${150-ph}" width="16" height="${ph}" fill="#159B82"/>`;
    }).join('');
    return `<h4>Gradient boosting regression</h4><div class="view">`+
      stats([['Training MSE', cur.mse.toFixed(3)],['Trees added', String(s.count)],['Learning rate', s.rate.toFixed(1)]])+
      `<svg viewBox="0 0 340 170" role="img" aria-label="Actual house values and boosted predictions">${bars}<text x="20" y="166" font-size="11">Blue actual, green prediction</text></svg></div>`+
      range('count','Number of added trees',0,8,1,s.count,0)+
      choices('rate',s.rate,[[0.1,'0.1'],[0.3,'0.3'],[0.5,'0.5'],[1,'1.0']])+
      `<p class="note">Every house starts at the mean, ${(y.reduce((a,b)=>a+b,0)/y.length).toFixed(2)}. Each tree is a greedy one-split fit to the remaining residuals of this six-house dataset. This is a training demonstration, not a generalization experiment.</p>`;
  },
  kmeans(s){
    const result=km8State(s.k,s.step);
    const colors=['#529BE8','#D49B3C','#159B82','#AA80D9'];
    const px=x=>28+x*280, py=y=>12+(1-y)*181;
    const lines=km8Points.map((p,i)=>`<line x1="${px(p[0])}" y1="${py(p[1])}" x2="${px(result.centers[result.labels[i]][0])}" y2="${py(result.centers[result.labels[i]][1])}" stroke="${colors[result.labels[i]]}" stroke-dasharray="3 4"/>`).join('');
    const dots=km8Points.map((p,i)=>`<circle cx="${px(p[0])}" cy="${py(p[1])}" r="5" fill="${colors[result.labels[i]]}"/>`).join('');
    const cents=result.centers.map((c,i)=>`<rect x="${px(c[0])-6}" y="${py(c[1])-6}" width="12" height="12" fill="${colors[i]}" stroke="#111"/>`).join('');
    return `<h4>Watch K-Means discover clusters</h4>`+
      choices('k',s.k,[[2,'K = 2'],[3,'K = 3'],[4,'K = 4']],'step',0)+
      `<div class="view"><svg viewBox="0 0 340 210" role="img" aria-label="K-means points and centroids">${lines}${dots}${cents}</svg>`+
      stats([['Within-cluster squared error', result.sse.toFixed(3)],['Step', s.step+' / 6']])+
      `</div>`+range('step','Update step',0,6,1,s.step,0)+
      `<p class="note">Squares are centroids. Step 0 is the seed assignment. Each later step moves the centroids to the mean of their members. Smaller squared error means tighter groups for a fixed K; comparing errors across different K values alone is misleading.</p>`;
  },
  dbscan(s){
    const result=db8Cluster(db8Points,s.eps,s.min);
    const colors=['#529BE8','#D49B3C','#159B82','#A77EDB','#D76A92'];
    const px=x=>20+x*300, py=y=>190-y*170;
    const dots=db8Points.map((p,i)=>{
      const label=result.labels[i];
      const fill=label<0?'#98a4b3':colors[label%colors.length];
      return `<circle cx="${px(p[0])}" cy="${py(p[1])}" r="${result.core[i]?5.5:4}" fill="${fill}"/>`;
    }).join('');
    return `<h4>How density changes cluster discovery</h4><div class="view">`+
      `<svg viewBox="0 0 340 210" role="img" aria-label="DBSCAN clusters and noise">${dots}</svg>`+
      stats([['Clusters', result.clusters],['Core points', result.core.filter(Boolean).length],['Noise points', result.noise]])+
      `</div>`+range('eps','Neighborhood radius (eps)',0.05,0.3,0.01,s.eps,2)+
      range('min','Minimum neighborhood size',2,8,1,s.min,0)+
      `<p class="note">Gray points are noise. A point’s neighborhood count includes the point itself.</p>`;
  },
  gmm(s){
    const share=gmmShare(s.x);
    const dens=(v,m)=>Math.exp(-0.5*((v-m)/1.5)**2);
    const px=x=>27+x*29, py=v=>145-v*109;
    const curve=(m,color)=>Array.from({length:101},(_,i)=>{const x=i/10;return px(x)+','+py(dens(x,m))}).join(' ');
    return `<h4>Two overlapping spending groups</h4><div class="view">`+
      `<svg viewBox="0 0 340 170" role="img" aria-label="Two Gaussian densities and the current customer"><polyline points="${curve(3,'a')}" fill="none" stroke="#529BE8" stroke-width="2.5"/><polyline points="${curve(7,'b')}" fill="none" stroke="#D49B3C" stroke-width="2.5"/><line x1="${px(s.x)}" y1="20" x2="${px(s.x)}" y2="150" stroke="#111" stroke-dasharray="4 3"/></svg>`+
      stats([['Cluster A', pct(share.a)],['Cluster B', pct(share.b)]])+
      `</div>`+range('x','Customer spending',0,10,0.1,s.x,1)+
      `<p class="note">Equal mixture weights and equal standard deviations, with means at 3 and 7. These are probabilities under the fitted model, not probabilities of objectively defined real-world categories.</p>`;
  },
  pca(s){
    const v=pca9Variance(s.angle), max=pca9Variance(pca9BestAngle);
    const px=x=>170+x*39, py=y=>111-y*35;
    const a=s.angle*Math.PI/180, dx=Math.cos(a), dy=Math.sin(a);
    const x0=pca9Mean[0]-dx*4, y0=pca9Mean[1]-dy*4, x1=pca9Mean[0]+dx*4, y1=pca9Mean[1]+dy*4;
    const links=pca9Points.map(p=>{const q=pca9Project(p,s.angle);return `<line x1="${px(p[0])}" y1="${py(p[1])}" x2="${px(q.x)}" y2="${py(q.y)}" stroke="#D49B3C" stroke-dasharray="3 3"/>`}).join('');
    const dots=pca9Points.map(p=>{const q=pca9Project(p,s.angle);return `<circle cx="${px(q.x)}" cy="${py(q.y)}" r="3.2" fill="#159B82"/><circle cx="${px(p[0])}" cy="${py(p[1])}" r="5" fill="#529BE8"/>`}).join('');
    return `<h4>Rotate the projection</h4><div class="view">`+
      stats([['Angle', s.angle+'°'],['Variance preserved', pct(v)],['Discarded', pct(1-v)],['Maximum', pct(max)]])+
      `<svg viewBox="0 0 340 230" role="img" aria-label="Points projected onto a rotating axis"><line x1="${px(x0)}" y1="${py(y0)}" x2="${px(x1)}" y2="${py(y1)}" stroke="#159B82" stroke-width="2"/>${links}${dots}</svg></div>`+
      range('angle','Projection angle',-90,90,1,s.angle,0)+
      `<p><button type="button" data-set="angle" data-val="0" data-num="1">Horizontal axis</button> <button type="button" data-set="angle" data-val="${Math.round(pca9BestAngle)}" data-num="1">Find PCA direction</button></p>`+
      `<p class="note">Observations are centered. The displayed fraction is the proportion of total variance retained by the selected one-dimensional projection.</p>`;
  },
  anomaly(s){
    const err=anomalyError(s.y), x=1.5, recon=(x+0.7*s.y)/(1+0.49);
    const mx=v=>165+v*46, my=v=>112-v*35;
    const far=err>0.25;
    return `<h4>How unusual is this new observation?</h4><div class="view">`+
      `<svg viewBox="0 0 340 200" role="img" aria-label="Point and its reconstruction on a fixed line"><line x1="${mx(-2)}" y1="${my(-1.4)}" x2="${mx(3)}" y2="${my(2.1)}" stroke="#159B82" stroke-width="2"/><line x1="${mx(x)}" y1="${my(s.y)}" x2="${mx(recon)}" y2="${my(0.7*recon)}" stroke="#D49B3C" stroke-dasharray="4 3"/><circle cx="${mx(x)}" cy="${my(s.y)}" r="6" fill="#529BE8"/></svg>`+
      stats([['Squared reconstruction error', err.toFixed(3)]])+
      `<p>${far?'Far from the learned linear pattern.':'Relatively close to the learned linear pattern.'}</p></div>`+
      range('y','Move the observation vertically',-2,3,0.1,s.y,1)+
      `<p><button type="button" data-set="y" data-val="1.1" data-num="1">Typical example</button> <button type="button" data-set="y" data-val="-1.8" data-num="1">Unusual example</button></p>`+
      `<p class="note">The error is squared perpendicular distance to the line. 0.25 is only a demonstration threshold.</p>`;
  },
  leakage(s){
    const leaked=(52+s.v)/5;
    return `<h4>Move the held-out observation</h4>`+
      range('v','Validation spending',10,150,1,s.v,0)+
      '<div class="view">'+stats([['Training-only mean', '13.0'],['Leaked mean', leaked.toFixed(1)]])+'</div>'+
      `<p class="note">The training-only mean uses the four training values and stays at 13.0. The leaked mean folds the held-out value into that average: (52 + spending) / 5. The leaked preprocessing statistic changes when the validation observation changes. Imputers, scalers, PCA and many feature selectors can all leak information when fitted on evaluation data.</p>`;
  },
  calibration(s){
    const pred=[0.1,0.3,0.5,0.7,0.9];
    const freq=s.mode==='good'?[0.12,0.28,0.52,0.69,0.88]:s.mode==='over'?[0.03,0.12,0.3,0.47,0.66]:[0.24,0.42,0.65,0.85,0.98];
    const X=p=>37+270*p, Y=p=>176-150*p;
    const ideal=`${X(0)},${Y(0)} ${X(1)},${Y(1)}`;
    const obs=pred.map((p,i)=>X(p)+','+Y(freq[i])).join(' ');
    const at70=Math.round(freq[3]*100);
    return `<h4>Three models with different probability quality</h4>`+
      choices('mode',s.mode,[['good','Well calibrated'],['over','Overconfident'],['under','Underconfident']])+
      `<div class="view"><svg viewBox="0 0 340 200" role="img" aria-label="Reliability curve against the ideal diagonal"><line x1="${X(0)}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(1)}" stroke="currentColor" stroke-dasharray="5 4" opacity=".4"/><polyline points="${obs}" fill="none" stroke="#159B82" stroke-width="2.6"/>${pred.map((p,i)=>`<circle cx="${X(p)}" cy="${Y(freq[i])}" r="4.5" fill="#529BE8"/>`).join('')}</svg>`+
      `<p>Among customers predicted to have 70% churn probability, ${at70}% actually churn in this scenario.</p></div>`+
      `<p class="note">Invented calibration scenarios for illustration, not measured model results. A real reliability diagram groups predictions into probability bins, and each bin's estimated frequency has sampling uncertainty. The dashed line is perfect calibration.</p>`;
  },
  drift(s){
    const model=sig(-1.6+0.62*4), reality=sig(-1.6+(0.62-s.gap)*4);
    const gap=(reality-model)*100;
    return `<h4>A frozen model meets a changed relationship</h4><div class="view">`+
      stats([['Frozen model at 4 tickets', pct(model)],['New true churn probability', pct(reality)],['Gap', gap.toFixed(1)+' percentage points']])+
      `</div>`+range('gap','Change in the ticket–churn relationship',0,1,0.01,s.gap,2)+
      `<p class="note">The model’s estimate stays fixed. The slider weakens the true effect of support tickets. This is a hypothetical scenario for teaching concept drift: the changed curve is assumed ground truth, not something observable without outcome data.</p>`;
  },
  outreach(s){
    const r=outreachAt(s.t);
    const precision=r.tp/(r.tp+r.fp), recall=r.tp/(r.tp+r.fn);
    return `<h4>Choose a business decision threshold</h4>`+
      range('t','Outreach threshold (%)',20,80,5,s.t,0)+
      '<div class="view">'+stats([['True positives', r.tp],['False positives', r.fp],['False negatives', r.fn],['True negatives', r.tn],['Precision', pct(precision)],['Recall', pct(recall)],['Error cost', '€'+r.cost]])+'</div>'+
      `<p class="note">Costs assume €15 per false positive, €80 per missed positive, and no other costs. Real outreach planning also needs intervention costs, treatment effectiveness, expected customer value, and trustworthy probability estimates.</p>`;
  }
};

const initial={
  poly:{degree:1,penalty:0},
  sigmoid:{z:1,t:0.5},
  softmax:{a:2,b:1,c:0},
  prior:{prior:10},
  fraud:{t:0.5},
  knn:{k:3,x:0.55,y:0.48},
  svm:{choice:'linear'},
  tree:{cut:4},
  boost:{count:0,rate:0.5},
  kmeans:{k:3,step:0},
  dbscan:{eps:0.11,min:4},
  gmm:{x:5.5},
  pca:{angle:0},
  anomaly:{y:1.1},
  leakage:{v:100},
  calibration:{mode:'good'},
  drift:{gap:0},
  outreach:{t:50}
};

if(typeof document==='undefined'){
  checkDefaults();
  console.log('defaults-ok');
}else{
  document.querySelectorAll('[data-widget]').forEach(root=>{
    const name=root.dataset.widget;
    const seed=initial[name];
    if(!seed||!drawers[name])return;
    bind(root, Object.assign({}, seed), drawers[name]);
  });
}

