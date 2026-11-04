"use strict";

// Deterministic teaching calculations. These are not trained weights or run artifacts.
const ArchitectureMath=(()=>{
  const matrix=(r,c,f)=>Array.from({length:r},(_,i)=>Array.from({length:c},(_,j)=>f(i,j)));
  const sum=a=>a.reduce((x,y)=>x+y,0);
  const dot=(a,b)=>sum(a.map((x,i)=>x*b[i]));
  const add=(a,b)=>a.map((row,i)=>row.map((v,j)=>v+b[i][j]));
  const mm=(a,b)=>a.map(row=>b[0].map((_,j)=>sum(row.map((v,k)=>v*b[k][j]))));
  const softmax=a=>{const m=Math.max(...a),e=a.map(v=>v===-Infinity?0:Math.exp(v-m)),z=sum(e);if(!z)throw Error("A probability row needs at least one allowed value.");return e.map(v=>v/z);};
  const norm=a=>a.map(row=>{const mean=sum(row)/row.length,variance=sum(row.map(v=>(v-mean)**2))/row.length;return row.map(v=>(v-mean)/Math.sqrt(variance+1e-5));});
  const gelu=x=>.5*x*(1+Math.tanh(Math.sqrt(2/Math.PI)*(x+.044715*x**3)));
  const weights=(r,c,seed)=>matrix(r,c,(i,j)=>Math.sin((i+1)*1.31+(j+1)*.73+seed)*.24);
  const vocabulary=["How","to","predict","text","tokens","<end>"];
  const embedding=[[.2,-.4,.6,.1],[.3,.1,-.2,.5],[.7,-.1,.2,-.3],[-.2,.6,.1,.4],[.5,.2,-.5,.3],[.1,-.3,.4,-.2]];
  function transformer(options={}){
    const cfg={heads:2,layers:2,block:0,position:2,head:0,ids:[0,1,2],...options},ids=[...cfg.ids];
    if(![1,2,4].includes(cfg.heads)||![1,2,3].includes(cfg.layers)||ids.length<1||ids.length>6||ids.some(i=>!Number.isInteger(i)||i<0||i>=6))throw Error("Use 1/2/4 heads, 1–3 blocks and 1–6 known token IDs.");
    if(!Number.isInteger(cfg.block)||cfg.block<0||cfg.block>=cfg.layers||!Number.isInteger(cfg.position)||cfg.position<0||cfg.position>=ids.length||!Number.isInteger(cfg.head)||cfg.head<0||cfg.head>=cfg.heads)throw Error("Choose an existing block, token and head.");
    const positions=ids.map((_,i)=>[.05*i,-.03*i,.02*i,.01*i]),embedded=ids.map(i=>embedding[i].slice());
    let x=add(embedded,positions);const blocks=[],d=4/cfg.heads;
    for(let l=0;l<cfg.layers;l++){
      const wq=weights(4,4,1+l),wk=weights(4,4,3+l),wv=weights(4,4,5+l),wo=weights(4,4,7+l),w1=weights(4,8,9+l),w2=weights(8,4,11+l);
      const normalized=norm(x),q=mm(normalized,wq),k=mm(normalized,wk),v=mm(normalized,wv);
      const scores=[],masked=[],attention=[],heads=[];
      for(let h=0;h<cfg.heads;h++){
        const s=matrix(ids.length,ids.length,(i,j)=>dot(q[i].slice(h*d,(h+1)*d),k[j].slice(h*d,(h+1)*d))/Math.sqrt(d));
        const m=s.map((row,i)=>row.map((value,j)=>j>i?-Infinity:value)),a=m.map(softmax);
        scores.push(s);masked.push(m);attention.push(a);
        heads.push(a.map(row=>Array.from({length:d},(_,j)=>sum(row.map((p,t)=>p*v[t][h*d+j])))));
      }
      const joined=matrix(ids.length,4,(i,j)=>heads[Math.floor(j/d)][i][j%d]),projected=mm(joined,wo),residual=add(x,projected);
      const norm2=norm(residual),expanded=mm(norm2,w1),activated=expanded.map(row=>row.map(gelu)),mlp=mm(activated,w2),output=add(residual,mlp);
      blocks.push({input:x,normalized,q,k,v,scores,masked,attention,heads,joined,projected,residual,norm2,expanded,activated,mlp,output,weights:{wq,wk,wv,wo,w1,w2}});
      x=output;
    }
    const final=norm(x),headWeights=weights(4,6,19),logits=mm(final,headWeights),probabilities=logits.map(softmax);
    const target=cfg.position<ids.length-1?ids[cfg.position+1]:3,p=probabilities[cfg.position],loss=-Math.log(p[target]);
    const gradient=matrix(4,6,(i,j)=>final[cfg.position][i]*(p[j]-(j===target?1:0)));
    const updated=headWeights.map((row,i)=>row.map((w,j)=>w-.1*gradient[i][j])),updatedLogits=mm([final[cfg.position]],updated)[0];
    const generation=probabilities.at(-1),nextId=generation.indexOf(Math.max(...generation));
    return {kind:"teaching",cfg,ids,vocabulary,embedding:embedded,positions,start:add(embedded,positions),blocks,final,headWeights,logits,probabilities,target,loss,gradient,updatedHead:updated,updatedLoss:-Math.log(softmax(updatedLogits)[target]),nextId};
  }
  function conv(input,kernels,bias,stride=1,padding=1){
    const channels=input.length,size=input[0].length,k=kernels[0][0].length,n=Math.floor((size+2*padding-k)/stride)+1;
    if(n<1)throw Error("Filter does not fit the input.");
    return kernels.map((filter,o)=>matrix(n,n,(r,c)=>bias[o]+sum(Array.from({length:channels},(_,ch)=>sum(filter[ch].flatMap((row,i)=>row.map((w,j)=>w*(input[ch][r*stride+i-padding]?.[c*stride+j-padding]??0))))))));
  }
  const relu=x=>x.map(channel=>channel.map(row=>row.map(v=>Math.max(0,v))));
  const pool=x=>x.map(channel=>matrix(Math.floor(channel.length/2),Math.floor(channel[0].length/2),(r,c)=>Math.max(channel[2*r][2*c],channel[2*r][2*c+1],channel[2*r+1][2*c],channel[2*r+1][2*c+1])));
  function cnn(options={}){
    const cfg={pattern:"vertical",stride:1,kernel:3,filter:0,row:2,col:2,...options};
    if(!["vertical","horizontal","cross"].includes(cfg.pattern)||![1,2].includes(cfg.stride)||![3,5].includes(cfg.kernel)||![0,1].includes(cfg.filter))throw Error("Choose one of the bounded CNN examples.");
    const input=[matrix(8,8,(r,c)=>cfg.pattern==="vertical"?(c===3||c===4?1:0):cfg.pattern==="horizontal"?(r===3||r===4?1:0):(r===3||c===3?1:0))],padding=(cfg.kernel-1)/2;
    const f1=Array.from({length:2},(_,f)=>[matrix(cfg.kernel,cfg.kernel,(r,c)=>(f===0?(c===0?1:c===cfg.kernel-1?-1:0):(r===0?1:r===cfg.kernel-1?-1:0))/cfg.kernel)]),b1=[.1,.05];
    const z1=conv(input,f1,b1,cfg.stride,padding),a1=relu(z1),p1=pool(a1),f2=Array.from({length:4},(_,o)=>Array.from({length:2},(_,ch)=>matrix(3,3,(r,c)=>.12*Math.sin((o+1)*.9+(ch+1)*.6+r+c)))),b2=[.02,.04,.06,.08];
    const z2=conv(p1,f2,b2),a2=relu(z2),p2=pool(a2),flat=p2.flat(2),w=weights(flat.length,3,23),bias=[.02,-.03,.01],logits=mm([flat],w)[0].map((v,j)=>v+bias[j]),probabilities=softmax(logits);
    if(!Number.isInteger(cfg.row)||!Number.isInteger(cfg.col)||cfg.row<0||cfg.col<0||cfg.row>=z1[0].length||cfg.col>=z1[0][0].length)throw Error("Choose an existing first-layer output cell.");
    const patch=matrix(cfg.kernel,cfg.kernel,(r,c)=>input[0][cfg.row*cfg.stride+r-padding]?.[cfg.col*cfg.stride+c-padding]??0),products=matrix(cfg.kernel,cfg.kernel,(r,c)=>patch[r][c]*f1[cfg.filter][0][r][c]);
    const target=1,loss=-Math.log(probabilities[target]),gradient=matrix(flat.length,3,(i,j)=>flat[i]*(probabilities[j]-(j===target?1:0)));
    const updated=w.map((row,i)=>row.map((v,j)=>v-.1*gradient[i][j])),updatedBias=bias.map((v,j)=>v-.1*(probabilities[j]-(j===target?1:0)));
    const after=mm([flat],updated)[0].map((v,j)=>v+updatedBias[j]);
    const parameters={conv1:2*(cfg.kernel*cfg.kernel+1),conv2:4*(2*3*3+1),head:3*(flat.length+1)};
    return {kind:"teaching",cfg,input,padding,f1,b1,z1,a1,p1,f2,b2,z2,a2,p2,flat,w,bias,logits,probabilities,patch,products,target,loss,gradient,updatedLoss:-Math.log(softmax(after)[target]),parameters,totalParameters:sum(Object.values(parameters))};
  }
  const fmt=v=>Number.isFinite(v)?Math.abs(v)<.00005?"0":Number(v.toFixed(4)).toString():v===-Infinity?"−∞":"—";
  const vector=a=>"["+a.map(fmt).join(", ")+"]";
  return {transformer,cnn,softmax,norm,mm,conv,pool,gelu,fmt,vector};
})();
