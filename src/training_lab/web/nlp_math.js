"use strict";

// Small authored tensors and counts. No checkpoint, network, random training or learner data.
const NLPMath=(()=>{
  const sigmoid=x=>1/(1+Math.exp(-x)),sum=a=>a.reduce((x,y)=>x+y,0),dot=(a,b)=>sum(a.map((x,i)=>x*b[i]));
  const softmax=ArchitectureMath.softmax;
  function recurrent(cell="rnn",directions=2,forget=.8,reset=.5){
    const inputs=[1,0,-1];
    function pass(xs,w,u){let h=0,c=0;return xs.map(x=>{
      const hp=h,cp=c;let candidate,i=.5,o=.75,z=.25;
      if(cell==="rnn"){candidate=Math.tanh(w*x+u*h);h=candidate;}
      else if(cell==="lstm"){candidate=Math.tanh(w*x+.1*h);c=forget*c+i*candidate;h=o*Math.tanh(c);}
      else if(cell==="gru"){candidate=Math.tanh(w*x+reset*(u*h));h=z*h+(1-z)*candidate;}
      else throw Error("Unknown recurrent cell");
      return {x,hp,cp,c,h,candidate,i,o,z,f:forget,r:reset};
    });}
    const forward=pass(inputs,.5,.8),backward=directions===2?pass([...inputs].reverse(),.3,.6).reverse():null;
    const output=forward.map((s,t)=>backward?[s.h,backward[t].h]:[s.h]);
    const terminal=backward?[forward.at(-1).h,backward[0].h]:[forward.at(-1).h];
    return {inputs,forward,backward,output,terminal};
  }
  function tfidf(term="cat"){
    const docs=[["cat","sat"],["cat","slept"],["dog","sat"]],vocabulary=["cat","dog","sat","slept"];
    const df=vocabulary.map(t=>docs.filter(d=>d.includes(t)).length),idf=df.map(n=>Math.log(4/(1+n))+1);
    const raw=docs.map(d=>vocabulary.map((t,i)=>d.filter(x=>x===t).length*idf[i]));
    const vectors=raw.map(v=>{const norm=Math.sqrt(dot(v,v));return v.map(x=>x/norm);});
    return {docs,vocabulary,df,idf,raw,vectors,index:vocabulary.indexOf(term)};
  }
  function classify(model="nb",alpha=1){
    const counts=[[3,1],[1,3]],query=[2,1],prior=.5;
    const probabilities=counts.map(row=>row.map(n=>(n+alpha)/(4+2*alpha)));
    const logJoint=probabilities.map(row=>Math.log(prior)+sum(row.map((p,i)=>query[i]*Math.log(p))));
    return {counts,query,probabilities,logJoint,posterior:softmax(logJoint),logistic:sigmoid(1),model};
  }
  function ngram(word="cats",alpha=1){const counts={cats:2,dogs:1,UNK:0},p=(counts[word]+alpha)/(3+7*alpha);return {p,nll:p?-Math.log(p):null,count:counts[word],total:3,vocabulary:7};}
  function skipgram(){const positive=dot([1,0],[.5,.5]),negative=dot([1,0],[0,1]);return {positive,negative,loss:-Math.log(sigmoid(positive))-Math.log(sigmoid(-negative))};}
  function cnn(pool="max"){const inputs=[1,-1,2],kernel=[.5,-.5],convolution=[dot(inputs.slice(0,2),kernel),dot(inputs.slice(1),kernel)],activated=convolution.map(x=>Math.max(0,x));return {inputs,kernel,convolution,activated,pooled:pool==="max"?Math.max(...activated):sum(activated)/2};}
  function tagger(kind="crf"){
    const emissions=kind==="crf"?[[2,1],[0,3],[2,0]]:[[Math.log(.8),Math.log(.3)],[Math.log(.2),Math.log(.7)],[Math.log(.8),Math.log(.3)]];
    const transitions=kind==="crf"?[[.5,-1],[-.5,.8]]:[[Math.log(.7),Math.log(.3)],[Math.log(.4),Math.log(.6)]];
    const start=kind==="crf"?[0,0]:[Math.log(.6),Math.log(.4)];
    const scores=[],backs=[];scores.push(emissions[0].map((e,j)=>e+start[j]));
    for(let t=1;t<3;t++){backs[t]=[];scores[t]=[0,1].map(j=>{const previous=[0,1].map(i=>scores[t-1][i]+transitions[i][j]);const winner=previous[0]>=previous[1]?0:1;backs[t][j]=winner;return previous[winner]+emissions[t][j];});}
    let last=scores[2][0]>=scores[2][1]?0:1;const best=[last];for(let t=2;t>0;t--){last=backs[t][last];best.unshift(last);}
    const paths=[];for(let code=0;code<8;code++){const tags=[(code>>2)&1,(code>>1)&1,code&1];let score=start[tags[0]];for(let t=0;t<3;t++)score+=emissions[t][tags[t]]+(t?transitions[tags[t-1]][tags[t]]:0);paths.push({tags,score});}
    const top=Math.max(...paths.map(p=>p.score)),logZ=top+Math.log(sum(paths.map(p=>Math.exp(p.score-top))));
    const bestScore=Math.max(...scores[2]);return {emissions,transitions,start,scores,backs,best,paths,logZ,bestScore,probability:Math.exp(bestScore-logZ)};
  }
  function pooling(includePadding=false){const vectors=[[1,0],[0,1],[2,-1]],mask=[1,1,0],denominator=includePadding?3:2,selected=includePadding?vectors:vectors.slice(0,2);const pooled=[0,1].map(j=>sum(selected.map(v=>v[j]))/denominator),norm=Math.hypot(...pooled);return {vectors,mask,pooled,normalized:pooled.map(x=>x/norm),denominator};}
  function sentence(temperature=.5,includePadding=false){const q=pooling(includePadding).normalized,documents=[[.8,.6],[0,1]],scores=documents.map(d=>dot(q,d)),p=softmax(scores.map(s=>s/temperature));return {q,documents,scores,p,loss:-Math.log(p[0]),late:[2,1.4]};}
  function translation(candidate="short"){
    const reference=["the","cat","sat","down"],words=candidate==="exact"?reference:candidate==="repeat"?["the","cat","sat","sat"]:["the","cat","sat"];
    const available={};reference.forEach(w=>available[w]=(available[w]||0)+1);let hits=0;for(const w of words)if(available[w]){hits++;available[w]--;}
    const precision=hits/words.length,bp=Math.min(1,Math.exp(1-reference.length/words.length));return {reference,words,hits,precision,bp,bleu1:bp*precision};
  }
  function qa(span="partial"){
    const words=["The","red","car","stopped","."],gold=[1,2],prediction=span==="exact"?[1,2]:span==="wide"?[0,2]:[2,2];
    const overlap=Math.max(0,Math.min(gold[1],prediction[1])-Math.max(gold[0],prediction[0])+1),p=overlap/(prediction[1]-prediction[0]+1),r=overlap/2;
    return {words,gold,prediction,overlap,precision:p,recall:r,f1:2*p*r/(p+r),em:Number(prediction[0]===1&&prediction[1]===2)};
  }
  function chunks(overlap=2){const length=8,width=4,stride=width-overlap,windows=[];for(let i=0;i<=length-width;i+=stride)windows.push([i,i+width-1]);const local=Array.from({length},(_,i)=>Array.from({length},(_,j)=>Math.abs(i-j)<=1)),global=local.map((row,i)=>row.map((v,j)=>v||i===0||j===0));return {windows,exposures:windows.length*width,unique:length,coversBoundary:windows.some(([s,e])=>s<=3&&e>=4),local,global,localPairs:sum(local.flat().map(Number)),globalPairs:sum(global.flat().map(Number))};}
  function gruPlacement(){const h=[1,2],r=[.2,.8],W=[[1,2],[3,4]],b=[.5,-.5],input=[.3,-.2];return {after:W.map((row,i)=>Math.tanh(input[i]+r[i]*(dot(row,h)+b[i]))),before:W.map((row,i)=>Math.tanh(input[i]+dot(row,h.map((x,j)=>x*r[j]))+b[i]))};}
  return {sigmoid,sum,dot,softmax,recurrent,tfidf,classify,ngram,skipgram,cnn,tagger,pooling,sentence,translation,qa,chunks,gruPlacement};
})();
