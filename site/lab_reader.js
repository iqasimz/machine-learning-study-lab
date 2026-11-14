"use strict";
// Shared lab helpers only. Course lessons are served by their own HTML pages.
const LabLearning = (()=>{
  function formula(node,tex,display=false){
    try{katex.render(tex,node,{displayMode:display,output:"mathml",trust:false,throwOnError:true,strict:"ignore"});}
    catch{node.textContent=tex;node.className="math-error";}
  }
  function openMetric(kind){
    location.href=kind==="language"?"courses/llm.html#ch-8":"courses/ml.html#lecture-04";
  }
  return {init:async()=>{},hide(){},setPage(){},openMetric,formula};
})();
