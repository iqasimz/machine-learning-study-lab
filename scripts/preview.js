"use strict";
// Static export adapter: only read-only course snapshots, never a remote trainer.
(()=>{
  const base=new URL("./",document.currentScript.src),originalFetch=window.fetch.bind(window);
  let dataPromise;
  const data=()=>dataPromise||(dataPromise=originalFetch(new URL("api-data.json",base)).then(r=>{if(!r.ok)throw Error("Course files could not load.");return r.json();}));
  window.fetch=async(input,options={})=>{
    const url=new URL(typeof input==="string"?input:input.url,location.href);
    if(url.origin!==location.origin||!url.pathname.startsWith("/api/"))return originalFetch(input,options);
    if((options.method||"GET").toUpperCase()!=="GET")return new Response(JSON.stringify({error:"This is the online course preview. Install and run the repository locally to use training, inference and experiment comparisons."}),{status:405,headers:{"Content-Type":"application/json"}});
    const snapshots=await data(),key=decodeURIComponent(url.pathname),found=Object.hasOwn(snapshots,key);
    return new Response(JSON.stringify(found?snapshots[key]:{error:"This feature requires the local installation."}),{status:found?200:404,headers:{"Content-Type":"application/json"}});
  };
  document.addEventListener("DOMContentLoaded",()=>{
    const ids=["run-training","cancel","model-start","model-stop","generate","inspect","inspect-update","export","resume","tokenize","compare-run","model-compare-start"];
    for(const id of ids){const button=document.getElementById(id);if(!button)continue;button.title="Requires the local installation";button.addEventListener("click",e=>{e.preventDefault();e.stopImmediatePropagation();alert("Training and saved-model operations run in the local installation. This website contains the complete courses and interactive quizzes.");},true);}
  });
})();
