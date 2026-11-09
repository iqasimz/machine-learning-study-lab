"use strict";
// Read-only snapshots and configuration arithmetic; no remote training backend.
(()=>{
  window.LAB_STATIC_PREVIEW=true;
  const base=new URL("./",document.currentScript.src),originalFetch=window.fetch.bind(window);
  let dataPromise;
  const data=()=>dataPromise||(dataPromise=originalFetch(new URL("api-data.json",base)).then(r=>{if(!r.ok)throw Error("Course files could not load.");return r.json();}));
  const reply=(value,status=200)=>new Response(JSON.stringify(value),{status,headers:{"Content-Type":"application/json"}});
  function plan(registry,payload){
    const guide=registry.guides.find(g=>g.id===payload.guide);
    if(!guide)throw Error("Unknown guided experiment.");
    const validate=overrides=>{
      const s={...registry.defaults,...overrides};
      for(const [key,value] of Object.entries(s))if(!registry.parameters[key]?.choices.includes(value))throw Error("Choose a listed value for "+key+".");
      if(s.width%s.heads)throw Error("Width must be divisible by attention heads.");
      if(s.warmup_steps>=s.schedule_steps)throw Error("Warmup must be shorter than the schedule horizon.");
      if(s.steps*s.batch_size*s.accumulation*s.block_size*s.layers*s.width**2>20000000000)throw Error("This combination exceeds the local work ceiling. Reduce updates, batch, context or model size.");
      return s;
    };
    const settings=validate({...guide.overrides,...payload.overrides});
    const variations=payload.variation==="all"?guide.variations:Number.isInteger(payload.variation)&&payload.variation>=0&&payload.variation<guide.variations.length?[guide.variations[payload.variation]]:null;
    if(!variations)throw Error("Choose all or a valid variation index.");
    return {guide:guide.id,runs:variations.map(v=>{
      const s=validate({...settings,[guide.parameter]:v.value});
      return {label:v.label,settings:s,estimate:{effective_windows_per_update:s.batch_size*s.accumulation,maximum_token_exposures:s.steps*s.batch_size*s.accumulation*s.block_size,attention_scores_per_layer_microbatch:s.batch_size*s.heads*s.block_size**2}};
    })};
  }
  window.fetch=async(input,options={})=>{
    const url=new URL(typeof input==="string"?input:input.url,location.href);
    if(url.origin!==location.origin||!url.pathname.startsWith("/api/"))return originalFetch(input,options);
    const method=(options.method||input?.method||"GET").toUpperCase(),key=decodeURIComponent(url.pathname);
    // Planning only performs arithmetic. All state-changing requests stay blocked.
    if(method==="POST"&&key==="/api/preview"){
      try{return reply(plan((await data())["/api/catalog"],JSON.parse(options.body)));}
      catch(error){return reply({error:error.message},400);}
    }
    if(method!=="GET")return reply({error:"Training, downloads and saved-model operations require the local installation. You can explore settings and code online."},405);
    const snapshots=await data(),found=Object.hasOwn(snapshots,key);
    return reply(found?snapshots[key]:{error:"This feature requires the local installation."},found?200:404);
  };
  document.addEventListener("DOMContentLoaded",()=>{
    const ids=["run-training","cancel","model-start","model-stop","generate","inspect","inspect-update","export","resume","tokenize","compare-run","model-compare-start"];
    const disable=()=>{for(const id of ids){const button=document.getElementById(id);if(!button)continue;if(!button.disabled)button.disabled=true;button.title="Available in the local installation only";}};
    disable();
    new MutationObserver(disable).observe(document.getElementById("main"),{subtree:true,childList:true,attributes:true,attributeFilter:["disabled"]});
    for(const id of ["train","model-results","model-compare","explore","compare"]){
      const section=document.getElementById(id);if(!section)continue;
      const note=document.createElement("p");note.className="local-only-note";note.textContent=id==="train"?"Explore both training methods, change settings and inspect the matching code below. Training is local only: install the repository to run an experiment on your own hardware.":"Saved results and comparisons come from your local experiments. This online site does not run training or upload your data.";
      section.prepend(note);
    }
  });
})();
