"use strict";
const $=id=>document.getElementById(id);
const make=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;};
const option=(value,label)=>{const n=make("option",label);n.value=value;return n;};
const palette=["#315d9c","#b87530","#79649a","#3e7861"];
const coursePages=["basics","ml","dl","language","nlp","training","post"],interviewPages=["interview","ml-interview","dl-interview","nlp-interview"];
const pageContexts={home:"Overview",basics:"Shared foundations",ml:"Machine learning · Course",dl:"Deep learning · Course",language:"NLP · Foundations",nlp:"Natural language processing · Course",training:"LLMs · Training",post:"LLMs · Post-training",visuals:"LLMs · Diagrams",train:"LLMs · Local experiments",explore:"LLMs · Saved experiments",compare:"LLMs · Analysis",interview:"LLMs · Interviews"};
const titles={home:"Machine Learning Study Lab",basics:"Shared foundations",ml:"Machine learning",dl:"Deep learning",language:"Language foundations",nlp:"Natural language processing",training:"LLM training",post:"Post-training and systems",visuals:"Model diagrams",train:"Language-model training lab",explore:"Language-model results",compare:"Compare language-model runs",interview:"LLM interview questions"};
titles["nlp-interview"]="NLP interview questions";pageContexts["nlp-interview"]="NLP · Interviews";
titles.interactive="Interactive mode";pageContexts.interactive="Guided courses";
for(const area of ["ml","dl"]){for(const [key,label] of Object.entries({lab:"Training lab",results:"Saved results",compare:"Compare models",interview:"Interview questions"})){titles[area+"-"+key]=label;pageContexts[area+"-"+key]=(area==="ml"?"Machine learning":"Deep learning")+" · "+label;}}
let page="home",engine="lora",tab="results",registry,adaptation,runList=[],selectedRun=null,monitorRun=null,run=null,monitor=null,job={running:false},comparisonSerial=0;
let plan=null,previewSerial=0,refreshing=false,comparison=null,lastTrace=null,exampleScene=null,architectureScene=null,targetScene=null;
let localHardware=null;
const selectedComparisons=new Set();
const pretty=value=>typeof value==="boolean"?value?"On":"Off":String(value);
const fixed=value=>Number.isFinite(value)?Math.abs(value)>0&&Math.abs(value)<.001?value.toExponential(2):value.toFixed(3):"—";
const elapsed=value=>!Number.isFinite(value)?"—":value<60?Math.round(value)+" s":Math.floor(value/60)+" m "+Math.round(value%60)+" s";
const statusLabel=s=>({MEASURED:"Completed",RUNNING:"Training",QUEUED:"Waiting",FAILED:"Failed",CANCELLED:"Stopped"}[s]||s);
async function api(url,body){const response=await fetch(url,body===undefined?{}:{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});const data=await response.json();if(!response.ok)throw Error(data.error||"Request failed.");return data;}
function error(where,value){$(where).textContent=value;}
function show(name,fromReader=false){
  if(page==="compare"&&name!=="compare")comparisonSerial++;
  if(page!==name){LabLearning.hide();exampleScene?.destroy();exampleScene=null;lastTrace?.destroy();lastTrace=null;targetScene?.destroy();targetScene=null;}
  page=name;const modelPage=/^(ml|dl)-(lab|results|compare)$/.test(name);
  const target=coursePages.includes(name)||name==="reference"?"course":interviewPages.includes(name)?"interview":modelPage?"model-"+name.split("-")[1]:name;
  document.querySelectorAll(".page").forEach(n=>n.hidden=n.id!==target);
  document.querySelectorAll("[data-page]").forEach(n=>{const active=n.dataset.page===name;n.classList.toggle("active",active);if(active&&!["basics","training","post","interview"].includes(name))n.setAttribute("aria-current","page");else n.removeAttribute("aria-current");});
  LabLearning.setPage(name,fromReader);
  $("page-title").textContent=titles[name]||"Course reference";$("page-context").textContent=pageContexts[name]||"Reference";document.title=name==="home"?"Machine Learning Study Lab":(titles[name]||"Reference")+" · Machine Learning Study Lab";
  const area=["nlp","language","nlp-interview"].includes(name)?"nlp":["ml","dl"].includes(name)?name:/^(ml|dl)-/.test(name)?name.split("-")[0]:["training","post","visuals","train","explore","compare","interview"].includes(name)?"llm":null;
  if(area){$("subject-"+area).hidden=false;document.querySelectorAll("[data-subject]").forEach(n=>{if(n.dataset.subject===area)n.setAttribute("aria-expanded","true");});}
  if(coursePages.includes(name)&&!fromReader)LabLearning.showStage(name);
  if(interviewPages.includes(name)&&!fromReader)LabLearning.showInterview(name==="interview"?"llm":name.split("-")[0]);
  ModelLab.hide();if(modelPage)ModelLab.show(name.split("-")[0],name.split("-")[1]);
  GuidedCourse.hide();if(name==="interactive")GuidedCourse.show();
  if(name==="visuals")mountExample();
  if(name==="train"){renderLive();mountTrainingStructure();}
  if(name==="explore"){renderRun();showTab(tab);}
  if(name==="compare"){renderCompareChoices();drawComparison();}
  if(["train","explore","compare"].includes(name))refresh();
  if(name!=="train"){architectureScene?.destroy();architectureScene=null;}
}
function showTab(name){tab=name;document.querySelectorAll(".run-tab").forEach(n=>n.hidden=n.id!=="tab-"+name);document.querySelectorAll("[data-tab]").forEach(n=>{n.classList.toggle("active",n.dataset.tab===name);n.setAttribute("aria-pressed",String(n.dataset.tab===name));});if(page==="explore"){drawRun();if(name==="tokens"){targetScene?.destroy();targetScene=LabVisuals.mount($("targets-visual"),"targets");}else{targetScene?.destroy();targetScene=null;}}}
function mountExample(){
  exampleScene?.destroy();
  const map={architecture:["transformer"],architectures:["T4"],attention:["DL08","s1"],masking:["T1","s1"],data:["T2","s3"],"training-loop":["T0"],batching:["T5"],normalization:["transformer","norm1"],lora:["R1"],preferences:["R2"],inference:["R6"],distributed:["T7"]},route=map[$("visual-select").value];
  exampleScene=route?ArchitectureExplorer.mount($("visual-stage"),route[0]):LabVisuals.mount($("visual-stage"),$("visual-select").value,{headingLevel:2});
  if(route?.[1])exampleScene.choose(route[1]);
}

const adaptationHelp={steps:["Weight updates","One update adjusts the adapter weights after the selected number of small batches. More is not automatically better."],learning_rate:["Learning rate","The scale of a weight adjustment. Too large can damage useful behaviour; too small can make progress slow."],block_size:["Maximum tokens per example","A token is a piece of text. Longer examples provide more context but use more memory. Prompts that leave no response room are skipped."],accumulation:["Small batches per update","Each small batch contains one example. Combine their gradients before one weight update to increase the effective batch without storing all activations together."],rank:["Adapter rank","The inner width of each low-rank update matrix. A larger rank trains more numbers and uses more memory; the original model remains frozen."],seed:["Random seed","Sets the starting sequence for random choices. Repeat seeds to study variation, not to guarantee the same results on every device."],device:["Local processor","Auto selects the Mac GPU (MPS) when available, otherwise CPU. No cloud worker is started."],examples:["Training examples","Maximum retained training exchanges from the licensed conversation dataset. Development/test exchanges are separate for this local adaptation, but may have been seen by the original pretrained model."]};
const trainingGuideNames={first:"Training duration: a first experiment",rate:"Compare learning rates",overfit:"Training duration and overfitting",dropout:"Compare dropout strengths",batch:"Compare microbatch sizes",accumulate:"Compare gradient accumulation",context:"Compare context-window lengths",width:"Compare hidden widths",depth:"Compare decoder depths",heads:"Compare attention-head counts",tokenizer:"Compare byte and merged tokenization",optimizer:"Compare SGD and AdamW",schedule:"Constant versus decaying learning rate",warmup:"Compare learning-rate warmup",decay:"Compare weight-decay strengths",positions:"With and without position embeddings",residuals:"With and without residual connections",normalization:"Compare normalization methods",data:"Compare training-data amounts",seeds:"Repeat a setup with different seeds"};
adaptationHelp.device[1]="Auto checks NVIDIA CUDA, Apple Metal, then CPU. The installed PyTorch build determines GPU support. No cloud worker is started.";
function buildControls(host,prefix,parameters,values,onchange,locked){
  host.replaceChildren();
  for(const [key,p] of Object.entries(parameters)){
    const block=make("div",undefined,"setting"),help=prefix==="adapt"?adaptationHelp[key]:[p.label,p.explanation];
    const label=make("label",help[0]),select=make("select");select.id=prefix+"-"+key;
    select.append(...p.choices.map(v=>option(JSON.stringify(v),pretty(v))));select.value=JSON.stringify(values[key]);select.disabled=key===locked;
    if(key==="device"&&localHardware)for(const o of select.options){const d=JSON.parse(o.value);if(["cuda","mps"].includes(d))o.disabled=!localHardware[d]||Boolean(localHardware.probe_errors?.[d]);}
    const description=make("p",help[1],"field-help");description.id=prefix+"-"+key+"-help";
    label.setAttribute("for",select.id);select.setAttribute("aria-describedby",description.id);
    select.onchange=onchange;block.append(label,description,select);
    if(prefix==="scratch"){
      const details=make("details");details.append(make("summary","Effects, interactions and code"));
      const effects=make("ul");[...p.effects,...p.interactions].forEach(t=>effects.append(make("li",t)));details.append(effects,make("pre",p.equation));
      const code=make("details");code.append(make("summary","Actual implementation"),make("pre",p.source.code));details.append(code);block.append(details);
    }
    host.append(block);
  }
}
function settings(prefix){const defs=prefix==="adapt"?adaptation.choices:registry.parameters;return Object.fromEntries(Object.keys(defs).map(k=>[k,JSON.parse($(prefix+"-"+k).value)]));}
function loadGuide(){
  const guide=registry.guides.find(g=>g.id===$("guide").value);
  $("guide-question").textContent=guide.question;
  $("variation").replaceChildren(...guide.variations.map((v,i)=>option(String(i),v.label)),option("all","Compare all variations, one after another"));
  $("variation").value="1";
  buildControls($("scratch-controls"),"scratch",registry.parameters,{...registry.defaults,...guide.overrides},()=>{architectureScene?.updateSettings(settings("scratch"));preview();},guide.parameter);
  preview();
}
function guidedPayload(){const value=$("variation").value;return {guide:$("guide").value,variation:value==="all"?"all":Number(value),overrides:settings("scratch"),prediction:""};}
async function preview(){const serial=++previewSerial;plan=null;try{const result=await api("/api/preview",guidedPayload());if(serial!==previewSerial)return;plan=result;$("plan").replaceChildren(...result.runs.map(r=>{const item=make("div",undefined,"plan-item");item.append(make("strong",r.label),make("span",` · ${r.settings.steps} updates · ${r.estimate.effective_windows_per_update} text windows per update`));return item;}));if(engine==="scratch")architectureScene?.updateSettings(result.runs[0].settings);buttons();}catch(e){error("plan",e.message);}}
function mountTrainingStructure(){architectureScene?.destroy();architectureScene=engine==="scratch"?LabVisuals.mount($("architecture-preview"),"architecture",{settings:plan?.runs[0].settings||settings("scratch")}):LabVisuals.mount($("language-structure"),"lora",{settings:{model_config:adaptation?.model_config,...settings("adapt")}});}
function setEngine(value){engine=value;$("language-setup").hidden=value!=="lora";$("scratch-setup").hidden=value!=="scratch";document.querySelectorAll("[data-engine]").forEach(n=>n.classList.toggle("active",n.dataset.engine===value));$("run-training").textContent=value==="lora"?"Train the adapter":"Run this experiment";mountTrainingStructure();buttons();}
async function launch(){
  try{
    $("run-training").disabled=true;error("train-message","Starting local training…");
    const result=await api(engine==="lora"?"/api/adapt":"/api/guided",engine==="lora"?{settings:settings("adapt")}:guidedPayload());
    job=result;if(result.study){monitorRun=result.study.runs[0].id;}else monitorRun=result.id;
    selectedRun=monitorRun;error("train-message","Training started. Closing this browser tab will not stop the process.");await refresh();
  }catch(e){error("train-message",e.message);}finally{buttons();}
}
function buttons(){
  $("run-training").disabled=!!job.running||(engine==="lora"?!adaptation?.assets.ready:!plan);
  $("cancel").disabled=!job.running;
  const complete=run?.manifest.status==="MEASURED";
  $("generate").disabled=job.running||!complete;$("inspect").disabled=job.running||!complete;$("inspect-update").disabled=job.running||!complete||run?.manifest.engine==="lora"||run?.manifest.model.parameters>5000000;
  $("export").disabled=job.running||!complete||run?.manifest.engine==="lora";
  $("resume").disabled=job.running||!run?.can_resume;
  if(job.running){$("generation-message").textContent="Generation is available when training finishes.";$("trace-message").textContent="Model inspection is available when training finishes.";}
  else{if($("generation-message").textContent==="Generation is available when training finishes.")$("generation-message").textContent="";if($("trace-message").textContent==="Model inspection is available when training finishes.")$("trace-message").textContent="";}
}
function choices(id,chosen){const box=$(id),signature=runList.map(r=>r.id+r.status).join("|");if(box.dataset.signature!==signature){box.replaceChildren(...runList.map(r=>option(r.id,`${r.engine==="lora"?"Language adapter":"Scratch decoder"} · ${r.id} · ${statusLabel(r.status)}`)));if(!runList.length)box.append(option("","No saved runs"));box.dataset.signature=signature;}if(chosen)box.value=chosen;}
function stats(target,pairs){target.replaceChildren(...pairs.map(([label,value])=>{const box=make("div",undefined,"stat");box.append(make("small",label),make("strong",value));return box;}));}
function table(headings,rows){const wrap=make("div",undefined,"table-scroll"),table=make("table"),head=make("thead"),tr=make("tr"),body=make("tbody");headings.forEach(h=>tr.append(make("th",h)));head.append(tr);rows.forEach(r=>{const tr=make("tr");r.forEach(v=>tr.append(make("td",String(v))));body.append(tr);});table.append(head,body);wrap.append(table);return wrap;}
function lossSeries(data){const rows=data?.metrics||[];return [{label:"Training batch",color:palette[1],values:rows.filter(r=>Number.isFinite(r.sampled_train_nll)).map(r=>({x:r.step,y:r.sampled_train_nll}))},{label:"Development",color:palette[0],values:rows.filter(r=>Number.isFinite(r.dev_nll)).map(r=>({x:r.step,y:r.dev_nll}))}];}
function chart(canvas,series,axis="weight updates"){
  // CSS fixes the display box; the backing bitmap must never size the layout.
  const {width,height}=canvas.getBoundingClientRect();if(!width||!height)return;
  const ratio=Math.min(3,Math.max(1,window.devicePixelRatio||1));
  canvas.width=Math.round(width*ratio);canvas.height=Math.round(height*ratio);
  const ctx=canvas.getContext("2d");ctx.setTransform(canvas.width/width,0,0,canvas.height/height,0,0);ctx.clearRect(0,0,width,height);
  const valid=series.flatMap(s=>s.values).filter(p=>Number.isFinite(p.x)&&Number.isFinite(p.y));ctx.font="11px system-ui";
  if(!valid.length){ctx.fillStyle="#626b76";ctx.fillText("No measurements recorded yet.",20,70);return;}
  const low=Math.max(0,Math.min(...valid.map(p=>p.y))*.94),high=Math.max(...valid.map(p=>p.y))*1.06+1e-12,end=Math.max(1,...valid.map(p=>p.x)),left=52,right=20,top=20,bottom=36;
  const x=v=>left+v/end*(width-left-right),y=v=>height-bottom-(v-low)/(high-low)*(height-top-bottom);
  for(let i=0;i<=4;i++){const v=low+(high-low)*i/4;ctx.strokeStyle="#e2e5e8";ctx.beginPath();ctx.moveTo(left,y(v));ctx.lineTo(width-right,y(v));ctx.stroke();ctx.fillStyle="#697584";ctx.fillText(high<.01?v.toExponential(1):v.toFixed(2),2,y(v)+3);}
  series.forEach(s=>{const points=s.values.filter(p=>Number.isFinite(p.x)&&Number.isFinite(p.y));ctx.strokeStyle=s.color;ctx.fillStyle=s.color;ctx.lineWidth=2;ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(x(p.x),y(p.y)):ctx.moveTo(x(p.x),y(p.y)));ctx.stroke();points.forEach(p=>ctx.fillRect(x(p.x)-2,y(p.y)-2,4,4));});
  ctx.fillStyle="#697584";ctx.fillText("0",left,height-11);ctx.fillText(`${end.toLocaleString()} ${axis}`,Math.max(left,width-220),height-11);
}
function renderLive(){
  const data=monitor,m=data?.manifest;
  $("live-phase").textContent=m?`${statusLabel(m.status)} · ${m.phase||"Recorded run"}`:"No training is active.";
  if(job.running&&["model","dataset"].includes(job.kind))$("live-phase").textContent=`Another local worker is active (${job.kind}); its progress is under the ML/DL training lab. LLM training is unavailable until it finishes.`;
  $("live-progress").value=m?m.completed_steps/Math.max(1,m.requested_steps):0;
  stats($("live-stats"),[["WEIGHT UPDATES",m?`${m.completed_steps} / ${m.requested_steps}`:"—"],["TIME ELAPSED",elapsed(m?.elapsed_seconds??m?.training_seconds)],["ESTIMATED REMAINING",m?.status==="RUNNING"?elapsed(m.eta_seconds):"—"],["TARGET TOKENS / SECOND",m?.tokens_per_second?Math.round(m.tokens_per_second).toLocaleString():"—"]]);
  chart($("live-loss"),lossSeries(data));chart($("live-rate"),[{color:palette[0],values:(data?.metrics||[]).filter(r=>Number.isFinite(r.learning_rate)).map(r=>({x:r.step,y:r.learning_rate}))}]);
  $("tensorboard-command").textContent=`python -m tensorboard.main --logdir ${m?"runs/"+m.id+"/tensorboard":"runs"} --host 127.0.0.1 --port 6006`;
}
function renderRun(){
  $("empty-run").hidden=!!run;$("run-content").hidden=!run;if(!run)return;
  const m=run.manifest,isLanguage=m.engine==="lora";
  const reading=LabReadouts.languageResult(run);$("run-reading").replaceChildren(make("h2",reading.title),make("p",reading.text));
  $("run-summary").textContent=`${statusLabel(m.status)} · ${isLanguage?"SmolLM2-360M with LoRA":"From-scratch decoder"} · ${m.model.parameters.toLocaleString()} total parameters${m.model.trainable_parameters?" · "+m.model.trainable_parameters.toLocaleString()+" trainable adapter parameters":""} · ${m.hardware.device.toUpperCase()}`;
  $("config").textContent=JSON.stringify({id:m.id,settings:m.settings,model:m.model,device:m.hardware,integrity:run.integrity,error:m.error},null,2);
  $("curve-values").textContent=JSON.stringify(run.metrics||[],null,2);
  $("resume-area").hidden=isLanguage;
  const docs=run.data?.documents||[],old=$("documents").value;
  $("documents").replaceChildren(...docs.map(d=>option(d.id,`${d.split} · ${d.id}`)));if(docs.some(d=>d.id===old))$("documents").value=old;
  $("documents").onchange=()=>$("document").textContent=docs.find(d=>d.id===$("documents").value)?.text||"No prepared documents yet.";$("documents").onchange();
  $("split-counts").replaceChildren(...Object.entries(run.data?.counts||{}).map(([s,n])=>make("span",`${s==="dev"?"Development":s==="train"?"Training":"Test"}: ${n} examples`)));
  const result=run.evaluation;$("scores").replaceChildren();
  $("score-definition").textContent=isLanguage?"The starting point is the pretrained model. Only reference assistant-response positions are scored. Development loss selects the adapter checkpoint, including the unchanged base model; test text is not used for selection.":"The bigram baseline reads only the previous token; the Transformer can use a longer window. Development loss selects the checkpoint; test text is not used for selection.";
  if(result){
    const baseline=result.pretrained||result.bigram;
    $("scores").append(table(["Text split",isLanguage?"Unchanged model loss":"Bigram reference loss","Selected checkpoint loss","Selected checkpoint perplexity"],["dev","test"].map(s=>[s==="dev"?"Development (selection)":"Test (held-out check)",fixed(baseline?.[s]?.nll),fixed(result.transformer[s].nll),fixed(result.transformer[s].perplexity)])));
    $("evaluation-note").textContent=`Selected update: ${result.checkpoint_step}. ${isLanguage?"The conversation data is synthetic and may have been seen in the model's original training; these are local adaptation holdouts, not a new benchmark.":"Development selects the checkpoint. Repeatedly using test scores to choose settings makes test part of development."}`;
  }else{$("scores").append(make("p","Evaluation appears when this run completes."));$("evaluation-note").textContent="";}
  $("samples").replaceChildren();
  const samples=run.samples||[],prompts=[...new Set(samples.map(s=>s.prompt))];
  for(const prompt of prompts){$("samples").append(make("p",prompt,"sample-prompt"));const region=make("div",undefined,isLanguage?"sample-grid":"");for(const s of samples.filter(s=>s.prompt===prompt)){const box=make("div",undefined,"sample");box.append(make("small",s.stage||"Saved output"),make("p",s.completion||"[End token immediately]"));if(s.completion?.includes("sublimation")&&prompt.includes("ice melt"))box.append(make("p","Factual error: melting is a change from solid to liquid. Sublimation is solid to gas.","sample-error"));region.append(box);}$("samples").append(region);}
  $("limits").replaceChildren(...(m.limits||[]).map(v=>make("li",v)));buttons();drawRun();
}
function drawRun(){if(page!=="explore"||!run||tab!=="curves")return;const axis=$("curve-axis").value,rows=run.metrics||[];chart($("curve"),[{label:"Training",color:palette[1],values:rows.map(r=>({x:r[axis],y:r.sampled_train_nll??r.train_nll}))},{label:"Development",color:palette[0],values:rows.map(r=>({x:r[axis],y:r.dev_nll}))}],axis==="step"?"weight updates":"target tokens");}
async function inspect(update=false){try{$("inspect").disabled=true;$("inspect-update").disabled=true;error("trace-message",update?"Computing one diagnostic SGD step on a disposable model copy…":"Reading this checkpoint's forward calculation…");const data=await api(update?"/api/trace-update":"/api/trace",{run:selectedRun,text:$("trace-text").value});lastTrace?.destroy();lastTrace=LabVisuals.mount($("actual-trace"),"trace",{trace:data});error("trace-message",data.note+(data.training_update?" The weight-change table below shows the measured before/gradient/after values.":""));}catch(e){error("trace-message",e.message);}finally{buttons();}}
function buildGenerationControls(){
  const rows={temperature:{label:"Temperature",choices:[0.,.3,.8,1.2],default:0.,help:"Zero chooses the highest-scoring token. Higher values spread probability more broadly and can add variety and mistakes."},top_k:{label:"Top-k",choices:[1,10,20,50],default:20,help:"Only the k highest-scoring tokens remain eligible when sampling. Top-k 1 always chooses the highest-scoring token."},tokens:{label:"Maximum new tokens",choices:[16,40,80,128],default:80,help:"A ceiling on generated pieces, not words. Generation can stop earlier at an end-of-response token."},seed:{label:"Sampling seed",choices:[7,17,29,43],default:17,help:"A repeatable starting point for sampling randomness. It does not change the learned weights."}};
  $("generation-controls").replaceChildren(...Object.entries(rows).map(([key,p])=>{const box=make("div",undefined,"setting"),label=make("label",p.label),select=make("select");select.id="generation-"+key;select.append(...p.choices.map(v=>option(String(v),String(v))));select.value=String(p.default);const description=make("p",p.help,"field-help");description.id="generation-"+key+"-help";label.setAttribute("for",select.id);select.setAttribute("aria-describedby",description.id);box.append(label,description,select);return box;}));
}
async function generate(){try{$("generate").disabled=true;error("generation-message","Generating with the saved weights…");const result=await api("/api/generate",{run:selectedRun,prompt:$("generation-prompt").value,...Object.fromEntries(["temperature","top_k","tokens","seed"].map(k=>[k,Number($("generation-"+k).value)]))});const box=make("div",undefined,"sample");box.append(make("small",`${result.run} · temperature ${result.temperature} · top-k ${result.top_k}`),make("p",result.completion||"[End token immediately]"));$("generation-output").prepend(box);error("generation-message","Saved locally. These controls did not update the model weights.");}catch(e){error("generation-message",e.message);}finally{buttons();}}

function renderCompareChoices(){
  const sig=runList.map(r=>r.id+r.status).join("|");if($("compare-choices").dataset.signature===sig)return;$("compare-choices").dataset.signature=sig;$("compare-choices").replaceChildren();
  const ready=runList.filter(r=>r.status==="MEASURED");
  const allowed=new Set(ready.map(r=>r.id));for(const id of selectedComparisons)if(!allowed.has(id)){selectedComparisons.delete(id);comparison=null;comparisonSerial++;$("comparison-results").hidden=true;}
  const update=()=>{$("compare-run").disabled=selectedComparisons.size<2||selectedComparisons.size>4;$("comparison-selection-help").textContent=ready.length<2?"At least two completed runs are needed. Train a reference run and a changed setting first; no scores are invented here.":`${selectedComparisons.size} selected. Choose two to four; the first table row will be the reference.`;};
  ready.forEach(r=>{const label=make("label",undefined,"compare-choice"),input=make("input");input.type="checkbox";input.checked=selectedComparisons.has(r.id);input.onchange=()=>{input.checked?selectedComparisons.add(r.id):selectedComparisons.delete(r.id);comparison=null;comparisonSerial++;$("comparison-results").hidden=true;$("comparison-warnings").replaceChildren();update();};label.append(input,make("span",`${r.id} · ${r.engine==="lora"?"Language adapter":"Scratch decoder"}`));$("compare-choices").append(label);});if(!ready.length)$("compare-choices").append(make("p","No completed runs yet. Completed training results, not lesson examples, appear in this list."));update();
}
async function compare(){const token=++comparisonSerial;try{const result=await api("/api/compare",{runs:[...selectedComparisons]});if(token!==comparisonSerial)return;comparison=result;$("comparison-warnings").replaceChildren(...comparison.warnings.map(w=>make("p",w)));$("comparison-results").hidden=false;for(const o of $("comparison-metric").options)o.disabled=o.value.includes("bits_per_byte")?!comparison.bits_per_byte_comparable:!comparison.nll_comparable;$("comparison-metric").value=comparison.nll_comparable?"dev_nll":comparison.bits_per_byte_comparable?"dev_bits_per_byte":"";$("comparison-changes").replaceChildren(table(["Setting",...comparison.runs.map((r,i)=>"Run "+(i+1))],comparison.changes.map(c=>[c.parameter,...c.values.map(v=>typeof v==="object"?JSON.stringify(v):String(v))])));drawComparison();}catch(e){if(token===comparisonSerial)error("comparison-warnings",e.message);}}
function drawComparison(){
  if(!comparison||page!=="compare")return;
  const metric=$("comparison-metric").value,split=metric.startsWith("test")?"test":"dev",key=metric.includes("bits_per_byte")?"bits_per_byte":"nll",axis=$("comparison-axis").value;
  const reading=LabReadouts.languageComparison(comparison,metric);$("comparison-reading").replaceChildren(make("h2",reading.title),make("p",reading.text));
  const reference=metric?comparison.runs[0].evaluation.transformer[split][key]:null;
  $("comparison-table").replaceChildren(table(["Run",metric?LabReadouts.metric(metric).name:"Score","Change from first row","Weight updates","Target-token exposures"],comparison.runs.map((r,i)=>[`${i+1} · ${r.id}`,metric?fixed(r.evaluation.transformer[split][key]):"Not comparable",metric?LabReadouts.difference(r.evaluation.transformer[split][key],reference):"Not comparable",r.manifest.completed_steps,r.manifest.tokens_processed])));
  $("comparison-chart-region").hidden=!comparison.nll_comparable;
  $("comparison-legend").replaceChildren(...comparison.runs.map((r,i)=>make("span",`Run ${i+1}`,"series-"+i)));
  if(comparison.nll_comparable)chart($("comparison-curve"),comparison.runs.map((r,i)=>({color:palette[i],values:r.metrics.filter(p=>Number.isFinite(p.dev_nll)).map(p=>({x:p[axis],y:p.dev_nll}))})),axis==="step"?"weight updates":"target tokens");
}

async function refresh(){
  if(refreshing)return;refreshing=true;
  try{
    const [list,active]=await Promise.all([api("/api/runs"),api("/api/job")]);runList=list;job=active;
    if(job.running){if(job.kind==="study"){const studies=await api("/api/studies"),s=studies.find(s=>s.id===job.id);const r=s?.runs.find(r=>r.status==="RUNNING")||s?.runs.filter(r=>r.status==="MEASURED").at(-1);if(r)monitorRun=r.id;}else if(job.kind==="run")monitorRun=job.id;}
    if(!selectedRun||!list.some(r=>r.id===selectedRun))selectedRun=list.find(r=>r.engine==="lora"&&r.status==="MEASURED")?.id||list.find(r=>r.status==="MEASURED")?.id||list[0]?.id||null;
    if(!monitorRun||!list.some(r=>r.id===monitorRun))monitorRun=list.find(r=>r.status==="RUNNING")?.id||selectedRun;
    choices("runs",selectedRun);choices("monitor-run",monitorRun);
    const previous=run?.manifest.id;run=selectedRun?await api("/api/run/"+selectedRun):null;
    monitor=monitorRun===selectedRun?run:monitorRun?await api("/api/run/"+monitorRun):null;
    if(previous!==run?.manifest.id){lastTrace?.destroy();lastTrace=null;$("actual-trace").replaceChildren();$("generation-output").replaceChildren();$("export-result").textContent="";$("trace-message").textContent="";}
    if(page==="explore")renderRun();if(page==="train")renderLive();if(page==="compare")renderCompareChoices();buttons();
  }catch(e){if(page==="train")error("train-message",e.message);else if(page==="explore")error("explore-error",e.message);}
  finally{refreshing=false;}
}
async function bootstrap(){
  try{
    [registry,adaptation,localHardware]=await Promise.all([api("/api/catalog"),api("/api/adaptation"),api("/api/hardware")]);
    for(const id of ["training-hardware","model-hardware"]){const host=$(id);if(host){host.textContent=localHardware.summary+" "+localHardware.note;}}
    $("asset-state").textContent=adaptation.assets.ready?"Prepared locally · 361.8M model parameters · Apache-2.0 model and conversation dataset. No API key needed.":"Model files are not prepared. In the project folder run: python lab.py prepare --model";
    const params=Object.fromEntries(Object.entries(adaptation.choices).map(([k,v])=>[k,{choices:v}]));buildControls($("adapt-controls"),"adapt",params,adaptation.defaults,()=>{if(engine==="lora")architectureScene?.updateSettings({model_config:adaptation.model_config,...settings("adapt")});});
    $("guide").replaceChildren(...registry.guides.map(g=>option(g.id,trainingGuideNames[g.id]||g.title)));loadGuide();buildGenerationControls();
    $("visual-select").replaceChildren(...LabVisuals.catalog.map(v=>option(v.id,v.title)));
    await Promise.all([LabLearning.init(show),ModelLab.init(),GuidedCourse.init(show)]);show("home");await refresh();
  }catch(e){$("home").append(make("p",e.message,"error"));}
}
document.querySelectorAll("[data-page]").forEach(n=>n.onclick=()=>{
  const name=n.dataset.page;
  if(page===name&&[...coursePages,...interviewPages].includes(name))LabLearning.toggleMenu(name);else show(name);
});
document.querySelectorAll("[data-subject]").forEach(n=>n.onclick=()=>{const menu=$("subject-"+n.dataset.subject);menu.hidden=!menu.hidden;n.setAttribute("aria-expanded",String(!menu.hidden));});
document.querySelectorAll("[data-engine]").forEach(n=>n.onclick=()=>setEngine(n.dataset.engine));
document.querySelectorAll("[data-tab]").forEach(n=>n.onclick=()=>showTab(n.dataset.tab));
$("visual-select").onchange=mountExample;$("guide").onchange=loadGuide;$("variation").onchange=preview;
$("read-training-loop").onclick=e=>{e.preventDefault();LabLearning.openDocument("T0");};
for(const id of ["metrics-language-results","metrics-language-compare"])$(id).onclick=()=>LabLearning.openMetric("language");
$("run-training").onclick=launch;$("cancel").onclick=async()=>{try{await api("/api/cancel",{});error("train-message","Stopping training; waiting for the worker to save its final state.");await refresh();}catch(e){error("train-message",e.message);}};
$("runs").onchange=async()=>{selectedRun=$("runs").value;await refresh();};$("monitor-run").onchange=async()=>{monitorRun=$("monitor-run").value;await refresh();};
$("curve-axis").onchange=drawRun;$("inspect").onclick=()=>inspect(false);$("inspect-update").onclick=()=>inspect(true);$("generate").onclick=generate;
$("tokenize").onclick=async()=>{try{const result=await api("/api/tokenize",{run:selectedRun,text:$("text").value});$("tokens-output").replaceChildren(...result.ids.map((v,i)=>{const c=make("span",result.pieces[i],"token-chip");c.append(make("small",String(v)));return c;}));$("roundtrip").textContent=`${result.ids.length} pieces. Decoded text: ${result.round_trip}`;}catch(e){$("roundtrip").textContent=e.message;}};
$("resume").onclick=async()=>{try{const result=await api("/api/resume",{run:selectedRun,additional_steps:Number($("resume-steps").value)});monitorRun=result.id;show("train");await refresh();}catch(e){error("explore-error",e.message);}};
$("export").onclick=async()=>{try{const r=await api("/api/export",{run:selectedRun});$("export-result").textContent=`${r.status}: ${r.directory}`;}catch(e){$("export-result").textContent=e.message;}};
$("compare-run").onclick=compare;$("comparison-metric").onchange=drawComparison;$("comparison-axis").onchange=drawComparison;
window.addEventListener("resize",()=>{if(page==="train")renderLive();drawRun();drawComparison();});
bootstrap();setInterval(()=>{if(!document.hidden&&(page==="train"||page==="explore"&&job.running))refresh();},1500);
