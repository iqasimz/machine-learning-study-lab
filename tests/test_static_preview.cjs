// Verify the Pages adapter beneath a repository URL, including read-only limits.
const assert=require("node:assert/strict"),fs=require("node:fs"),path=require("node:path"),vm=require("node:vm");
const source=fs.readFileSync(path.resolve(__dirname,"../scripts/preview.js"),"utf8");
const cp=require("node:child_process");
const fixture=cp.spawnSync(process.env.LAB_PYTHON||"python3",["-c",`import json,sys
sys.path.insert(0,'src')
from training_lab.experiments import catalog,make_plan
r=catalog()
print(json.dumps({'registry':r,'plans':{g['id']:make_plan(g['id'])['runs'] for g in r['guides']}}))`],{cwd:path.resolve(__dirname,".."),encoding:"utf8",maxBuffer:4*1024*1024});
assert.equal(fixture.status,0,fixture.stderr);
const fixtures=JSON.parse(fixture.stdout);
const calls=[],snapshots={"/api/hardware":{selected:null},"/api/learning/document/T3":{id:"T3"},"/api/catalog":fixtures.registry};
const context={URL,Response,location:{href:"https://example.github.io/study-lab/",origin:"https://example.github.io"},
  document:{currentScript:{src:"https://example.github.io/study-lab/preview.js"},addEventListener(){}},
  window:{fetch:async url=>{calls.push(String(url));return new Response(JSON.stringify(snapshots),{status:200});}}};
vm.createContext(context);vm.runInContext(source,context);
(async()=>{
  const response=await context.window.fetch("/api/learning/document/T3");assert.equal((await response.json()).id,"T3");
  assert.equal(calls[0],"https://example.github.io/study-lab/api-data.json");
  assert.equal((await context.window.fetch("/api/train",{method:"POST",body:"{}"})).status,405);
  assert.equal((await context.window.fetch("/api/unknown")).status,404);
  assert.equal((await context.window.fetch("/api/hardware")).status,200);
  assert.equal(context.window.LAB_STATIC_PREVIEW,true);
  for(const [guide,expected] of Object.entries(fixtures.plans)){
    const result=await context.window.fetch("/api/preview",{method:"POST",body:JSON.stringify({guide,variation:"all",overrides:{}})});
    assert.equal(result.status,200,guide);const actual=(await result.json()).runs;
    assert.equal(actual.length,expected.length);
    actual.forEach((run,i)=>{
      assert.deepEqual(run.settings,expected[i].settings);
      for(const key of ["effective_windows_per_update","maximum_token_exposures","attention_scores_per_layer_microbatch"])assert.equal(run.estimate[key],expected[i].estimate[key],guide+" "+key);
    });
  }
  assert.equal((await context.window.fetch("/api/preview",{method:"POST",body:JSON.stringify({guide:fixtures.registry.guides[0].id,variation:-1})})).status,400);
  for(const endpoint of ["/api/models/train","/api/models/prepare","/api/adapt","/api/guided","/api/cancel"]){
    assert.equal((await context.window.fetch(endpoint,{method:"POST",body:"{}"})).status,405);
  }
  assert.equal(calls.length,1,"Snapshots are cached; no backend request is made");
  console.log("PASS: subpath loading, cache, all guided plans match Python, invalid plans rejected and every training/download write blocked");
})().catch(e=>{console.error(e);process.exitCode=1;});
