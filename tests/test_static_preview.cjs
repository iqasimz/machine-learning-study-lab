// Verify the Pages adapter beneath a repository URL, including read-only limits.
const assert=require("node:assert/strict"),fs=require("node:fs"),path=require("node:path"),vm=require("node:vm");
const source=fs.readFileSync(path.resolve(__dirname,"../scripts/preview.js"),"utf8");
const calls=[],snapshots={"/api/hardware":{selected:null},"/api/learning/document/T3":{id:"T3"}};
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
  assert.equal(calls.length,1,"Snapshots are cached; no backend request is made");
  console.log("PASS: repository-subpath snapshot loading, cache, missing endpoint and training-write rejection");
})().catch(e=>{console.error(e);process.exitCode=1;});
