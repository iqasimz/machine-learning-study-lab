// Test the offline lab without installing a browser or external packages.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const html = fs.readFileSync(path.join(__dirname,"..","..","VISUAL_LAB.html"),"utf8");
const source = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const context = {console};
vm.createContext(context);
vm.runInContext(source,context);
const m = context.courseMath;
const close = (a,b) => assert.ok(Math.abs(a-b)<1e-6,`${a} != ${b}`);
close(m.softmax([1,2]).reduce((a,b)=>a+b,0),1);
close(m.softmax([1,2])[0],m.softmax([1001,1002])[0]);
close(m.entropy([.5,.5]),Math.log(2));
close(m.cosine([1,1],[1,0]),1/Math.sqrt(2));
assert.equal(m.cosine([0,0],[1,0]),null);
close(m.dpo(.4,-.2,1).loss,.4374879504858856);
close(m.dpo(0,0,1).loss,Math.log(2));
close(m.group([0,0,1,1]).advantages[0],-1);
assert.ok(m.group([1,1,1,1]).advantages.every(x=>x===0));
assert.equal(m.kv(32,1,8192,8,128),2**30);

// Minimal DOM stand-ins check referenced IDs, defaults and callback wiring.
const elements={};
for(const match of html.matchAll(/\bid="([^"]+)"/g)) {
  assert.ok(!elements[match[1]],"Duplicate ID: "+match[1]);
  elements[match[1]]={value:"",checked:false,textContent:"",innerHTML:"",listeners:{},addEventListener(event,callback){this.listeners[event]=callback;}};
}
for(const match of html.matchAll(/<input\b([^>]+)>/g)) {
  const id=match[1].match(/\bid="([^"]+)"/)[1],value=match[1].match(/\bvalue="([^"]*)"/);
  if(value) elements[id].value=value[1];
}
elements.target.value="0";
elements["weight-bits"].value="16";
context.initLab({getElementById(id){return elements[id]||null;}});
assert.match(elements["probability-result"].textContent,/Entropy:/);
assert.match(elements["attention-result"].textContent,/1.3395/);
assert.match(elements["memory-result"].textContent,/KV: 1.00 GiB/);
elements["mask-second"].checked=true;
elements["mask-second"].listeners.change();
assert.match(elements["attention-result"].textContent,/\[2.0000, 0.0000\]/);
elements.rewards.value="1,1,1,1";
elements.rewards.listeners.input();
assert.match(elements["grpo-result"].textContent,/No relative reward signal/);
elements.rewards.value="1,,2";
elements.rewards.listeners.input();
assert.match(elements["grpo-result"].textContent,/Enter 2–20/);
elements["query-x"].value="0";
elements["query-y"].value="0";
elements["query-x"].listeners.input();
assert.match(elements["retrieval-result"].textContent,/Zero vector/);
elements.context.value="16384";
elements.context.listeners.input();
assert.match(elements["memory-result"].textContent,/KV: 2.00 GiB/);
assert.ok(!/https?:\/\//.test(html),"Offline lab must not request external resources");
console.log("Visual-lab arithmetic, element IDs, defaults and changed-state tests passed.");
console.log("This is not a browser layout/accessibility certification.");
