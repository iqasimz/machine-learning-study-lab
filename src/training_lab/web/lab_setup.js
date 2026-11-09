"use strict";

// Visible choices share the existing select-backed form state and validators.
const LabSetup=(()=>{
  const el=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;};
  function choices(select,host,title,help=""){
    select.hidden=true;select.setAttribute("aria-hidden","true");select.tabIndex=-1;
    const field=el("fieldset",undefined,"lab-choice-field"),list=el("div",undefined,"lab-choice-list");field.append(el("legend",title));
    if(help)field.append(el("p",help,"field-help"));
    const options=[...select.options],boolean=options.length===2&&options.some(o=>o.value==="true")&&options.some(o=>o.value==="false");
    select._visibleChoices=[];
    for(const [i,option] of options.entries()){
      if(boolean&&option.value==="false")continue;
      const label=el("label",undefined,"lab-choice"),input=el("input"),text=el("span",boolean?"Enabled":option.textContent);
      input.type=boolean?"checkbox":"radio";input.name="choice-"+select.id;input.value=option.value;input.id=select.id+"-choice-"+i;
      input.onchange=()=>{
        if(!boolean&&!input.checked)return;
        select.value=boolean?String(input.checked):option.value;
        sync(select);select.onchange?.();
      };
      label.append(input,text);list.append(label);select._visibleChoices.push({input,label,option,boolean});
    }
    field.append(list);host.replaceChildren(field);sync(select);return field;
  }
  function sync(select){
    for(const {input,label,option,boolean} of select?._visibleChoices||[]){
      input.checked=boolean?select.value==="true":select.value===option.value;
      input.disabled=select.disabled||option.disabled;label.classList.toggle("selected",input.checked);label.classList.toggle("unavailable",input.disabled);
      label.setAttribute("aria-disabled",String(input.disabled));label.title=select.title||option.title||(input.disabled?"Not available for this setup.":"");
    }
  }
  function python(value,depth=0){
    if(value===null)return "None";
    if(typeof value==="boolean")return value?"True":"False";
    if(typeof value!=="object")return JSON.stringify(value);
    if(Array.isArray(value))return "["+value.map(v=>python(v,depth+1)).join(", ")+"]";
    const indent="    ".repeat(depth+1),end="    ".repeat(depth);
    return "{\n"+Object.entries(value).map(([k,v])=>indent+JSON.stringify(k)+": "+python(v,depth+1)).join(",\n")+"\n"+end+"}";
  }
  function review(host,{paragraphs,entry,sources}){
    const chosen=Number(host.dataset.sourceTab||sources.length-1);host.replaceChildren();
    const summary=el("section",undefined,"experiment-review");summary.append(el("h2","What this experiment will do"));
    for(const [heading,text] of paragraphs){const item=el("div");item.append(el("h3",heading),el("p",text));summary.append(item);}host.append(summary);
    const panel=el("section",undefined,"training-code");panel.append(el("h2","Code for this setup"),el("p","This entry point uses the selected values. The tabs show the actual functions it calls, including data preparation and evaluation; they use imports/helpers from the package.","plot-help"),el("p","To run an entry point yourself, use the project environment from this folder with PYTHONPATH=src. Executing it starts real local training and saves a run. Reading the code here does not execute it.","plot-help"));
    const call=el("pre",undefined,"code-entry");call.append(el("code",entry));panel.append(call);
    const tabs=el("div",undefined,"tab-row source-tabs"),source=el("div",undefined,"source-window");tabs.setAttribute("role","group");tabs.setAttribute("aria-label","Training source sections");
    const buttons=[];
    function show(index){
      host.dataset.sourceTab=String(index);source.replaceChildren();
      for(const [i,b] of buttons.entries()){b.classList.toggle("active",i===index);b.setAttribute("aria-pressed",String(i===index));}
      for(const part of sources[index].parts){
        source.append(el("p",part.file+":"+part.first_line+" · "+part.symbol,"source-location"));
        const pre=el("pre"),code=el("code",part.code);pre.append(code);source.append(pre);
      }
    }
    sources.forEach((item,i)=>{const b=el("button",item.label);b.type="button";b.onclick=()=>show(i);tabs.append(b);buttons.push(b);});
    panel.append(tabs,source);host.append(panel);if(sources.length)show(chosen<sources.length?chosen:sources.length-1);
  }
  return {choices,sync,python,review};
})();
