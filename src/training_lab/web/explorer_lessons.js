"use strict";

// Authored worked examples. Never used as training configurations or measured results.
const ConceptLessons=(()=>{
  const definitions=new Map(),placement={};
  const f=ArchitectureMath.fmt,v=ArchitectureMath.vector;
  const sources={
    math:["Dive into Deep Learning: preliminaries","https://d2l.ai/chapter_preliminaries/index.html"],
    ml:["scikit-learn user guide","https://scikit-learn.org/stable/user_guide.html"],
    splits:["Data leakage and preprocessing","https://scikit-learn.org/stable/common_pitfalls.html"],
    autograd:["PyTorch autograd","https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html"],
    neural:["Dive into Deep Learning","https://d2l.ai/"],
    transformer:["Attention Is All You Need","https://arxiv.org/abs/1706.03762"],
    lora:["LoRA paper","https://arxiv.org/abs/2106.09685"],
    dpo:["Direct Preference Optimization","https://arxiv.org/abs/2305.18290"],
    rag:["Retrieval-Augmented Generation","https://arxiv.org/abs/2005.11401"],
    python:["Python virtual environments","https://docs.python.org/3/library/venv.html"],
    distributed:["PyTorch distributed overview","https://docs.pytorch.org/tutorials/beginner/dist_overview.html"],
    cache:["Transformers cache explanation","https://huggingface.co/docs/transformers/cache_explanation"],
    safety:["OWASP LLM application risks","https://genai.owasp.org/llm-top-10/"],
    mcp:["Model Context Protocol architecture","https://modelcontextprotocol.io/docs/learn/architecture"]
  };
  function step(title,why,terms,input,action,equation,calculation,output,change,values=[]){
    return {id:"",title,why,terms,input,action,equation,calculation,output,change,values};
  }
  function add(id,definition){
    if(definitions.has(id))throw Error("Duplicate explorer "+id);
    definitions.set(id,definition);placement[id]={[definition.section||1]:id};
  }
  function build(id,options={}){
    const definition=definitions.get(id);if(!definition)throw Error("Unknown worked explorer.");
    const cfg={...definition.defaults,...options};
    for(const control of definition.controls||[]){
      if(!control.choices.some(([value])=>value===cfg[control.key]))throw Error("Unsupported setting "+control.key);
    }
    const result=definition.build(cfg),steps=Array.isArray(result)?result:result.steps;
    steps.forEach((s,i)=>{s.id="s"+i;s.title=(i+1)+" · "+s.title;});
    const diagram=graph(steps,result.layout||definition.layout||"chain",id);
    if(id==="ML08"){
      diagram.nodes[1].sub="x ≤ "+cfg.cut+" ?";
      diagram.nodes[2].sub="Left x: "+[1,2,3,4].filter(x=>x<=cfg.cut).join(", ");
      diagram.nodes[3].sub="Right x: "+[1,2,3,4].filter(x=>x>cfg.cut).join(", ");
    }
    steps.forEach(s=>{
      const successors=diagram.edges.filter(e=>e.from===s.id&&e.kind!=="training").map(e=>steps.find(x=>x.id===e.to)?.title.replace(/^\d+ · /,"")).filter(Boolean);
      s.next=s.next||(successors.length?"The outgoing connection carries this result to: "+[...new Set(successors)].join("; ")+".":"This is the endpoint of the worked comparison. In the lecture, connect it back to the assumptions and evaluation protocol.");
    });
    return {cfg,steps,diagram,title:definition.title,subtitle:definition.subtitle+" These are authored teaching examples, not saved-run measurements.",
      controls:result.controls||definition.controls||[],facts:result.facts||null,sources:(definition.sources||["neural"]).map(s=>Array.isArray(s)?s:sources[s]),
      legend:"Select a component or connection. Blue marks the selected operation and its inputs/outputs; amber marks a bypass or preserved branch. Dashed returns indicate repeated optimization, not automatic execution.",
      figures:Array.isArray(result)?[]:result.figures||[]};
  }
  function graph(steps,layout,id){
    const nodes=steps.map((s,i)=>({id:s.id,ids:[s.id],label:s.title.replace(/^\d+ · /,""),sub:s.output.split(/\.(?=\s|$)/)[0].slice(0,41),x:100,y:24+i*83,w:230,h:55}));
    const edges=[],connect=(a,b,kind="forward",note="")=>edges.push({id:"e"+edges.length,from:"s"+a,to:"s"+b,kind,note});
    for(let i=1;i<steps.length;i++)connect(i-1,i);
    if(layout==="loop"){
      const [from,to]=({ML09:[3,1],ML11:[2,1],DL13:[2,0],T0:[4,0]})[id]||[steps.length-1,0];
      connect(from,to,"training","Continue from the updated state for the next iteration; do not reset learned values. Selecting this path only explains the repetition.");
    }
    if(layout==="skip"&&steps.length>=3)connect(0,2,"bypass","Carry the original input unchanged around the learned branch, then add matching entries.");
    if(layout==="fork"&&steps.length>=4){
      nodes[1].x=10;nodes[1].y=126;nodes[1].w=190;
      nodes[2].x=230;nodes[2].y=126;nodes[2].w=190;
      nodes[3].y=242;if(nodes[4])nodes[4].y=325;
      for(let i=5;i<nodes.length;i++)nodes[i].y=325+(i-4)*83;
      edges.length=0;connect(0,1);connect(0,2);connect(1,3);connect(2,3);for(let i=4;i<nodes.length;i++)connect(i-1,i);
    }
    if(layout==="tree"&&steps.length>=5){
      nodes[1].y=112;nodes[2].x=10;nodes[2].y=218;nodes[2].w=190;
      nodes[3].x=230;nodes[3].y=218;nodes[3].w=190;nodes[4].y=328;
      for(let i=5;i<nodes.length;i++)nodes[i].y=328+(i-4)*83;
      edges.length=0;connect(0,1);connect(1,2);connect(1,3);connect(2,4);connect(3,4);for(let i=5;i<nodes.length;i++)connect(i-1,i);
    }
    return {width:430,height:Math.max(...nodes.map(n=>n.y+n.h))+28,nodes,edges};
  }
  const choice=(key,label,choices,help)=>({key,label,choices,help});
  const table=(label,headers,rows)=>({label,headers,rows});
  function read(trace,profile){
    // Explicit authored edition: retain arithmetic, replace exposition and false dependencies.
    const edition={...profile,...profile.variants?.[trace.cfg?.[profile.variantKey||"family"]]};
    if(edition.steps.length!==trace.steps.length)throw Error("Reader edition does not match worked example.");
    trace.steps=trace.steps.map((step,i)=>({...step,...edition.steps[i],id:step.id,
      title:(i+1)+" · "+edition.steps[i].title,why:edition.steps[i].prose,action:edition.steps[i].prose,
      next:edition.steps[i].next}));
    trace.title=edition.title;trace.reading=edition;
    if(edition.view==="diagram"){
      const d=edition.diagram;
      let diagram=graph(trace.steps,d.layout==="inputs"?"chain":d.layout||"chain","");
      if(d.layout==="inputs"){
        Object.assign(diagram.nodes[0],{x:10,y:24,w:190});Object.assign(diagram.nodes[1],{x:230,y:24,w:190});
        Object.assign(diagram.nodes[2],{x:100,y:142});Object.assign(diagram.nodes[3],{x:100,y:252});
      }
      if(d.layout==="probability-branch"){
        Object.assign(diagram.nodes[2],{x:10,y:220,w:190});Object.assign(diagram.nodes[3],{x:230,y:220,w:190});
      }
      for(const [index,position]of Object.entries(d.positions||{}))Object.assign(diagram.nodes[Number(index)],position);
      diagram.nodes=diagram.nodes.filter((n,i)=>d.nodes.includes(i)).map(n=>{
        const step=trace.steps[Number(n.id.slice(1))];
        return {...n,label:step.node,sub:step.diagramSummary||n.sub};
      });
      if(trace.reading.title==="Vectors, matrices and gradients")diagram.nodes[0].sub="One row × two features; two weights";
      if(trace.reading.title==="Linear regression")diagram.nodes[0].sub="XᵀX = 14; Xᵀy = 11";
      if(trace.reading.title==="Logistic regression and classification")diagram.nodes[3].sub=trace.steps[3].calculation.replace("For y=1, ","");
      if(trace.reading.title==="Naive Bayes classification"){
        diagram.nodes[0].sub="Positive: 0.5; negative: 0.5";
        diagram.nodes[1].sub=trace.steps[1].calculation.replace(/(\d+\.\d{4})/g,value=>Number(value).toFixed(3));
        diagram.nodes[2].sub=trace.steps[2].calculation.replaceAll(" mass","");
        diagram.nodes[3].sub=trace.steps[3].calculation.replace("Posterior positive", "Positive probability");
      }
      if(trace.reading.title==="Gradient boosting")diagram.nodes[3].sub=trace.steps[3].calculation;
      if(trace.reading.title==="Decision trees and random forests"){
        diagram.nodes[1].sub="x ≤ "+trace.cfg.cut+" ?";
        diagram.nodes[2].sub="Left x: "+[1,2,3,4].filter(x=>x<=trace.cfg.cut).join(", ");
        diagram.nodes[3].sub="Right x: "+[1,2,3,4].filter(x=>x>trace.cfg.cut).join(", ");
      }
      diagram.edges=d.links.map(([a,b,note,kind="forward"],i)=>({id:"e"+i,from:"s"+a,to:"s"+b,note,kind}));
      diagram.height=Math.max(...diagram.nodes.map(n=>n.y+n.h))+28;trace.diagram=diagram;
    }
    return trace;
  }
  return {add,build,read,placement,has:id=>definitions.has(id),ids:()=>[...definitions.keys()],step,choice,table,f,v};
})();
