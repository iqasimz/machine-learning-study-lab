"use strict";

// Draws the picture payload produced by dsa_scenes.py. It does not re-run the algorithm.
const AlgorithmScenes = (() => {
  const NS = "http://www.w3.org/2000/svg";
  let diagramSerial=0;
  const el = (tag, text, cls) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (cls) node.setAttribute("class", cls);
    return node;
  };
  const sv = (tag, attrs, text) => {
    const node = document.createElementNS(NS, tag);
    for (const [key, value] of Object.entries(attrs || {})) node.setAttribute(key, String(value));
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const draw = {
    bounds(picture) {
      const values = picture.values || [];
      const width = Math.max(240, 48 + Math.max(values.length, picture.hi || 0) * 64);
      const svg = sv("svg", {viewBox: `0 0 ${width} 128`, role: "img"});
      const role = index => picture.mid === index ? "inspected" : index < picture.lo ? "below" : index >= picture.hi ? "later" : "unresolved";
      values.forEach((value, index) => {
        const x = 24 + index * 64;
        const mark = role(index);
        svg.append(sv("rect", {x, y: 32, width: 52, height: 36, class: "dsa-box " + mark}));
        svg.append(sv("text", {x: x + 26, y: 54, "text-anchor": "middle"}, String(value)));
        svg.append(sv("text", {x: x + 26, y: 86, "text-anchor": "middle", class: "dsa-index"}, "index " + index));
      });
      svg.append(sv("text", {x: 24 + picture.lo * 64, y: 14, "text-anchor": "start"}, "lo = " + picture.lo));
      svg.append(sv("text", {x: 24 + picture.hi * 64, y: 112, "text-anchor": "start"}, "hi " + picture.hi));
      if (picture.mid !== undefined) svg.append(sv("text", {x: 24 + picture.mid * 64 + 26, y: 106, "text-anchor": "middle"}, "mid = " + picture.mid));
      if (!values.length) svg.append(sv("text", {x: 16, y: 56}, "no cells"));
      return svg;
    },
    heap(picture) {
      const entries = picture.entries || [];
      const svg = sv("svg", {viewBox: "0 0 380 220", role: "img"});
      if (!entries.length) {
        svg.append(sv("text", {x: 16, y: 40}, "empty heap"));
        return svg;
      }
      const pos = entries.map((_, index) => {
        const depth = Math.floor(Math.log2(index + 1));
        const count = 2 ** depth;
        const slot = index - (count - 1);
        return {x: ((slot + 0.5) / count) * 360 + 10, y: 28 + depth * 52};
      });
      entries.forEach((_, index) => {
        if (!index) return;
        const parent = Math.floor((index - 1) / 2);
        svg.append(sv("line", {x1: pos[parent].x, y1: pos[parent].y, x2: pos[index].x, y2: pos[index].y, class: "dsa-edge"}));
      });
      entries.forEach((entry, index) => {
        const focus = entry.id === picture.focus ? " focus" : "";
        svg.append(sv("circle", {cx: pos[index].x, cy: pos[index].y, r: 16, class: "dsa-box" + focus}));
        svg.append(sv("text", {x: pos[index].x, y: pos[index].y + 4, "text-anchor": "middle"}, entry.id + ":" + entry.score));
      });
      entries.forEach((entry, index) => {
        const x = 12 + index * 72;
        svg.append(sv("rect", {x, y: 176, width: 66, height: 28, class: "dsa-box"}));
        svg.append(sv("text", {x: x + 33, y: 194, "text-anchor": "middle"}, index + " " + entry.id + ":" + entry.score));
      });
      return svg;
    },
    graph(picture) {
      const svg = sv("svg", {viewBox: "0 0 280 220", role: "img"});
      const defs = sv("defs");
      const arrowId="dsa-graph-arrow-"+(++diagramSerial);
      const marker = sv("marker", {id:arrowId, viewBox:"0 0 10 10", refX:9, refY:5, markerWidth:6, markerHeight:6, orient:"auto-start-reverse"});
      marker.append(sv("path", {d:"M 0 0 L 10 5 L 0 10 z", fill:"#6d8296"}));defs.append(marker);svg.append(defs);
      for (const edge of picture.edges || []) {
        const source = picture.nodes.find(node => node.id === edge.src);
        const target = picture.nodes.find(node => node.id === edge.dst);
        const dx=target.x-source.x,dy=target.y-source.y,len=Math.hypot(dx,dy)||1;
        svg.append(sv("line", {x1: source.x+dx/len*10, y1: source.y+dy/len*10, x2: target.x-dx/len*12, y2: target.y-dy/len*12, class: "dsa-edge", "marker-end":"url(#"+arrowId+")"}));
        if(edge.weight!==undefined)svg.append(sv("text",{x:(source.x+target.x)/2+5,y:(source.y+target.y)/2-5,class:"dsa-weight"},edge.weight));
      }
      for (const node of picture.nodes) {
        const words = [];
        if (picture.active === node.id) words.push("current");
        else if ((picture.path || []).includes(node.id)) words.push("on path");
        else if ((picture.frontier || []).includes(node.id)) words.push("waiting");
        else if ((picture.visited || []).includes(node.id)) words.push(picture.visitedLabel||"discovered");
        if(picture.nodeNotes?.[node.id])words.push(picture.nodeNotes[node.id]);
        svg.append(sv("circle", {cx: node.x, cy: node.y, r: 8, class: "dsa-box" + (picture.active === node.id ? " focus" : "")}));
        svg.append(sv("text", {x: node.x+12, y: node.y - 13, "text-anchor": "start"}, node.id));
        svg.append(sv("text", {x: node.x, y: node.y + 20, "text-anchor": "middle", class: "dsa-index"}, words.join(" ") || "idle"));
      }
      return svg;
    },
    grid(picture) {
      const table = el("table", undefined, "dsa-grid");
      if(picture.meaning)table.append(el("caption",picture.meaning));
      const head = el("tr");
      head.append(el("th", (picture.rowAxis||"row")+" \\ "+(picture.colAxis||"column")));
      for (const label of picture.colLabels) head.append(el("th", label === "·" ? "empty" : label));
      table.append(head);
      picture.cells.forEach((row, rowIndex) => {
        const line = el("tr");
        const label = picture.rowLabels[rowIndex];
        line.append(el("th", label === "·" ? "empty" : label));
        row.forEach((value, column) => {
          const focus = picture.focus && picture.focus[0] === rowIndex && picture.focus[1] === column;
          const dependency = (picture.deps || []).some(pair => pair[0] === rowIndex && pair[1] === column);
          line.append(el("td", value === null || value === undefined ? "·" : String(value), focus ? "focus" : dependency ? "dep" : ""));
        });
        table.append(line);
      });
      return table;
    },
    tree(picture) {
      const svg = sv("svg", {viewBox: "0 0 320 190", role: "img"});
      for (const edge of picture.edges || []) {
        const source = picture.nodes.find(node => node.id === edge.src);
        const target = picture.nodes.find(node => node.id === edge.dst);
        svg.append(sv("line", {x1: source.x, y1: source.y, x2: target.x, y2: target.y, class: "dsa-edge"}));
      }
      for (const node of picture.nodes) {
        svg.append(sv("circle", {cx: node.x, cy: node.y, r: 16, class: "dsa-box" + (node.id === picture.focus ? " focus" : "")}));
        svg.append(sv("text", {x: node.x, y: node.y + 4, "text-anchor": "middle"}, node.label));
      }
      return svg;
    },
    intervals(picture) {
      const span = Math.max(1, picture.max - picture.min);
      const height = 44 + picture.rows.length * 42;
      const svg = sv("svg", {viewBox: `0 0 450 ${height}`, role: "img"});
      for(let i=0;i<=4;i++){
        const value=picture.min+span*i/4,x=90+i*58;
        svg.append(sv("text",{x,y:17,"text-anchor":"middle"},Number(value.toFixed(2))));
      }
      picture.rows.forEach((row, index) => {
        const y = 30 + index * 42;
        const x = 90 + ((row.start - picture.min) / span) * 232;
        const width = ((row.end - row.start) / span) * 232;
        svg.append(sv("rect", {x, y, width, height: 20, class: "dsa-box " + row.state}));
        svg.append(sv("text", {x: 4, y: y + 15}, row.name));
        svg.append(sv("text", {x: 335, y: y + 15}, row.state));
      });
      return svg;
    },
    rows(picture) {
      const wrap = el("div", undefined, "dsa-rows");
      for (const row of picture.rows) {
        const line = el("div", undefined, "dsa-row");
        if (row.label) line.append(el("span", row.label, "dsa-row-label"));
        for (const cell of row.cells) {
          const text = cell.mark ? cell.text + " (" + cell.mark + ")" : cell.text || "empty";
          line.append(el("span", text, "dsa-cell" + (cell.mark ? " " + cell.mark : "")));
        }
        wrap.append(line);
      }
      return wrap;
    },
  };

  function mount(host, scene) {
    let caseIndex = 0;
    let stepIndex = 0;
    const root = el("section", undefined, "dsa-scene");
    root.dataset.scene = scene.id;
    root.tabIndex = 0;
    root.setAttribute("aria-label", scene.title);
    const titles={DA01:"Keeping the first occurrence",DA02:"What gets copied",DA03:"Keys, buckets and collisions",DA04:"How values move into order",DA05:"Narrowing the search interval",DA06:"Ruling out impossible pairs",DA07:"Reusing a total",DA08:"Choosing which item comes out next",DA09:"Calls going down; answers coming back",DA10:"Exploring choices and stopping branches",DA11:"Following keys and prefixes",DA12:"One heap, shown as a tree and an array",DA13:"The graph and the waiting work",DA14:"Dependencies versus route costs",DA15:"Building the edit-distance table",DA16:"Comparing skip and take",DA17:"Which interval leaves room for more?",DA18:"What the stream keeps and discards",DA19:"Finding a counterexample",DA20:"Connecting the stages"};
    const heading = el("h3", titles[scene.id]||scene.title);
    const legend = el("ul", undefined, "dsa-legend");
    for (const item of scene.legend || []) legend.append(el("li", item));
    const picker = el("label", "Case ");
    const select = el("select");
    scene.cases.forEach((item, index) => {
      const option = el("option", item.label);
      option.value = String(index);
      select.append(option);
    });
    picker.append(select);
    const prompt = el("p", undefined, "dsa-prompt");
    const layout = el("div", undefined, "dsa-layout");
    const picture = el("div", undefined, "dsa-picture");
    const state = el("div",undefined,"dsa-state");
    const side = el("div", undefined, "dsa-side");
    const count = el("p", undefined, "dsa-count");
    const action = el("p", undefined, "dsa-action");
    const code = el("p", undefined, "dsa-code");
    const predict = el("p", undefined, "dsa-predict");
    const why = el("p", undefined, "dsa-why");
    const alt = el("p", undefined, "dsa-alt");
    const accessible=el("details",undefined,"dsa-text-state");
    accessible.append(el("summary","Read this picture as text"),alt);
    const nav = el("div", undefined, "dsa-nav");
    const previous = el("button", "Previous");
    const next = el("button", "Next");
    const reset = el("button", "Reset");
    for (const button of [previous, next, reset]) button.setAttribute("type", "button");
    nav.append(previous, next, reset);
    // State changes belong here; complete control flow belongs in the recipe below.
    side.append(count, action, why, state, code, predict, nav, accessible);
    layout.append(picture, side);
    const key=el("details",undefined,"dsa-picture-key");key.append(el("summary","How to read the picture"),legend);
    root.append(el("p","1 · Follow a worked example","dsa-eyebrow"),heading,picker,prompt,layout,key);
    host.append(root);

    function render() {
      const item = scene.cases[caseIndex];
      const step = item.steps[stepIndex];
      select.value = String(caseIndex);
      prompt.textContent = item.prompt;
      count.textContent = "Step " + (stepIndex + 1) + " of " + item.steps.length;
      action.textContent = step.action;
      code.textContent = step.code ? "Operation: " + step.code : "";
      predict.textContent = step.predict ? "Before moving on: " + step.predict : "";
      why.textContent = step.why;
      alt.textContent = "Text alternative: " + step.alt;
      picture.setAttribute("aria-label",step.alt);
      state.replaceChildren();
      if(step.picture.kind==="graph"){
        for(const [label,values] of [[step.picture.frontierLabel||"Waiting, in order",step.picture.frontierEntries||step.picture.frontier],["Current DFS path",step.picture.path]]){
          if(values?.length)state.append(el("p",label+": "+values.join(" → ")));
        }
        const visitedLabel=step.picture.visitedLabel||"discovered";
        state.append(el("p",visitedLabel.charAt(0).toUpperCase()+visitedLabel.slice(1)+": "+((step.picture.visited||[]).join(", ")||"none")));
      }
      previous.disabled = stepIndex === 0;
      next.disabled = stepIndex === item.steps.length - 1;
      const painter = draw[step.picture.kind];
      if (!painter) throw Error("Unknown DSA picture.");
      picture.replaceChildren(painter(step.picture));
    }
    function move(delta) {
      const steps = scene.cases[caseIndex].steps;
      const target = stepIndex + delta;
      if (target < 0 || target >= steps.length) return;
      stepIndex = target;
      render();
    }
    previous.onclick = () => move(-1);
    next.onclick = () => move(1);
    reset.onclick = () => { stepIndex = 0; render(); };
    select.onchange = () => { caseIndex = Number(select.value); stepIndex = 0; render(); };
    root.addEventListener("keydown", event => {
      if(["SELECT","INPUT","TEXTAREA","BUTTON"].includes(event.target?.tagName))return;
      if (event.key === "ArrowRight") { event.preventDefault(); move(1); }
      else if (event.key === "ArrowLeft") { event.preventDefault(); move(-1); }
    });
    render();
    return root;
  }

  function paragraph(parent,title,text){
    parent.append(el("h3",title),el("p",text));
  }

  function introduce(host,teaching){
    const intro=el("section",undefined,"dsa-introduction");
    intro.append(el("p",teaching.definition,"dsa-definition"));
    paragraph(intro,"The problem",teaching.problem);
    paragraph(intro,"The idea before the code",teaching.intuition);
    const terms=el("dl",undefined,"dsa-terms");
    for(const [term,meaning] of teaching.terms){terms.append(el("dt",term),el("dd",meaning));}
    intro.append(terms);host.append(intro);
    const worked=el("section",undefined,"dsa-worked");
    paragraph(worked,"Work through the numbers",teaching.worked);host.append(worked);
  }

  function wrapLabel(text,size=25){
    const lines=[];let current="";
    for(const word of text.split(" ")){
      if(current.length+word.length+1>size&&current){lines.push(current);current=word;}
      else current+=(current?" ":"")+word;
    }
    if(current)lines.push(current);return lines;
  }

  function mountRecipe(host,item,index){
    const arrowId="dsa-flow-arrow-"+(++diagramSerial);
    const root=el("section",undefined,"dsa-recipe");root.dataset.recipe=String(index);
    root.append(el("p","2 · Connect the decisions to Python","dsa-eyebrow"),el("h3",item.title));
    root.append(el("p","Follow the arrows to understand the procedure. Select a box to highlight the matching code and read why it is there. This inspects the recipe; it does not run the program.","dsa-recipe-help"));
    const layout=el("div",undefined,"dsa-recipe-layout");
    const graph=el("div",undefined,"dsa-flowgraph");
    graph.setAttribute("aria-label","Decision flowchart; scroll to follow the complete recipe");
    const height=item.nodes.length*92+30;
    const svg=sv("svg",{viewBox:`0 0 460 ${height}`,role:"group","aria-label":item.title+" flowchart"});
    const defs=sv("defs"),marker=sv("marker",{id:arrowId,viewBox:"0 0 10 10",refX:9,refY:5,markerWidth:6,markerHeight:6,orient:"auto"});
    marker.append(sv("path",{d:"M 0 0 L 10 5 L 0 10 z",fill:"#667f91"}));defs.append(marker);svg.append(defs);
    const x=226,y=i=>50+i*92,positions=new Map(item.nodes.map((n,i)=>[n.id,i]));
    let back=0,forward=0;
    for(const edge of item.edges){
      const a=positions.get(edge.from),b=positions.get(edge.to),ya=y(a),yb=y(b),next=b===a+1;
      let d,lx,ly;
      if(next){d=`M ${x} ${ya+35} L ${x} ${yb-36}`;lx=x+10;ly=ya+47;}
      else if(b<=a){const lane=22+(back++%3)*17;d=`M ${x-140} ${ya} H ${lane} V ${yb} H ${x-141}`;lx=lane+4;ly=(ya+yb)/2;}
      else {const lane=415+(forward++%2)*20;d=`M ${x+140} ${ya} H ${lane} V ${yb} H ${x+141}`;lx=lane-4;ly=(ya+yb)/2;}
      svg.append(sv("path",{d,class:"dsa-flow-edge","marker-end":`url(#${arrowId})`}));
      if(edge.label)svg.append(sv("text",{x:lx,y:ly,class:"dsa-edge-label","text-anchor":!next&&b>a?"end":"start"},edge.label));
    }
    const codePanel=el("div",undefined,"dsa-recipe-code");
    const codeTitle=el("p","Python recipe","code-label");
    const code=el("pre",undefined,"dsa-numbered-code");
    const codeLines=item.code.split("\n").map((line,i)=>{
      const row=el("span",undefined,"dsa-code-line");row.dataset.line=String(i+1);
      const number=el("span",String(i+1),"dsa-line-number");number.setAttribute("aria-hidden","true");
      row.append(number,el("code",line||" "));code.append(row);return row;
    });
    const explanation=el("div",undefined,"dsa-code-explanation");explanation.setAttribute("aria-live","polite");
    const output=el("div",undefined,"dsa-recipe-outputs");
    output.append(el("p","Example calls and checked results","code-label"));
    for(const example of item.examples){
      const pre=el("pre");pre.append(el("code",example.expression+"\n→ "+example.output));output.append(pre);
    }
    const groups=[];
    function select(node){
      let lineIndex=item.code.split("\n").findIndex(line=>line.includes(node.anchor));
      for(const [i,row] of codeLines.entries())row.setAttribute("class","dsa-code-line"+(i===lineIndex?" active":""));
      for(const [id,group] of groups){group.setAttribute("aria-pressed",String(id===node.id));}
      explanation.replaceChildren(el("strong",node.label),el("p",node.explanation),el("small","Code line "+(lineIndex+1)));
      root.dataset.selectedNode=node.id;
      // Keep the selected line inside its own bounded panel, not move the page.
      const line=codeLines[lineIndex];
      if(line&&typeof line.offsetTop==="number")code.scrollTop=Math.max(0,line.offsetTop-code.offsetTop-code.clientHeight/2);
    }
    for(const [i,node] of item.nodes.entries()){
      const group=sv("g",{class:"dsa-flow-node",role:"button",tabindex:0,"aria-label":node.label,"aria-pressed":"false","data-node":node.id});
      const cy=y(i);
      if(node.kind==="decision")group.append(sv("path",{d:`M ${x} ${cy-35} L ${x+140} ${cy} L ${x} ${cy+35} L ${x-140} ${cy} Z`,class:"dsa-flow-shape"}));
      else group.append(sv("rect",{x:x-140,y:cy-30,width:280,height:60,rx:node.kind==="start"||node.kind==="end"?28:7,class:"dsa-flow-shape"}));
      const lines=wrapLabel(node.label);
      lines.forEach((line,k)=>group.append(sv("text",{x,y:cy+(k-(lines.length-1)/2)*16+5,"text-anchor":"middle"},line)));
      group.onclick=()=>select(node);
      group.addEventListener("keydown",event=>{if(event.key==="Enter"||event.key===" "){event.preventDefault();select(node);}});
      groups.push([node.id,group]);svg.append(group);
    }
    graph.append(svg);
    codePanel.append(codeTitle,code,explanation,output);layout.append(graph,codePanel);root.append(layout);host.append(root);
    select(item.nodes[0]);return root;
  }

  function teach(host,teaching){
    if(teaching.recipes.length===1)mountRecipe(host,teaching.recipes[0],0);
    else{
      const picker=el("nav",undefined,"dsa-recipe-picker");picker.setAttribute("aria-label","Algorithm recipes in this lesson");
      const panels=el("div"),buttons=[],recipes=[];
      teaching.recipes.forEach((item,index)=>{
        const button=el("button",item.title);button.type="button";
        button.onclick=()=>{recipes.forEach((panel,i)=>{panel.hidden=i!==index;buttons[i].setAttribute("aria-pressed",String(i===index));});};
        buttons.push(button);picker.append(button);const panel=mountRecipe(panels,item,index);panel.hidden=index!==0;recipes.push(panel);
        button.setAttribute("aria-pressed",String(index===0));
      });
      host.append(picker,panels);
    }
    const reasoning=el("section",undefined,"dsa-reasoning");
    paragraph(reasoning,"What the work costs",teaching.cost);
    paragraph(reasoning,"Where this can go wrong",teaching.trap);
    const practice=el("section",undefined,"dsa-practice");
    paragraph(practice,"Try explaining it without the code",teaching.practice);
    const answer=el("details");answer.append(el("summary","Compare your reasoning"),el("p",teaching.answer));practice.append(answer);
    host.append(reasoning,practice);
  }

  return {mount,introduce,teach};
})();
