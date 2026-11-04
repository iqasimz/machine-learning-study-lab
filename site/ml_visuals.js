/* Complete static figures for the ML and DL lessons. Values are illustrative, not run metrics. */
const ModelFigures = (() => {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const C = {ink:"#26364c",muted:"#627383",blue:"#3664a3",teal:"#28796e",line:"#b8c5cf",pale:"#e8eef5",green:"#e1eee7",sand:"#f4eadd"};
  const definitions = {
    selection: ["Model selection and the untouched test set", "A hyperparameter is a setting chosen before fitting. A fold is one temporary data split. Inner validation chooses settings; outer validation estimates the selection procedure. A final test set is kept outside both loops."],
    ensembles: ["One tree, bagging and gradient boosting", "A tree makes successive feature-based decisions. Bagging fits separate models on resampled training data and combines predictions. Gradient boosting adds models sequentially to improve the current prediction under a chosen loss."],
    pca: ["PCA: compress, then reconstruct", "A principal component is a unit direction of variation learned from centered training data. Projection records a point's coordinate along that direction. Reconstruction places it back in the original feature space, losing discarded directions."],
    backprop: ["Backpropagation through one calculation", "The forward pass computes predictions and loss. A derivative measures how a small change affects a result. Backpropagation multiplies local derivatives to compute each weight's effect on loss; the optimizer then changes weights."],
    convolution: ["A convolution reuses one filter across an image", "A filter is a small learned array of weights. A patch is the matching part of the input at one position. A feature map collects the filter's outputs. Stride is the movement between patches; padding adds boundary cells."],
    recurrence: ["A recurrent model reuses a state update", "A hidden state is a learned summary carried between time steps. Unrolling draws each use of the same weights separately. Backpropagation through time follows this unfolded calculation backward and sums the shared weights' gradient contributions."]
  };
  const node = (tag,text,cls) => {const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;};
  const sv = (tag,attrs,text) => {const n=document.createElementNS(NS,tag);Object.entries(attrs||{}).forEach(([k,v])=>n.setAttribute(k,String(v)));if(text!==undefined)n.textContent=text;return n;};
  function mount(host,id) {
    if(!definitions[id])throw new Error("Unknown ML/DL figure: "+id);
    const root=node("figure",undefined,"visual-lab");
    root.setAttribute("aria-label",definitions[id][0]);
    const header=node("div",undefined,"visual-title");header.append(node("h3",definitions[id][0]));
    const body=node("div",undefined,"visual-display");
    body.append(node("p",definitions[id][1],"topic-terms"));root.append(header,body);host.replaceChildren(root);
    let svg;
    function canvas(height,label){svg=sv("svg",{viewBox:`0 0 720 ${height}`,role:"img",class:"visual-svg","aria-label":label});svg.append(sv("title",{},label));body.append(svg);}
    function text(x,y,t,attrs={}){svg.append(sv("text",{x,y,fill:C.ink,"font-size":13,...attrs},t));}
    function rect(x,y,w,h,fill=C.pale){svg.append(sv("rect",{x,y,width:w,height:h,rx:5,fill,stroke:C.line}));}
    function box(x,y,w,h,lines,fill=C.pale){rect(x,y,w,h,fill);lines=Array.isArray(lines)?lines:[lines];lines.forEach((s,i)=>text(x+w/2,y+h/2+4+(i-(lines.length-1)/2)*19,s,{"text-anchor":"middle"}));}
    function line(x1,y1,x2,y2,color=C.line,dash=false){svg.append(sv("line",{x1,y1,x2,y2,stroke:color,"stroke-width":1.7,...(dash?{"stroke-dasharray":"5 4"}:{})}));}
    function arrow(x1,y1,x2,y2,color=C.muted,dash=false){line(x1,y1,x2,y2,color,dash);const a=Math.atan2(y2-y1,x2-x1),r=7;svg.append(sv("polygon",{points:`${x2},${y2} ${x2-r*Math.cos(a-.45)},${y2-r*Math.sin(a-.45)} ${x2-r*Math.cos(a+.45)},${y2-r*Math.sin(a+.45)}`,fill:color}));}
    function finish(caption,steps,sources){body.append(node("figcaption",caption,"diagram-caption"));const list=node("ol",undefined,"diagram-steps");steps.forEach(s=>list.append(node("li",s)));body.append(list);const links=node("p",undefined,"diagram-sources");links.append(node("span","Sources: "));sources.forEach(([label,url],i)=>{if(i)links.append(node("span"," · "));const a=node("a",label);a.href=url;a.target="_blank";a.rel="noopener noreferrer";links.append(a);});body.append(links);}

    if(id==="selection") {
      canvas(460,"Nested validation operates only inside development data; a separate final test set stays untouched until selection is complete");
      box(20,18,455,48,"Development data: used for selection",C.green);
      box(515,18,185,48,"Final test: untouched",C.sand);
      arrow(247,66,247,98);
      rect(20,98,455,228,"#f8fafb");text(38,124,"One outer fold (repeat over outer folds)",{"font-weight":600});
      box(40,145,255,45,"Outer-training portion",C.green);box(315,145,140,45,["Outer-validation","portion"]);
      arrow(167,190,167,217);box(40,217,255,54,["Inner folds choose settings","Fit preprocessing inside each fold"]);
      arrow(167,271,167,293);text(40,313,"Refit chosen pipeline on outer-training data");
      arrow(295,242,385,242);arrow(385,242,385,190);
      text(493,155,"Not used to tune",{"font-size":12});text(493,176,"or inspect candidates",{"font-size":12});
      arrow(247,326,247,355);box(20,355,455,76,["After the procedure is fixed:","select settings on development data; refit there"],C.green);
      arrow(475,393,680,393);arrow(680,393,680,66,C.teal);text(498,339,"Final evaluation",{"font-size":12});text(498,358,"of the fixed pipeline",{"font-size":12});
      finish("The diagram includes an additional untouched test set. Outer cross-validation estimates the whole selection procedure; it is not another opportunity to tune candidates against its results.",[
        "Inside each outer-training portion, inner folds compare candidate settings. Every learned preprocessing step is fitted on that fold's training portion only.",
        "Refit the inner-selected pipeline on the outer-training portion and score its outer-validation portion. Repeat to estimate how the selection procedure generalizes.",
        "Once choices are fixed, select and refit using development data, then evaluate the final pipeline on the held-out test set. Repeated test-driven revision makes that set part of development."
      ],[["Nested cross-validation", "https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html"],["Data leakage", "https://scikit-learn.org/stable/common_pitfalls.html#data-leakage"]]);
    } else if(id==="ensembles") {
      canvas(450,"A single decision tree versus parallel bagged trees and sequential gradient-boosted trees");
      text(20,29,"One tree",{"font-weight":600});box(20,48,145,44,"Training data");arrow(92,92,92,115);box(20,115,145,44,"Fit one tree");arrow(92,159,92,183);box(20,183,145,44,"One prediction",C.green);
      text(210,29,"Bagging / random forest",{"font-weight":600});box(210,48,230,44,"Resample training examples");
      [0,1,2].forEach(i=>{const x=210+i*80;arrow(325,92,x+35,115);box(x,115,70,44,"Tree "+(i+1));arrow(x+35,159,325,183);});box(210,183,230,44,"Average / vote",C.green);
      text(485,29,"Gradient boosting",{"font-weight":600});box(485,48,215,44,"Start with baseline prediction");arrow(592,92,592,115);box(485,115,215,44,"Fit tree to loss gradients");arrow(592,159,592,183);box(485,183,215,44,"Add a scaled correction",C.green);
      arrow(700,205,710,205,C.teal);line(710,205,710,136,C.teal);arrow(710,136,700,136,C.teal);
      rect(20,267,680,155,"#f8fafb");text(38,294,"What changes between these methods?",{"font-weight":600});
      text(38,324,"One tree: depth controls detail; small data changes can change its splits.");
      text(38,352,"Bagging: combine varied trees; random forests also sample candidate features.");
      text(38,380,"Boosting: each added tree depends on the current ensemble's remaining error.");
      text(38,405,"These are different training strategies, not a guaranteed ranking of accuracy.",{"font-size":12});
      finish("For regression, bagging usually averages predictions. Classification aggregation depends on the estimator; a random forest averages class probabilities. Gradient boosting adds scaled trees under a differentiable loss.",[
        "A single tree recursively partitions feature space. Increasing depth can fit more detail but also more noise.",
        "Bagging fits models separately on bootstrap samples. Combining imperfectly correlated predictions can reduce variance; nearly identical errors limit the gain.",
        "Gradient boosting fits successive loss-gradient corrections. More trees and a smaller learning rate interact; use validation to choose them rather than assuming the largest ensemble wins."
      ],[["Ensemble methods", "https://scikit-learn.org/stable/modules/ensemble.html"],["Decision trees", "https://scikit-learn.org/stable/modules/tree.html"]]);
    } else if(id==="pca") {
      canvas(425,"A two-dimensional point is projected onto one principal direction and reconstructed on that line");
      arrow(60,290,395,290);arrow(60,290,60,45);text(366,314,"Feature 1");text(20,30,"Feature 2");
      line(85,270,330,86,C.teal);text(226,68,"First principal direction",{"font-size":12,fill:C.teal});
      const points=[[105,248],[132,246],[153,219],[169,231],[192,185],[211,196],[236,153],[263,159],[280,124],[310,113]];
      points.forEach(([x,y])=>svg.append(sv("circle",{cx:x,cy:y,r:4,fill:C.muted})));
      svg.append(sv("circle",{cx:230,cy:110,r:6,fill:C.blue}));text(150,90,"Original x");
      svg.append(sv("circle",{cx:254.54,cy:142.67,r:5,fill:C.teal}));line(230,110,254.54,142.67,C.blue,true);
      text(270,161,"Reconstruction",{"font-size":12});text(274,180,"on the line",{"font-size":12});
      box(435,70,265,57,["Center using training mean μ","x − μ"],C.green);arrow(567,127,567,156);
      box(435,156,265,57,["Project onto unit direction u","z = uᵀ(x − μ)"]);arrow(567,213,567,242);
      box(435,242,265,57,["Reconstruct in feature space","x̂ = μ + uz"],C.green);
      text(20,353,"Illustrative geometry, not a fitted dataset or measured score.",{"font-size":12});
      text(20,382,"The dropped perpendicular component is reconstruction error.",{"font-size":13});
      finish("Keeping one component stores one coordinate instead of two. PCA learns directions of input variance, not directions that necessarily predict the target. Centering and optional scaling must be learned on training data only.",[
        "Subtract the training mean so the origin represents the center of the data. Different feature units can dominate variance, so decide whether standardization is appropriate.",
        "Choose orthogonal directions with the largest training variance and project onto them. For one retained direction u, the compressed coordinate is z = uᵀ(x − μ).",
        "Map z back to the retained subspace to reconstruct x. More components usually reduce training reconstruction error, but downstream predictive usefulness must be validated separately."
      ],[["PCA", "https://scikit-learn.org/stable/modules/decomposition.html#pca"]]);
    } else if(id==="backprop") {
      canvas(405,"A scalar example shows forward values followed by reverse derivatives and a separate optimizer update");
      text(20,30,"Forward: compute values",{"font-weight":600});
      box(20,57,120,65,["Input x = 2","Weight w = 3"]);arrow(140,89,185,89);
      box(185,57,150,65,["Prediction a = wx","a = 6"]);arrow(335,89,380,89);
      box(380,57,150,65,["Error e = a − y","y = 4; e = 2"]);arrow(530,89,575,89);
      box(575,57,125,65,["Loss L = ½e²","L = 2"],C.sand);
      text(20,177,"Backward: compute sensitivities",{"font-weight":600});
      box(575,204,125,63,["dL/de = e","= 2"],C.green);arrow(575,235,530,235,C.teal);
      box(380,204,150,63,["de/da = 1","dL/da = 2"],C.green);arrow(380,235,335,235,C.teal);
      box(185,204,150,63,["da/dw = x = 2","dL/dw = 2 × 2 = 4"],C.green);
      arrow(260,267,260,314);box(20,314,680,62,["Only the optimizer changes the weight: w_new = w − learning_rate × gradient","One SGD step at learning rate 0.1: 3 − 0.1 × 4 = 2.6"],C.sand);
      finish("This hand calculation uses one scalar weight, one example and squared error. The backward pass computes a derivative; it does not itself update w. A real network applies the same chain rule across many tensor operations.",[
        "Run the forward calculation with the current weight. Save the intermediate values needed by the backward rules: here x and the prediction error.",
        "Start from the loss and multiply derivatives backward: dL/dw = (dL/de) × (de/da) × (da/dw) = 2 × 1 × 2 = 4. Shared paths add contributions.",
        "Apply the chosen optimizer at its update boundary. In this example the new prediction is 2.6 × 2 = 5.2 and the new loss is 0.72; larger steps are not always better."
      ],[["PyTorch automatic differentiation", "https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html"]]);
    } else if(id==="convolution") {
      canvas(438,"A three-by-three input patch and shared filter produce one output; the filter slides across the full image");
      const patch=[[1,2,0],[0,1,3],[2,0,1]],filter=[[1,0,-1],[1,0,-1],[1,0,-1]];
      text(20,28,"One 3 × 3 input patch",{"font-weight":600});text(255,28,"One learned 3 × 3 filter",{"font-weight":600});
      function grid(x,y,values){values.forEach((row,i)=>row.forEach((v,j)=>{box(x+j*48,y+i*48,42,42,String(v),i===1?C.green:C.pale);}));}
      grid(20,50,patch);grid(255,50,filter);text(191,121,"×",{"font-size":22});
      arrow(401,120,452,120);box(465,75,235,90,["Multiply matching cells","Sum products; add bias b = 0","Output = −1"],C.green);
      rect(20,226,680,184,"#f8fafb");text(38,255,"Move the same filter to the next patch",{"font-weight":600});
      text(38,285,"5 × 5 input · 3 × 3 filter · stride 1 · padding 0 → 3 × 3 output");
      text(38,316,"One input channel, one output channel: 3 × 3 + 1 bias = 10 parameters");
      text(38,347,"With C_in input channels and C_out filters (groups = 1):");
      text(38,376,"parameters = C_out × (C_in × 3 × 3 + 1 bias)",{"font-weight":600});
      finish("Illustrative filter values show one cross-correlation calculation, which is the operation commonly called convolution in deep-learning libraries. The input patch sum is (1 − 0) + (0 − 3) + (2 − 1) = −1, before any activation.",[
        "Take a local patch and multiply each entry by the filter entry at the same position. Sum across both spatial positions and all input channels, then add the output channel's bias.",
        "Reuse those learned weights at every valid location. Stride and padding change the output grid, not the number of filter parameters; dilation changes the spacing within a patch.",
        "Each output channel has its own filter. For dilation 1, output width is floor((input_width + 2 × padding − kernel_width) / stride) + 1; the same rule applies to height."
      ],[["PyTorch Conv2d", "https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html"]]);
    } else if(id==="recurrence") {
      canvas(455,"Three recurrent time steps share weights; losses send gradients backward through the unrolled state chain");
      text(20,27,"Forward: h_t = tanh(W_x x_t + W_h h_(t−1) + b)",{"font-weight":600});
      text(20,46,"All state boxes below reuse the same W_x, W_h and b.",{"font-size":12});
      box(20,105,75,48,"h0 = 0");
      [0,1,2].forEach(i=>{const x=140+i*190;box(x,53,140,38,"Input x"+(i+1));box(x,105,140,48,"State h"+(i+1),C.green);box(x,192,140,48,"Loss L"+(i+1),C.sand);arrow(x+70,91,x+70,105);arrow(x+70,153,x+70,192);if(i===0)arrow(95,129,x,129);else arrow(x-50,129,x,129);});
      text(20,283,"Backward through time:",{"font-size":12,"font-weight":600});text(20,303,"sum shared gradients",{"font-size":12});
      arrow(590,308,400,308,C.teal);arrow(400,308,210,308,C.teal);
      line(590,240,590,308,C.teal,true);line(400,240,400,308,C.teal,true);line(210,240,210,308,C.teal,true);
      rect(20,341,680,88,"#f8fafb");text(38,370,"Unrolling creates repeated operations, not new independent copies of weights.");
      text(38,398,"Shared-weight gradients add across time; repeated Jacobians can shrink or grow.");
      finish("The figure uses a loss at each time step to expose all paths. Sequence classification may instead score only the final output. The initial state can be zeros or another specified state; it must not silently mix unrelated sequences.",[
        "Read the next input and combine it with the previous hidden state using the same recurrent parameters. The new state can affect both the current output and later states.",
        "Backpropagation through time differentiates the unfolded computation. A weight receives contributions from every use that can influence the scored losses; these contributions are summed before the optimizer update.",
        "Truncated backpropagation detaches state at chosen boundaries: values may carry forward, but gradients do not cross the detach. LSTM/GRU gates change the state update; they do not remove the need for honest split and masking rules."
      ],[["PyTorch RNN", "https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html"],["LSTM", "https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html"]]);
    }
    return {destroy(){root.remove();}};
  }
  return {mount};
})();
