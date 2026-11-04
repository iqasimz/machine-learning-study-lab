"use strict";

// Read-only interpretations of saved measurements, never invented scores or rankings.
const LabReadouts=(()=>{
  const metrics={
    accuracy:{name:"Accuracy",higher:true,meaning:"The share of cases assigned the correct class. A common class can dominate it."},
    macro_f1:{name:"Macro-F1",higher:true,meaning:"Precision/recall balance calculated for each class, then averaged with equal class weight."},
    balanced_accuracy:{name:"Balanced accuracy",higher:true,meaning:"Recall averaged equally across classes, rather than weighted by their frequency."},
    log_loss:{name:"Log loss",higher:false,meaning:"The penalty for assigning too little probability to the correct class; confident errors are expensive."},
    brier_multiclass:{name:"Multiclass Brier score",higher:false,meaning:"Squared probability errors summed over classes and averaged over cases; this lab does not halve the sum."},
    ece_10_bins:{name:"Calibration gap (ECE)",higher:false,meaning:"A ten-bin summary of the gap between confidence and observed accuracy. Its binning and sample size affect the value."},
    mae:{name:"Mean absolute error",higher:false,meaning:"Average error size in the target's original units. It does not show whether errors are mostly high or low."},
    rmse:{name:"Root mean squared error",higher:false,meaning:"Error in the target's units, with larger misses receiving more weight than in MAE."},
    r2:{name:"R²",higher:true,meaning:"Variation explained relative to predicting the mean of the scored targets. Zero is that reference level; negative values are possible. A baseline fitted on training targets can have a different score."},
    dev_nll:{name:"Development loss per token",higher:false,meaning:"Average penalty for predicting the known next token on selection text; token units and scoring rules must match."},
    test_nll:{name:"Test loss per token",higher:false,meaning:"The same next-token penalty on held-out test text. Repeated test-guided choices compromise its final-check role."},
    dev_bits_per_byte:{name:"Development bits per byte",higher:false,meaning:"Prediction cost per raw text byte. Matching raw text is necessary; context and scoring conventions still matter."},
    test_bits_per_byte:{name:"Test bits per byte",higher:false,meaning:"Prediction cost per raw test-text byte. It does not make different training budgets or context windows equivalent."}
  };
  const number=x=>Number.isFinite(x)?x.toFixed(4):"not recorded";
  function difference(value,reference){return Number.isFinite(value)&&Number.isFinite(reference)?(value-reference>0?"+":"")+(value-reference).toFixed(4):"not recorded";}
  function metric(key){return metrics[key]||{name:key,higher:null,meaning:"Read this metric's definition and scoring units before interpreting its value."};}
  function modelResult(run){
    if(!run)return {title:"No saved result yet",text:"A completed experiment will appear here after you explicitly train a model. Reading a lesson does not create a run."};
    if(run.manifest.status!=="MEASURED")return {title:"This run has no completed evaluation",text:"Progress measurements may be available, but a stopped, failed or still-running fit is not a finished model comparison. Check its status and error before interpreting the curves."};
    const r=run.result,s=run.manifest.settings;if(!r)return {title:"Evaluation is not available",text:"The saved run has no result record to interpret. Check its artifacts rather than treating a missing score as zero."};
    if(r.predictions){const key=s.task==="regression"?"mae":"macro_f1",m=metric(key),value=r.predictions.dev.scores[key],base=r.baseline?.dev?.[key];
      return {title:"What this run shows",text:`On development examples, ${m.name} is ${number(value)}${Number.isFinite(base)?` versus ${number(base)} for a constant baseline (model minus baseline: ${difference(value,base)})`:"; no matching baseline value was recorded"}. ${m.higher?"Higher":"Lower"} is better for this metric. ${m.meaning} This describes one fitted run, not certainty about future data.`};}
    if(r.representation)return {title:"What the reconstruction score shows",text:`Development reconstruction error is ${number(r.representation.reconstruction_mse.dev)}. Lower means the reconstructed inputs are closer under the stated units. Inspect lost details as well; reconstruction alone does not establish useful categories or meaning.`};
    if(r.clustering)return {title:"What these groups mean",text:"The model formed groups from the input representation, without fitting the reference class labels. Inspect their geometry, stability and usefulness. Reference-label agreement is a diagnostic, not proof that the groups are meaningful for a real decision."};
    if(r.anomaly)return {title:"What the anomaly test shows",text:"This run tests whether generated unusual points receive higher anomaly scores than ordinary cluster points. That is a controlled synthetic check, not evidence of fraud detection. The share flagged also depends on the chosen cutoff."};
    return {title:"No recognized score record is available",text:"This result has no supported prediction, reconstruction, clustering or anomaly score to interpret. Inspect its saved artifacts; a missing result must not be described as a successful evaluation."};
  }
  function modelComparison(c,key){
    const m=metric(key),first=c.rows[0];
    if(["classification","regression"].includes(c.task))return {title:"Read quality and cost together",text:`The first row (${first.id}) is the reference for the score differences below. ${m.name}: ${m.meaning} ${m.higher?"Higher":"Lower"} is better. A positive difference means a larger number, not automatically an improvement. Compare development quality with fitting time and settings; the table does not estimate uncertainty in the difference.`};
    if(c.task==="representation")return {title:"Compare the same reconstruction problem",text:"Lower reconstruction error means less information lost only when the input scale, preprocessing and scoring units match. PCA and autoencoders can use different units here; inspect the Units column before ranking numbers. This is not a classification comparison."};
    if(c.task==="clustering")return {title:"Compare useful groupings, not accuracy",text:"Silhouette describes separation under a chosen distance; adjusted Rand compares with reference labels; noise fraction counts unassigned points. Check all three and inspect examples. A high silhouette alone does not show that the groups help a real task."};
    return {title:"Separate ranking from alert volume",text:"ROC-AUC measures how the anomaly score orders the synthetic ordinary/unusual examples. Flagged fraction is how many pass the cutoff. A model can flag more points without ranking them better; neither number establishes real-world fraud performance."};
  }
  function languageResult(run){
    if(!run||run.manifest.status!=="MEASURED"||!run.evaluation)return {title:"A completed evaluation is not available",text:"The curves may show partial progress. Wait for a completed evaluation or inspect the run's failure message; missing results are not zero loss."};
    const e=run.evaluation,lora=run.manifest.engine==="lora",base=e.pretrained||e.bigram,value=e.transformer?.dev?.nll,b=base?.dev?.nll;
    return {title:"What the saved evaluation shows",text:`Development loss is ${number(value)}${Number.isFinite(b)?` versus ${number(b)} for ${lora?"the unchanged pretrained model":"the previous-token bigram baseline"} (difference: ${difference(value,b)})`:"; no matching reference loss was recorded"}. Lower means better prediction of these known target tokens, not necessarily more factual or helpful answers. The selected checkpoint is update ${e.checkpoint_step}; zero can mean that further training did not beat the starting model.`};
  }
  function languageComparison(c,key){
    if(!key)return {title:"These scores cannot be ranked together",text:"The saved runs do not share an enabled scoring unit/protocol. Read the compatibility messages and compare the setups descriptively, or make a new matched experiment. An empty curve is not zero loss."};
    const m=metric(key);return {title:"What this comparison can tell you",text:`${m.name}: ${m.meaning} Lower is better. The first row is the reference for the differences. Look at training-token exposure and changed settings before attributing the difference to one method; equal update counts do not guarantee equal work. Check generated answers separately for factual and task errors.`};
  }
  return {metric,number,difference,modelResult,modelComparison,languageResult,languageComparison};
})();
