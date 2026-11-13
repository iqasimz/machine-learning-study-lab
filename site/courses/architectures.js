const visual = (heading, content, note) =>
  '<section class="visual"><strong>' + heading + '</strong>' + content + '<p class="widget-caption">' + note + '</p></section>';

function grid(rows, selected, cell, columns, rowLabels, corner='Q / K') {
  return '<table class="matrix">' +
    (columns ? '<thead><tr><th scope="col">'+corner+'</th>' + columns.map(t => '<th scope="col">'+t+'</th>').join('') + '</tr></thead>' : '') +
    '<tbody>' + rows.map((row,r) => '<tr>' +
      (rowLabels ? '<th scope="row">'+rowLabels[r]+'</th>' : '') +
      row.map((value,c) => {
        const active=selected(r,c);
        const mix=Number(cell(value));
        const color=[117,186,177].map(component=>Math.round(255-(255-component)*mix)).join(',');
        return '<td class="' + (active ? 'selected' : '') + '" style="background-color:rgb('+color+')">' + value + '</td>';
      }).join('') + '</tr>').join('') + '</tbody></table>';
}

const fixed = (v,n=2) => Number(v).toFixed(n);
const rowOf = (n, fn) => Array.from({length:n}, (_,i) => fn(i));
const bars = (labels, values) => '<div class="bars">' + values.map((v,i) =>
  '<div><span>'+labels[i]+'</span><meter min="0" max="1" value="'+v+'"></meter><span>'+fixed(100*v,1)+'%</span></div>').join('') + '</div>';

const widgets = {
  conv(v) {
    const r=Math.floor(v/3), c=v%3;
    const input=rowOf(5,()=>[0,0,1,0,0]);
    const kernel=rowOf(3,()=>[0,1,0]);
    const out=rowOf(3,i=>rowOf(3,j=>
      kernel.reduce((total,row,dy)=>total + row.reduce((part,w,dx)=>part+w*input[i+dy][j+dx],0),0)));
    const product=kernel.map((row,dy)=>row.map((w,dx)=>w*input[r+dy][c+dx]));
    return visual('Input · 5 × 5',grid(input,(i,j)=>i>=r&&i<r+3&&j>=c&&j<c+3,x=>x*.65),'Highlighted input patch') +
      visual('Kernel · 3 × 3',grid(kernel,()=>false,x=>x*.65),'Stride 1 · padding 0 · bias 0') +
      visual('Element-wise products',grid(product,()=>false,x=>x*.65),'Sum + bias = '+out[r][c]+'.') +
      visual('Output · 3 × 3',grid(out,(i,j)=>i===r&&j===c,x=>x/5),'Illustrative cross-correlation. Output at row '+(r+1)+', column '+(c+1)+' = '+out[r][c]+'.');
  },
  rnn(v) {
    const inputs=[1,0,1,0,1], states=[0];
    inputs.forEach(x=>states.push(Math.tanh(x+.5*states[states.length-1])));
    const body=grid([inputs,states.slice(1).map(x=>fixed(x,4))],(_,c)=>c===v-1,x=>Number(x)*.5,['1','2','3','4','5'],['Input xₜ','State hₜ'],'Time step');
    return visual('State by time step',body,'Illustrative hand-set weights. At step '+v+', tanh('+inputs[v-1]+' + 0.5 × '+fixed(states[v-1],4)+') = '+fixed(states[v],4)+'.');
  },
  gates(v,input,output) {
    const old=.8, candidate=-.4, updated=v*old+input*candidate;
    return visual('Retain and update memory','<p>cₜ = '+fixed(v)+' × '+old+' + '+input+' × ('+candidate+') = <strong>'+fixed(updated,3)+'</strong></p><p>hₜ = '+output+' × tanh(cₜ) = '+fixed(output*Math.tanh(updated),3)+'</p>','Illustrative scalar gates. The forget gate retains old memory, the input gate scales the candidate, and the output gate exposes the resulting memory.');
  },
  softmax(v) {
    const a=1/(1+Math.exp(-v));
    return visual('Two-token attention',bars(['Token 1','Token 2'],[a,1-a]),'Illustrative scores ['+fixed(v,1)+', 0]. The weighted output vector is ['+fixed(2*a,3)+', '+fixed(4*(1-a),3)+'].');
  },
  reversal(v) {
    const tokens=[2,7,4,9], latents=[.2,.8,-.4,1.1];
    const weights=rowOf(4,i=>rowOf(4,j=>j===3-i?.85:.05));
    const selected=weights[v-1];
    const context=selected.reduce((s,a,i)=>s+a*latents[i],0);
    return visual('Attention weights · output rows, source columns',
      grid(weights.map(row=>row.map(w=>fixed(w)),),(r)=>r===v-1,x=>x,tokens,[9,4,7,2]),
      'Illustrative distributions. Step '+v+' targets '+tokens[4-v]+'. With source states [0.2, 0.8, −0.4, 1.1], the weighted context is '+fixed(context,3)+'. Actual learned attention can differ.');
  },
  encoder(v) {
    const tokens=['CLS','A','x','B'];
    const weights=[[.25,.35,.25,.15],[.15,.35,.35,.15],[.1,.4,.3,.2],[.1,.2,.5,.2]];
    return visual('Attention weights by query',
      grid(weights.map(row=>row.map(w=>fixed(w))),(r)=>r===v-1,x=>x,tokens,tokens),
      'Illustrative hand-designed attention. Selected query: '+tokens[v-1]+'. Each query can mix information from every key and each row sums to 1.');
  },
  causal(v) {
    const labels=['t','h','e','␣','c'];
    const mask=rowOf(5,r=>rowOf(5,c=>c<=r?'✓':'×'));
    return visual('Allowed query–key pairs',grid(mask,r=>r===v-1,x=>x==='✓'?.47:0,labels,labels),
      'Structural mask. The selected query can read '+labels.slice(0,v).join(' · ')+'. Future keys receive zero softmax weight.');
  },
  temperature(v) {
    const logits=[2.8,1.7,.7,-.2];
    const exponentials=logits.map(x=>Math.exp((x-logits[0])/v));
    const sum=exponentials.reduce((a,b)=>a+b,0);
    return visual('Softmax at τ = '+fixed(v),
      bars(logits.map(x=>'Logit '+x),exponentials.map(x=>x/sum)),
      'Illustrative fixed logits. Low temperature concentrates probability on high logits.');
  }
};
if (typeof module !== 'undefined') module.exports = widgets;
if (typeof document !== 'undefined') {
document.documentElement.classList.add('interactive');
document.querySelectorAll('[data-widget]').forEach(widget => {
  const inputs=Array.from(widget.querySelectorAll('input')), body=widget.querySelector('.widget-body');
  function render(){ body.innerHTML=widgets[widget.dataset.widget](...inputs.map(input=>Number(input.value))); }
  inputs.forEach(input=>input.addEventListener('input',render));
});
document.querySelectorAll('.codehilite').forEach(block=>{
  const button=document.createElement('button');
  button.type='button';
  button.className='copy';
  button.textContent='Copy Python';
  button.addEventListener('click',async()=>{
    const code=block.querySelector('pre').textContent;
    try { await navigator.clipboard.writeText(code); button.textContent='Copied'; }
    catch { const range=document.createRange(); range.selectNodeContents(block.querySelector('pre')); const sel=getSelection(); sel.removeAllRanges(); sel.addRange(range); button.textContent='Code selected'; }
    setTimeout(()=>button.textContent='Copy Python',1800);
  });
  block.prepend(button);
});
window.addEventListener('beforeprint',()=>document.querySelectorAll('details.answer').forEach(el=>el.open=true));
}
