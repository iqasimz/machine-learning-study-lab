'use strict';

const colors = { blue: '#1769aa', gray: '#66839b', teal: '#087f6b', pale: '#e0edf9' };
const path = (d, color, dash = '') =>
  '<path d="' + d + '" fill="none" stroke="' + color + '" stroke-width="2.2" ' + dash + '/>';
const point = (x, y, color, r = 4) =>
  '<circle cx="' + x + '" cy="' + y + '" r="' + r + '" fill="' + color + '"/>';
const label = (x, y, value) =>
  '<text x="' + x + '" y="' + y + '" fill="#253f5e" font-size="11">' + value + '</text>';

function curve(f, px, py, start, end) {
  return Array.from({length: 101}, (_, index) => {
    const x = start + index * (end - start) / 100;
    return (index ? 'L' : 'M') + px(x).toFixed(2) + ' ' + py(f(x)).toFixed(2);
  }).join(' ');
}

function graph(svg, { xmin, xmax, ymin, ymax, fn, reference, dots = [], names }) {
  const px = x => 39 + 360 * (x - xmin) / (xmax - xmin);
  const py = y => 117 - 104 * (y - ymin) / (ymax - ymin);
  const axes = path('M39 13 V117 H399', '#8aa0b8')
    + label(7, 18, names[1]) + label(389, 141, names[0])
    + label(38, 134, xmin) + label(367, 134, xmax);
  svg.innerHTML = axes
    + (reference ? path(curve(reference, px, py, xmin, xmax), colors.gray, 'stroke-dasharray="5 4"') : '')
    + path(curve(fn, px, py, xmin, xmax), colors.blue)
    + dots.map(([x,y,c]) => point(px(x), py(y), c ?? colors.teal)).join('');
}

const gaussianCDF = z => {
  const x = Math.abs(z) / Math.SQRT2;
  const t = 1 / (1 + 0.3275911 * x);
  const poly = (((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t;
  const erf = Math.sign(z) * (1 - poly * Math.exp(-x*x));
  return 0.5 * (1 + erf);
};
const activations = {
  Tanh: Math.tanh,
  Sigmoid: z => 1 / (1 + Math.exp(-z)),
  ReLU: z => Math.max(0, z),
  'Leaky ReLU': z => Math.max(z, .01*z),
  GELU: z => z * gaussianCDF(z),
};
const lr = s => s <= 10 ? .0001 + .0009 * s / 10
  : .00001 + .00099 * (1 + Math.cos(Math.PI * (s-10)/90)) / 2;
const heads = {
  Binary: 'One logit · sigmoid for probability · BCEWithLogitsLoss for training',
  Multiclass: 'One logit per class · softmax across classes · CrossEntropyLoss',
  Multilabel: 'One logit per label · sigmoid per label · BCEWithLogitsLoss',
};
const sample = [[0.5,0.52],[1.1,0.78],[1.7,1.4],[2.3,0.52],[2.9,0.65],[3.5,-0.41],[4.1,-0.57],[4.7,-1.37],[5.3,-0.56],[5.9,-0.49]];
const fits = {
  Underfit: x => .54 - .18*x,
  'Good fit': x => .95*Math.sin(x),
  Overfit: x => {
    let y = Math.sin(x);
    for (const [sx, sy] of sample) y += (sy-Math.sin(sx))*Math.exp(-((x-sx)**2)/.016);
    return y;
  }
};

for (const lab of document.querySelectorAll('[data-lab]')) {
  const controls = lab.querySelector('.controls');
  const result = lab.querySelector('output');
  const svg = lab.querySelector('svg');
  const get = name => controls.querySelector('[name="' + name + '"]').value;
  const num = name => Number(get(name));
  let randomDraws = [.8,.1,.9,.3];

  function render() {
    const key = lab.dataset.lab;
    if (key === 'batch' || key === 'descent') {
      const batch = key === 'batch';
      const old = batch ? .5 : 0;
      const next = old + (batch ? 1 : 6) * num('eta');
      const loss = batch ? w => ((w-2)**2 + w*w)/2 : w => (w-3)**2;
      result.value = 'η = ' + num('eta').toFixed(2) + ' · w: ' + old.toFixed(2)
        + ' → ' + next.toFixed(2) + ' · loss: ' + loss(old).toFixed(3) + ' → ' + loss(next).toFixed(3);
      graph(svg, {xmin: batch ? 0 : -1, xmax: batch ? 2 : 8, ymin: 0, ymax: batch ? 2.2 : 26,
        fn:loss, dots:[[old,loss(old),colors.gray],[next,loss(next)]],names:['w','loss']});
    } else if (key === 'schedule') {
      const step = num('step');
      result.value = 'Step ' + step + ' / 100 · learning rate ' + lr(step).toFixed(6);
      graph(svg, {xmin:0,xmax:100,ymin:0,ymax:.0011,fn:lr,dots:[[step,lr(step)]],names:['step','LR']});
    } else if (key === 'activation') {
      const kind = get('kind');
      const z = num('z');
      const fn = activations[kind];
      result.value = kind + '(' + z.toFixed(1) + ') = ' + fn(z).toFixed(3);
      graph(svg,{xmin:-4,xmax:4,ymin:-1.2,ymax:4.2,fn,dots:[[z,fn(z)]],names:['z','a']});
    } else if (key === 'bce') {
      const y = get('target') === 'Spam (1)' ? 1 : 0;
      const p = num('prob');
      const correct = y ? p : 1-p;
      result.value = 'P(spam) = ' + p.toFixed(2) + ' · P(true class) = ' + correct.toFixed(2)
        + ' · BCE = ' + (-Math.log(correct)).toFixed(3);
    } else if (key === 'head') {
      result.value = heads[get('task')];
    } else if (key === 'capacity') {
      const kind = get('kind');
      result.value = kind;
      graph(svg,{xmin:0,xmax:6.4,ymin:-1.6,ymax:1.6,fn:fits[kind],reference:Math.sin,dots:sample,names:['x','y']});
    } else if (key === 'norm') {
      const isBatch = get('kind') === 'BatchNorm';
      result.value = isBatch ? 'Column 2 · [10, 20, 30]' : 'Row 2 · [4, 20, 6, 12]';
      svg.innerHTML = [[2,10,3,8],[4,20,6,12],[6,30,9,16]].flatMap((row,r) =>
        row.map((value,c) => {
          const x = 74 + c * 69, y = 14 + r * 40;
          const active = isBatch ? c === 1 : r === 1;
          return '<rect x="' + x + '" y="' + y + '" width="64" height="35" rx="5" fill="'
            + (active ? '#a4d4f2' : colors.pale) + '"/>' + label(x+22,y+22,value);
        })).join('');
    } else if (key === 'dropout') {
      const p = num('p');
      const values = [.8,-.4,.6,1];
      const mask = randomDraws.map(draw => Number(draw >= p));
      result.value = 'p = ' + p.toFixed(2) + ' · mask [' + mask.join(', ') + '] · outputs ['
        + values.map((v,i) => (mask[i] ? v/(1-p) : 0).toFixed(2)).join(', ') + ']';
    } else if (key === 'residual') {
      const x = num('x'), f = num('f');
      result.value = 'x = ' + x.toFixed(1) + ' · F(x) = ' + f.toFixed(1) + ' · y = ' + (x+f).toFixed(1);
    } else if (key === 'clip') {
      const cap = num('cap'), norm = Math.min(cap,10);
      const x = 6*norm/10, y = 8*norm/10;
      result.value = 'Limit ' + cap.toFixed(1) + ' · new norm ' + norm.toFixed(1)
        + ' · g = [' + x.toFixed(1) + ', ' + y.toFixed(1) + ']';
      svg.innerHTML = '<circle cx="185" cy="78" r="' + (cap*5.1) + '" fill="none" stroke="#547590" stroke-dasharray="5 4"/>'
        + path('M185 78 L215.6 37.2', colors.gray) + point(215.6,37.2,colors.gray)
        + path('M185 78 L'+(185+x*5.1)+' '+(78-y*5.1),colors.blue) + point(185+x*5.1,78-y*5.1,colors.blue)
        + label(254,43,'Original [6, 8]') + label(254,63,'Clipped vector in blue');
    }
    if (svg) svg.setAttribute('aria-label', result.value);
  }

  controls.addEventListener('input', render);
  controls.querySelector('[data-sample]')?.addEventListener('click', () => {
    randomDraws = Array.from({length:4}, () => Math.random());
    render();
  });
  render();
}
