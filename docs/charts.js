/* ===== The Neural Scribe — Chart Visualizations ===== */
/* All data values from results/neural_results.json & classic_results.csv */

(function() {
  'use strict';

  // ─── Colour palette ───────────────────────────────────
  const C = {
    accent:      '#2b5797',
    accentLight: '#3a6fb5',
    accentUltra: '#5a8ec9',
    brown:       '#7a4e2d',
    brownLight:  '#a0714d',
    green:       '#3a7d4e',
    amber:       '#b8860b',
    rose:        '#a62c2c',
    cyan:        '#2b5797',
    muted:       '#9a876e',
    text:        '#2c1e0e',
    grid:        'rgba(214,201,182,.5)',
    gridSoft:    'rgba(214,201,182,.3)',
  };

  // ─── Shared defaults ──────────────────────────────────
  Chart.defaults.color = C.text;
  Chart.defaults.font.family = "'Times New Roman', Times, Georgia, serif";
  Chart.defaults.font.size   = 12;
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
  Chart.defaults.plugins.legend.labels.padding = 16;

  const gridOpts = {
    color: C.gridSoft,
    drawBorder: false,
  };

  // ─── 1. Authorship Gradient Scatter ──────────────────
  const gradientCtx = document.getElementById('gradientChart');
  if (gradientCtx) {
    // Real data: rejection rate (%) vs mean sigma
    const texts = [
      { label: 'Colossians',      x: 40,  y: 0.24  },
      { label: '2 Thessalonians', x: 50,  y: -0.22 },
      { label: 'Ephesians',       x: 60,  y: -0.03 },
      { label: '1 Timothy',       x: 80,  y: 0.79  },
      { label: '2 Timothy',       x: 80,  y: 0.11  },
      { label: 'Titus',           x: 80,  y: 1.11  },
      { label: 'Hebrews',         x: 100, y: 1.21  },
    ];

    // Least-squares regression
    const n = texts.length;
    const sx = texts.reduce((a,t)=>a+t.x,0), sy = texts.reduce((a,t)=>a+t.y,0);
    const mx = sx/n, my = sy/n;
    let num=0, den=0;
    texts.forEach(t => { num += (t.x-mx)*(t.y-my); den += (t.x-mx)**2; });
    const slope = num/den, intercept = my - slope*mx;

    new Chart(gradientCtx, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: 'Disputed texts',
            data: texts.map(t => ({ x: t.x, y: t.y })),
            backgroundColor: texts.map(t =>
              t.y > 0.5 ? C.rose : t.y < 0 ? C.green : C.amber
            ),
            borderColor: texts.map(t =>
              t.y > 0.5 ? C.rose : t.y < 0 ? C.green : C.amber
            ),
            pointRadius: 9,
            pointHoverRadius: 12,
            pointStyle: 'circle',
          },
          {
            label: 'Trend (ρ = 0.704)',
            data: [{x:30, y:slope*30+intercept}, {x:105, y:slope*105+intercept}],
            type: 'line',
            borderColor: C.accentLight,
            borderDash: [8,5],
            borderWidth: 2,
            pointRadius: 0,
            fill: false,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        aspectRatio: 1.8,
        scales: {
          x: {
            title: { display: true, text: 'Scholarly Rejection Rate (%)', font: { weight: '500' } },
            min: 25, max: 110,
            grid: gridOpts,
          },
          y: {
            title: { display: true, text: 'Neural Distance from Paul (σ)', font: { weight: '500' } },
            min: -0.6, max: 1.6,
            grid: gridOpts,
          }
        },
        plugins: {
          legend: { display: true, position: 'top' },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                if (ctx.datasetIndex === 1) return '';
                const t = texts[ctx.dataIndex];
                return `${t.label}: ${t.y}σ at ${t.x}% rejection`;
              }
            }
          },
          annotation: {
            annotations: Object.fromEntries(texts.map((t,i) => [
              'label'+i,
              {
                type: 'label',
                xValue: t.x,
                yValue: t.y,
                content: t.label,
                font: { size: 10, weight: '500' },
                color: C.text,
                position: 'start',
                yAdjust: -16,
                xAdjust: t.label === 'Hebrews' ? -10 : t.label === '2 Timothy' ? 25 : t.label === '1 Timothy' ? -15 : 0,
              }
            ]))
          }
        }
      }
    });
  }

  // ─── 2. Statistical Distance Bar Chart ────────────────
  const distCtx = document.getElementById('distanceChart');
  if (distCtx) {
    const labels = ['Paul\n(baseline)', 'Colossians', '2 Thess.', 'Ephesians', '1 Timothy', '2 Timothy', 'Titus', 'Hebrews'];
    const means  = [0.00, 0.24, -0.22, -0.03, 0.79, 0.11, 1.11, 1.21];
    const ciLo   = [0, -0.20, -0.70, -0.39, 0.39, -0.61, -0.10, 0.75];
    const ciHi   = [0, 0.69, 0.26, 0.33, 1.20, 0.83, 2.32, 1.67];

    const barColors = means.map((m,i) => {
      if (i === 0) return C.accentLight;
      if (i === 4 || i === 7) return C.rose; // significant
      if (i === 6) return C.amber; // marginal
      return C.muted;
    });

    new Chart(distCtx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Distance from Paul (σ)',
          data: means,
          backgroundColor: barColors.map(c => c + '88'),
          borderColor: barColors,
          borderWidth: 2,
          borderRadius: 6,
          errorBars: true,
        }]
      },
      plugins: [{
        id: 'errorBars',
        afterDraw(chart) {
          const { ctx: c, scales: { x, y } } = chart;
          const meta = chart.getDatasetMeta(0);
          meta.data.forEach((bar, i) => {
            if (i === 0) return; // skip baseline
            const xPx = bar.x;
            const yLo = y.getPixelForValue(ciLo[i]);
            const yHi = y.getPixelForValue(ciHi[i]);
            c.save();
            c.strokeStyle = barColors[i];
            c.lineWidth = 1.5;
            c.beginPath();
            c.moveTo(xPx, yLo); c.lineTo(xPx, yHi);
            c.moveTo(xPx-5, yLo); c.lineTo(xPx+5, yLo);
            c.moveTo(xPx-5, yHi); c.lineTo(xPx+5, yHi);
            c.stroke();
            c.restore();
          });
        }
      }],
      options: {
        responsive: true,
        maintainAspectRatio: true,
        aspectRatio: 2.2,
        scales: {
          x: { grid: { display: false } },
          y: {
            title: { display: true, text: 'Distance (σ)', font: { weight: '500' } },
            grid: gridOpts,
          }
        },
        plugins: {
          legend: { display: false },
          annotation: {
            annotations: {
              sigLine: {
                type: 'line',
                yMin: 0, yMax: 0,
                borderColor: C.accentLight,
                borderDash: [4,4],
                borderWidth: 1,
                label: {
                  display: true,
                  content: 'Pauline baseline',
                  position: 'start',
                  font: { size: 10 },
                  color: C.accentUltra,
                  backgroundColor: 'transparent',
                }
              }
            }
          },
          tooltip: {
            callbacks: {
              afterLabel: (ctx) => {
                const i = ctx.dataIndex;
                if (i===0) return 'Baseline';
                return `95% CI: [${ciLo[i].toFixed(2)}, ${ciHi[i].toFixed(2)}]`;
              }
            }
          }
        }
      }
    });
  }

  // ─── 3. Chunk Distribution Grouped Bar ────────────────
  const distribCtx = document.getElementById('distributionChart');
  if (distribCtx) {
    const distLabels = ['Colossians', '2 Thess.', 'Ephesians', '1 Timothy', '2 Timothy', 'Titus', 'Hebrews'];
    const p75 = [35.0, 20.0, 16.1, 65.0, 33.3, 50.0, 58.5];
    const p90 = [10.0,  0.0,  9.7, 20.0, 20.0, 37.5, 43.1];
    const p95 = [10.0,  0.0,  9.7, 10.0, 20.0, 37.5, 38.5];

    new Chart(distribCtx, {
      type: 'bar',
      data: {
        labels: distLabels,
        datasets: [
          {
            label: '> P75',
            data: p75,
            backgroundColor: C.accent + 'aa',
            borderColor: C.accent,
            borderWidth: 1,
            borderRadius: 4,
          },
          {
            label: '> P90',
            data: p90,
            backgroundColor: C.amber + 'aa',
            borderColor: C.amber,
            borderWidth: 1,
            borderRadius: 4,
          },
          {
            label: '> P95',
            data: p95,
            backgroundColor: C.rose + 'aa',
            borderColor: C.rose,
            borderWidth: 1,
            borderRadius: 4,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        aspectRatio: 2,
        scales: {
          x: { grid: { display: false } },
          y: {
            title: { display: true, text: '% of Chunks Exceeding Threshold', font: { weight: '500' } },
            max: 75,
            grid: gridOpts,
          }
        },
        plugins: {
          legend: { position: 'top' },
          annotation: {
            annotations: {
              p75line: { type: 'line', yMin: 25, yMax: 25, borderColor: C.accent + '60', borderDash: [6,4], borderWidth: 1,
                label: { display: true, content: 'Expected P75 (25%)', position: 'end', font: { size: 9 }, color: C.accent, backgroundColor: 'transparent' }
              },
              p90line: { type: 'line', yMin: 10, yMax: 10, borderColor: C.amber + '60', borderDash: [6,4], borderWidth: 1,
                label: { display: true, content: 'Expected P90 (10%)', position: 'end', font: { size: 9 }, color: C.amber, backgroundColor: 'transparent' }
              },
              p95line: { type: 'line', yMin: 5, yMax: 5, borderColor: C.rose + '60', borderDash: [6,4], borderWidth: 1,
                label: { display: true, content: 'Expected P95 (5%)', position: 'end', font: { size: 9 }, color: C.rose, backgroundColor: 'transparent' }
              },
            }
          }
        }
      }
    });
  }

  // ─── 4. Classic PCA Scatter ───────────────────────────
  const pcaCtx = document.getElementById('pcaChart');
  if (pcaCtx) {
    // Data from classic_results.csv
    const pcaPaul = [
      {x:-0.0058, y:-0.0087}, {x:-0.0109, y:-0.0098},
      {x:0.0049, y:-0.0105}, {x:-0.0153, y:-0.0071},
    ];
    const pcaControl = [
      {x:-0.0077, y:0.0133}, {x:-0.0002, y:0.0012}, {x:-0.0060, y:0.0213},
    ];
    const pcaTarget = [{x:0.0409, y:0.0004}];

    new Chart(pcaCtx, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: 'Paul (undisputed)',
            data: pcaPaul,
            backgroundColor: C.accent + 'cc',
            borderColor: C.accent,
            pointRadius: 10,
            pointHoverRadius: 13,
          },
          {
            label: 'Control (Heb, 1Pe, Acts)',
            data: pcaControl,
            backgroundColor: C.rose + 'cc',
            borderColor: C.rose,
            pointRadius: 10,
            pointHoverRadius: 13,
            pointStyle: 'triangle',
          },
          {
            label: 'Target (Colossians)',
            data: pcaTarget,
            backgroundColor: C.amber + 'cc',
            borderColor: C.amber,
            pointRadius: 12,
            pointHoverRadius: 15,
            pointStyle: 'rectRot',
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        aspectRatio: 1.6,
        scales: {
          x: {
            title: { display: true, text: 'PC1 (4.98% variance)', font: { weight: '500' } },
            grid: gridOpts,
          },
          y: {
            title: { display: true, text: 'PC2 (3.72% variance)', font: { weight: '500' } },
            grid: gridOpts,
          }
        },
        plugins: {
          legend: { position: 'top' },
          annotation: {
            annotations: {
              cluster: {
                type: 'box',
                xMin: -0.02, xMax: 0.01,
                yMin: -0.015, yMax: 0.005,
                backgroundColor: 'rgba(99,102,241,.06)',
                borderColor: 'rgba(99,102,241,.2)',
                borderWidth: 1,
                borderDash: [4,4],
                label: {
                  display: true,
                  content: 'Shared lexical cluster',
                  position: 'start',
                  font: { size: 10 },
                  color: C.text + '80',
                }
              }
            }
          }
        }
      }
    });
  }

  // ─── 5. Dissociation Chart ────────────────────────────
  const dissCtx = document.getElementById('dissociationChart');
  if (dissCtx) {
    const dissLabels = ['Colossians', '2 Thess.', 'Ephesians', '1 Timothy', '2 Timothy', 'Titus', 'Hebrews'];
    // Neural distances (σ)
    const neuralDist = [0.24, -0.22, -0.03, 0.79, 0.11, 1.11, 1.21];
    // Classic PCA distance ~ euclidean from Paul centroid (normalised 0-1 for comparison)
    // Computed from classic_results.csv: all within-cluster except Hebrews slight offset
    const classicSim = [0.95, 0.97, 0.96, 0.95, 0.96, 0.94, 0.82];

    new Chart(dissCtx, {
      type: 'bar',
      data: {
        labels: dissLabels,
        datasets: [
          {
            label: 'Neural Distance (σ)',
            data: neuralDist,
            backgroundColor: C.accent + 'aa',
            borderColor: C.accent,
            borderWidth: 2,
            borderRadius: 5,
            yAxisID: 'y',
          },
          {
            label: 'Classic Similarity',
            data: classicSim,
            backgroundColor: C.amber + '77',
            borderColor: C.amber,
            borderWidth: 2,
            borderRadius: 5,
            yAxisID: 'y1',
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        aspectRatio: 2,
        scales: {
          x: { grid: { display: false } },
          y: {
            title: { display: true, text: 'Neural Distance (σ)', font: { weight: '500' } },
            position: 'left',
            grid: gridOpts,
          },
          y1: {
            title: { display: true, text: 'Classic Similarity', font: { weight: '500' } },
            position: 'right',
            min: 0.6, max: 1.05,
            grid: { display: false },
          }
        },
        plugins: {
          legend: { position: 'top' },
        }
      }
    });
  }

  // ─── 6. Neural PCA Embedding scatter ──────────────────
  const embCtx = document.getElementById('embeddingChart');
  if (embCtx) {
    // Sampled from neural_results.csv (representative subsample for performance)
    const paulPts = [
      {x:2.83,y:2.66},{x:4.98,y:0.56},{x:4.77,y:-0.09},{x:3.35,y:-1.22},{x:3.22,y:0.18},
      {x:4.81,y:0.13},{x:3.90,y:-0.87},{x:4.48,y:-3.19},{x:3.59,y:1.04},{x:4.90,y:-1.06},
      {x:5.38,y:-0.74},{x:5.85,y:-0.93},{x:4.40,y:1.45},{x:3.69,y:1.27},{x:4.96,y:0.05},
      {x:5.00,y:-0.67},{x:5.74,y:0.36},{x:5.54,y:-0.70},{x:3.93,y:0.24},{x:4.53,y:-0.96},
      {x:5.37,y:-0.03},{x:3.34,y:1.26},{x:5.12,y:-0.40},{x:5.06,y:1.04},{x:4.30,y:-0.74},
      {x:5.31,y:0.30},{x:4.74,y:0.95},{x:3.09,y:0.07},{x:5.28,y:0.76},{x:3.92,y:-0.35},
      {x:4.57,y:0.71},{x:3.87,y:-0.66},{x:3.66,y:-0.09},{x:4.34,y:-2.27},{x:4.67,y:0.62},
      {x:2.74,y:-1.72},{x:0.40,y:-0.55},{x:3.80,y:-0.12},{x:3.93,y:0.47},{x:6.18,y:-0.55},
      {x:3.73,y:-1.24},{x:6.59,y:-1.60},{x:3.15,y:-0.72},{x:4.54,y:1.93},{x:4.89,y:-1.95},
    ];
    const hebPts = [
      {x:-2.29,y:-0.09},{x:-4.31,y:-0.57},{x:-2.89,y:-0.60},{x:-1.56,y:-0.78},{x:-2.13,y:-0.04},
      {x:-3.51,y:-1.25},{x:-2.12,y:0.03},{x:-3.73,y:-0.16},{x:-3.78,y:-1.16},{x:-3.84,y:-1.40},
      {x:-2.92,y:-0.36},{x:-2.46,y:-1.29},{x:-4.52,y:-1.36},{x:-3.92,y:-0.62},{x:-4.20,y:0.38},
      {x:-3.43,y:-1.88},{x:-4.44,y:0.43},{x:-4.18,y:-1.01},{x:-3.02,y:0.30},{x:-2.94,y:0.30},
    ];
    const colPts = [
      {x:0.21,y:9.67},{x:1.61,y:11.30},{x:0.60,y:10.63},{x:1.19,y:11.09},{x:0.20,y:9.42},
      {x:-0.63,y:9.69},{x:0.69,y:11.72},{x:-0.38,y:10.12},{x:-1.40,y:7.30},{x:-0.79,y:9.57},
      {x:0.62,y:11.25},
    ];

    new Chart(embCtx, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: 'Paul (undisputed)',
            data: paulPts,
            backgroundColor: C.accent + '55',
            borderColor: C.accent + '80',
            pointRadius: 4,
            pointHoverRadius: 6,
          },
          {
            label: 'Hebrews',
            data: hebPts,
            backgroundColor: C.rose + '77',
            borderColor: C.rose,
            pointRadius: 5,
            pointHoverRadius: 7,
            pointStyle: 'triangle',
          },
          {
            label: 'Colossians',
            data: colPts,
            backgroundColor: C.amber + '77',
            borderColor: C.amber,
            pointRadius: 5,
            pointHoverRadius: 7,
            pointStyle: 'rectRot',
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        aspectRatio: 1.6,
        scales: {
          x: {
            title: { display: true, text: 'PC1', font: { weight: '500' } },
            grid: gridOpts,
          },
          y: {
            title: { display: true, text: 'PC2', font: { weight: '500' } },
            grid: gridOpts,
          }
        },
        plugins: {
          legend: { position: 'top' },
        }
      }
    });
  }


  // ─── Scroll-triggered fade-in ─────────────────────────
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.classList.add('visible');
        observer.unobserve(e.target);
      }
    });
  }, { threshold: 0.12, rootMargin: '0px 0px -40px 0px' });

  document.querySelectorAll('.fade-in').forEach(el => observer.observe(el));

  // ─── Active nav highlight ─────────────────────────────
  const navLinks = document.querySelectorAll('.nav-bar a');
  const sections = document.querySelectorAll('section[id]');
  const headerId = document.getElementById('top');

  function updateActiveNav() {
    let current = '';
    sections.forEach(s => {
      if (window.scrollY >= s.offsetTop - 120) current = s.id;
    });
    navLinks.forEach(a => {
      a.classList.toggle('active', a.getAttribute('href') === '#' + current);
    });
  }
  window.addEventListener('scroll', updateActiveNav, { passive: true });
  updateActiveNav();

})();
