/* ===== The Neural Scribe v2.0 — Chart Visualizations ===== */
/* Koine-BERT v0.1 (primary) + Ancient-Greek-BERT (robustness) */

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

  // ─── 1. Authorship Gradient Scatter (Koine-BERT) ──────
  const gradientCtx = document.getElementById('gradientChart');
  if (gradientCtx) {
    // Koine-BERT v0.1 data
    const kTexts = [
      { label: 'Colossians',      x: 40,  y: 0.34  },
      { label: '2 Thessalonians', x: 50,  y: -0.20 },
      { label: 'Ephesians',       x: 60,  y: 0.33  },
      { label: '1 Timothy',       x: 80,  y: 0.74  },
      { label: '2 Timothy',       x: 80,  y: 0.48  },
      { label: 'Titus',           x: 80,  y: 1.25  },
      { label: 'Hebrews',         x: 100, y: 0.79  },
    ];

    // Ancient-Greek-BERT data (robustness)
    const aTexts = [
      { label: 'Colossians',      x: 40,  y: 0.24  },
      { label: '2 Thessalonians', x: 50,  y: -0.22 },
      { label: 'Ephesians',       x: 60,  y: -0.03 },
      { label: '1 Timothy',       x: 80,  y: 0.79  },
      { label: '2 Timothy',       x: 80,  y: 0.11  },
      { label: 'Titus',           x: 80,  y: 1.11  },
      { label: 'Hebrews',         x: 100, y: 1.21  },
    ];

    // Least-squares regression for Koine-BERT
    function lsq(pts) {
      const n = pts.length;
      const sx = pts.reduce((a,t)=>a+t.x,0), sy = pts.reduce((a,t)=>a+t.y,0);
      const mx = sx/n, my = sy/n;
      let num=0, den=0;
      pts.forEach(t => { num += (t.x-mx)*(t.y-my); den += (t.x-mx)**2; });
      const slope = num/den, intercept = my - slope*mx;
      return { slope, intercept };
    }

    const kReg = lsq(kTexts);
    const aReg = lsq(aTexts);

    new Chart(gradientCtx, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: 'Koine-BERT (ρ = 0.778, p = 0.039)',
            data: kTexts.map(t => ({ x: t.x, y: t.y })),
            backgroundColor: kTexts.map(t =>
              t.y > 0.5 ? C.rose : t.y < 0 ? C.green : C.amber
            ),
            borderColor: kTexts.map(t =>
              t.y > 0.5 ? C.rose : t.y < 0 ? C.green : C.amber
            ),
            pointRadius: 9,
            pointHoverRadius: 12,
            pointStyle: 'circle',
          },
          {
            label: 'Koine-BERT trend',
            data: [{x:30, y:kReg.slope*30+kReg.intercept}, {x:105, y:kReg.slope*105+kReg.intercept}],
            type: 'line',
            borderColor: C.accentLight,
            borderWidth: 2,
            pointRadius: 0,
            fill: false,
          },
          {
            label: 'Ancient-Greek-BERT (ρ = 0.704, p = 0.077)',
            data: aTexts.map(t => ({ x: t.x, y: t.y })),
            backgroundColor: 'transparent',
            borderColor: C.amber,
            pointRadius: 7,
            pointHoverRadius: 10,
            pointStyle: 'circle',
            borderWidth: 2,
          },
          {
            label: 'AG-BERT trend',
            data: [{x:30, y:aReg.slope*30+aReg.intercept}, {x:105, y:aReg.slope*105+aReg.intercept}],
            type: 'line',
            borderColor: C.amber,
            borderDash: [8,5],
            borderWidth: 1.5,
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
                if (ctx.datasetIndex === 1 || ctx.datasetIndex === 3) return '';
                const pts = ctx.datasetIndex === 0 ? kTexts : aTexts;
                const t = pts[ctx.dataIndex];
                return `${t.label}: ${t.y}σ at ${t.x}% rejection`;
              }
            }
          },
          annotation: {
            annotations: Object.fromEntries(kTexts.map((t,i) => [
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

  // ─── 2. Statistical Distance Bar Chart (Koine-BERT) ───
  const distCtx = document.getElementById('distanceChart');
  if (distCtx) {
    const labels = ['Paul\n(baseline)', 'Colossians', '2 Thess.', 'Ephesians', '1 Timothy', '2 Timothy', 'Titus', 'Hebrews'];
    const means  = [0.00, 0.34, -0.20, 0.33, 0.74, 0.48, 1.25, 0.79];
    const ciLo   = [0, -0.19, -0.72, -0.08, 0.35, -0.26, 0.20, 0.46];
    const ciHi   = [0, 0.86, 0.32, 0.74, 1.14, 1.21, 2.31, 1.12];

    const barColors = means.map((m,i) => {
      if (i === 0) return C.accentLight;
      if (i === 4) return C.rose;        // 1 Timothy: Significant
      if (i === 6 || i === 7) return C.amber; // Titus, Hebrews: Moderate
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

  // ─── 3. Chunk Distribution Grouped Bar (Koine-BERT) ───
  const distribCtx = document.getElementById('distributionChart');
  if (distribCtx) {
    const distLabels = ['Colossians', '2 Thess.', 'Ephesians', '1 Timothy', '2 Timothy', 'Titus', 'Hebrews'];
    const p75 = [40.0, 20.0, 41.9, 70.0, 33.3, 75.0, 58.5];
    const p90 = [15.0, 10.0, 12.9, 30.0, 26.7, 37.5, 36.9];
    const p95 = [10.0,  0.0,  9.7, 10.0, 20.0, 37.5, 23.1];

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
            max: 85,
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
    // Data from classic_results.csv (representative centroids)
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

  // ─── 5. Dissociation Chart (Koine-BERT) ───────────────
  const dissCtx = document.getElementById('dissociationChart');
  if (dissCtx) {
    const dissLabels = ['Colossians', '2 Thess.', 'Ephesians', '1 Timothy', '2 Timothy', 'Titus', 'Hebrews'];
    // Koine-BERT neural distances (σ)
    const neuralDist = [0.34, -0.20, 0.33, 0.74, 0.48, 1.25, 0.79];
    // Classic PCA distance ~ euclidean from Paul centroid (normalised 0-1)
    const classicSim = [0.95, 0.97, 0.96, 0.95, 0.96, 0.94, 0.82];

    new Chart(dissCtx, {
      type: 'bar',
      data: {
        labels: dissLabels,
        datasets: [
          {
            label: 'Neural Distance (σ) — Koine-BERT',
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

  // ─── 6. Neural PCA Embedding scatter (Koine-BERT) ─────
  const embCtx = document.getElementById('embeddingChart');
  if (embCtx) {
    // Sampled from Koine-BERT embeddings PCA (v2.0)
    const paulPts = [
      {x:2.56,y:1.84},{x:-1.15,y:0.97},{x:-1.69,y:-0.71},{x:-0.81,y:2.25},{x:-1.15,y:2.42},
      {x:-1.61,y:0.67},{x:-0.98,y:0.08},{x:0.25,y:0.85},{x:-1.35,y:0.38},{x:-1.39,y:-0.34},
      {x:2.08,y:0.96},{x:-2.34,y:-0.31},{x:1.89,y:0.24},{x:1.44,y:-0.12},{x:-0.0,y:0.31},
      {x:-2.05,y:-0.43},{x:0.37,y:-1.07},{x:-1.54,y:-0.8},{x:-1.02,y:-0.36},{x:-0.95,y:-1.13},
      {x:-0.61,y:-0.63},{x:-0.22,y:-2.09},{x:-2.53,y:-0.44},{x:-1.41,y:-2.31},{x:0.53,y:-1.51},
      {x:-2.79,y:0.76},{x:2.61,y:-0.58},{x:1.75,y:-1.65},{x:-0.26,y:0.84},{x:1.56,y:-0.15},
      {x:2.16,y:-0.26},{x:1.35,y:1.02},{x:0.64,y:-0.33},{x:0.02,y:-1.6},{x:0.21,y:-2.3},
      {x:0.1,y:-1.14},{x:-0.69,y:0.14},{x:0.73,y:-2.05},{x:-0.34,y:-0.82},{x:1.51,y:-0.25},
      {x:1.62,y:-1.2},{x:1.39,y:0.72},{x:1.42,y:0.2},{x:3.22,y:0.5},{x:2.22,y:-1.21},
    ];
    const hebPts = [
      {x:-0.84,y:2.37},{x:-0.85,y:1.16},{x:-0.04,y:1.12},{x:-0.07,y:-0.4},{x:-1.28,y:0.44},
      {x:-0.26,y:1.8},{x:0.11,y:2.41},{x:-0.52,y:2.43},{x:-2.17,y:1.55},{x:-1.06,y:2.49},
      {x:-1.09,y:-0.04},{x:-1.55,y:2.43},{x:-1.62,y:2.46},{x:-1.73,y:0.46},{x:0.51,y:1.39},
      {x:-0.22,y:0.17},{x:-1.11,y:1.63},{x:-2.09,y:0.72},{x:0.57,y:1.04},{x:-0.73,y:0.12},
      {x:0.13,y:-0.52},{x:1.24,y:0.96},
    ];
    const colPts = [
      {x:3.79,y:1.5},{x:2.93,y:2.48},{x:1.5,y:3.98},{x:-0.32,y:3.26},{x:0.82,y:1.41},
      {x:1.36,y:1.27},{x:1.68,y:0.89},{x:2.29,y:-0.15},{x:1.33,y:1.0},{x:0.46,y:1.81},
      {x:-0.11,y:1.18},{x:-0.28,y:0.59},{x:-0.02,y:0.05},{x:0.62,y:0.11},{x:2.06,y:0.64},
      {x:2.41,y:-0.24},{x:1.53,y:-0.71},{x:1.99,y:-0.4},{x:2.14,y:-0.25},{x:1.83,y:-0.14},
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
            title: { display: true, text: 'PC1 (10.86%)', font: { weight: '500' } },
            grid: gridOpts,
          },
          y: {
            title: { display: true, text: 'PC2 (9.00%)', font: { weight: '500' } },
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
