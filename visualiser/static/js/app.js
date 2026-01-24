/**
 * RL Trade Visualizer - Frontend Application
 *
 * Handles WebSocket communication and UI updates.
 */

// ============================================================================
// Global State
// ============================================================================

let ws;
let isPlaying = false;
let currentTab = 'snapshot';

const actionNames = ['HOLD', 'EXIT', 'TIGHTEN_SL', 'TRAIL_BREAKEVEN', 'PARTIAL_EXIT'];
const actionColors = ['#8b949e', '#f85149', '#d29922', '#58a6ff', '#a371f7'];

// ============================================================================
// Panel Toggle System
// ============================================================================

function togglePanel(panelId) {
    const panel = document.getElementById(panelId);
    if (!panel) return;

    panel.classList.toggle('collapsed');

    // Persist state to localStorage
    const states = JSON.parse(localStorage.getItem('panelStates') || '{}');
    states[panelId] = panel.classList.contains('collapsed');
    localStorage.setItem('panelStates', JSON.stringify(states));
}

function restorePanelStates() {
    const states = JSON.parse(localStorage.getItem('panelStates') || '{}');
    Object.keys(states).forEach(panelId => {
        const panel = document.getElementById(panelId);
        if (panel && states[panelId]) {
            panel.classList.add('collapsed');
        }
    });
}

// ============================================================================
// Tab System
// ============================================================================

function switchTab(tabName) {
    currentTab = tabName;

    // Update tab buttons
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.tab === tabName);
    });

    // Update tab content visibility
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.toggle('hidden', content.id !== 'tab-' + tabName);
    });

    // Trigger resize for Plotly charts in newly visible tab
    setTimeout(() => {
        if (tabName === 'timeline') {
            Plotly.Plots.resize('timeline-chart');
        } else if (tabName === 'correlations') {
            Plotly.Plots.resize('correlations-chart');
        } else if (tabName === 'snapshot') {
            Plotly.Plots.resize('activations-chart');
        }
    }, 50);
}

// ============================================================================
// Chart Initialization
// ============================================================================

function initCharts() {
    // Candlestick chart
    Plotly.newPlot('candlestick-chart', [{
        type: 'candlestick',
        x: [],
        open: [],
        high: [],
        low: [],
        close: [],
        increasing: {line: {color: '#3fb950', width: 1}, fillcolor: '#238636'},
        decreasing: {line: {color: '#f85149', width: 1}, fillcolor: '#da3633'},
        name: 'AUDUSD'
    }], {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(13,17,23,0.5)',
        font: {color: '#8b949e', family: 'SF Pro Display'},
        xaxis: {
            gridcolor: 'rgba(48,54,61,0.5)',
            linecolor: '#30363d',
            rangeslider: {visible: false},
            tickfont: {size: 10}
        },
        yaxis: {
            gridcolor: 'rgba(48,54,61,0.5)',
            linecolor: '#30363d',
            side: 'right',
            tickfont: {size: 10},
            tickformat: '.5f'
        },
        margin: {l: 10, r: 70, t: 10, b: 40},
        showlegend: false
    }, {responsive: true});

    // Activations heatmap (snapshot view)
    Plotly.newPlot('activations-chart', [{
        type: 'heatmap',
        z: [new Array(32).fill(0), new Array(32).fill(0), new Array(32).fill(0), new Array(32).fill(0)],
        colorscale: [
            [0, '#0d1117'],
            [0.3, '#1f6feb'],
            [0.6, '#a371f7'],
            [1, '#f85149']
        ],
        showscale: false
    }], {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: {l: 5, r: 5, t: 5, b: 5},
        xaxis: {visible: false},
        yaxis: {visible: false}
    }, {responsive: true});

    // Timeline heatmap (activation evolution during trade)
    Plotly.newPlot('timeline-chart', [{
        type: 'heatmap',
        z: [[0]],
        colorscale: [
            [0, '#0d1117'],
            [0.3, '#1f6feb'],
            [0.6, '#a371f7'],
            [1, '#f85149']
        ],
        showscale: false
    }], {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(13,17,23,0.5)',
        margin: {l: 35, r: 5, t: 5, b: 25},
        xaxis: {
            title: 'Neuron',
            titlefont: {size: 9, color: '#6e7681'},
            tickfont: {size: 8, color: '#6e7681'},
            gridcolor: 'rgba(48,54,61,0.3)'
        },
        yaxis: {
            title: 'Bar',
            titlefont: {size: 9, color: '#6e7681'},
            tickfont: {size: 8, color: '#6e7681'},
            gridcolor: 'rgba(48,54,61,0.3)'
        }
    }, {responsive: true});

    // Correlations chart (neuron-action relationships)
    Plotly.newPlot('correlations-chart', [], {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(13,17,23,0.5)',
        margin: {l: 70, r: 10, t: 10, b: 30},
        barmode: 'group',
        xaxis: {
            title: 'Mean Activation',
            titlefont: {size: 9, color: '#6e7681'},
            tickfont: {size: 8, color: '#6e7681'},
            gridcolor: 'rgba(48,54,61,0.3)'
        },
        yaxis: {
            title: '',
            tickfont: {size: 9, color: '#8b949e'}
        },
        showlegend: false
    }, {responsive: true});
}

// ============================================================================
// WebSocket Connection
// ============================================================================

function connectWebSocket() {
    ws = new WebSocket('ws://localhost:8765/ws');

    ws.onopen = () => {
        document.getElementById('connection-status').className = 'status-dot connected';
        document.getElementById('connection-text').textContent = 'Connected';
    };

    ws.onclose = () => {
        document.getElementById('connection-status').className = 'status-dot disconnected';
        document.getElementById('connection-text').textContent = 'Disconnected';
        setTimeout(connectWebSocket, 2000);
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        updateVisualization(data);
    };
}

// ============================================================================
// Visualization Update
// ============================================================================

function updateVisualization(data) {
    // Update timestamp
    document.getElementById('timestamp').textContent = data.timestamp;

    // Update candlestick chart
    if (data.candles) {
        updateCandlestickChart(data);
    }

    // Update current trade panel
    updateCurrentTradePanel(data.current_trade);

    // Update action display
    if (data.action !== undefined) {
        updateActionDisplay(data);
    }

    // Update neural activations (snapshot)
    if (data.activations) {
        updateActivationsChart(data.activations);
    }

    // Update activation timeline
    if (data.activation_timeline && data.activation_timeline.length > 0) {
        updateTimelineChart(data.activation_timeline);
    }

    // Update neuron-action correlations
    if (data.action_correlations) {
        updateCorrelationsChart(data.action_correlations);
    }

    // Update metrics panel
    if (data.metrics) {
        updateMetricsPanel(data.metrics, data.stats);
    }

    // Update trade log
    if (data.trade_history) {
        updateTradeLog(data.trade_history);
    }
}

function updateCandlestickChart(data) {
    let traces = [{
        type: 'candlestick',
        x: data.candles.timestamps,
        open: data.candles.open,
        high: data.candles.high,
        low: data.candles.low,
        close: data.candles.close,
        increasing: {line: {color: '#3fb950', width: 1}, fillcolor: '#238636'},
        decreasing: {line: {color: '#f85149', width: 1}, fillcolor: '#da3633'}
    }];

    // Entry markers
    if (data.entries && data.entries.length > 0) {
        traces.push({
            type: 'scatter',
            mode: 'markers+text',
            x: data.entries.map(e => e.time),
            y: data.entries.map(e => e.price),
            marker: {
                size: 20,
                color: data.entries.map(e => e.direction === 1 ? '#3fb950' : '#f85149'),
                symbol: data.entries.map(e => e.direction === 1 ? 'triangle-up' : 'triangle-down'),
                line: {color: '#ffffff', width: 2}
            },
            text: data.entries.map(e => e.direction === 1 ? 'LONG' : 'SHORT'),
            textposition: data.entries.map(e => e.direction === 1 ? 'top center' : 'bottom center'),
            textfont: {
                size: 10,
                color: data.entries.map(e => e.direction === 1 ? '#3fb950' : '#f85149'),
                family: 'SF Mono, monospace',
                weight: 700
            },
            name: 'Entry'
        });
    }

    // Exit markers (final trade exits at actual price)
    if (data.exits && data.exits.length > 0) {
        traces.push({
            type: 'scatter',
            mode: 'markers+text',
            x: data.exits.map(e => e.time),
            y: data.exits.map(e => e.price),
            marker: {
                size: 18,
                color: data.exits.map(e => e.pnl >= 0 ? '#3fb950' : '#f85149'),
                symbol: 'x',
                line: {color: '#ffffff', width: 2}
            },
            text: data.exits.map(e => {
                const sign = e.pnl >= 0 ? '+' : '';
                const pips = e.pnl.toFixed(1);
                const dollars = e.pnl_dollars ? (e.pnl_dollars >= 0 ? '+$' : '-$') + Math.abs(e.pnl_dollars).toFixed(0) : '';
                return sign + pips + 'p ' + dollars;
            }),
            textposition: data.exits.map(e => e.direction === 1 ? 'bottom center' : 'top center'),
            textfont: {
                size: 10,
                color: data.exits.map(e => e.pnl >= 0 ? '#3fb950' : '#f85149'),
                family: 'SF Mono, monospace',
                weight: 700
            },
            name: 'Exit',
            hovertemplate: '%{text}<extra>EXIT</extra>'
        });
    }

    // RL Action markers (PARTIAL, TIGHTEN_SL, TRAIL_BE) at actual price levels
    if (data.action_markers && data.action_markers.length > 0) {
        // Group by action type
        const partials = data.action_markers.filter(m => m.action === 'PARTIAL');
        const tightens = data.action_markers.filter(m => m.action === 'TIGHTEN_SL');
        const trails = data.action_markers.filter(m => m.action === 'TRAIL_BE');
        const exits = data.action_markers.filter(m => m.action === 'EXIT');

        // PARTIAL markers (purple diamonds)
        if (partials.length > 0) {
            traces.push({
                type: 'scatter',
                mode: 'markers+text',
                x: partials.map(p => p.time),
                y: partials.map(p => p.price),
                marker: {
                    size: 14,
                    color: '#a371f7',
                    symbol: 'diamond',
                    line: {color: '#ffffff', width: 1}
                },
                text: partials.map(p => {
                    const sign = p.pnl_pips >= 0 ? '+' : '';
                    return sign + p.pnl_pips.toFixed(1) + 'p';
                }),
                textposition: partials.map(p => p.direction === 1 ? 'top center' : 'bottom center'),
                textfont: {size: 9, color: '#a371f7', family: 'SF Mono, monospace'},
                name: 'Partial',
                hovertemplate: 'PARTIAL<br>%{y:.5f}<br>%{text}<extra></extra>'
            });
        }

        // TIGHTEN_SL markers (orange square)
        if (tightens.length > 0) {
            traces.push({
                type: 'scatter',
                mode: 'markers',
                x: tightens.map(t => t.time),
                y: tightens.map(t => t.price),
                marker: {
                    size: 10,
                    color: '#d29922',
                    symbol: 'square',
                    line: {color: '#ffffff', width: 1}
                },
                name: 'Tighten SL',
                hovertemplate: 'TIGHTEN SL<br>%{y:.5f}<extra></extra>'
            });
        }

        // TRAIL_BE markers (blue circle)
        if (trails.length > 0) {
            traces.push({
                type: 'scatter',
                mode: 'markers',
                x: trails.map(t => t.time),
                y: trails.map(t => t.price),
                marker: {
                    size: 10,
                    color: '#58a6ff',
                    symbol: 'circle',
                    line: {color: '#ffffff', width: 1}
                },
                name: 'Trail BE',
                hovertemplate: 'TRAIL BE<br>%{y:.5f}<extra></extra>'
            });
        }

        // EXIT markers from action_markers (red X) - at actual price
        if (exits.length > 0) {
            traces.push({
                type: 'scatter',
                mode: 'markers+text',
                x: exits.map(e => e.time),
                y: exits.map(e => e.price),
                marker: {
                    size: 16,
                    color: exits.map(e => e.pnl_pips >= 0 ? '#3fb950' : '#f85149'),
                    symbol: 'x-thin',
                    line: {width: 3, color: exits.map(e => e.pnl_pips >= 0 ? '#3fb950' : '#f85149')}
                },
                text: exits.map(e => {
                    const sign = e.pnl_dollars >= 0 ? '+$' : '-$';
                    return sign + Math.abs(e.pnl_dollars).toFixed(0);
                }),
                textposition: exits.map(e => e.direction === 1 ? 'bottom center' : 'top center'),
                textfont: {
                    size: 10,
                    color: exits.map(e => e.pnl_pips >= 0 ? '#3fb950' : '#f85149'),
                    family: 'SF Mono, monospace'
                },
                name: 'RL Exit',
                hovertemplate: 'RL EXIT<br>%{y:.5f}<br>%{text}<extra></extra>'
            });
        }
    }

    let layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(13,17,23,0.5)',
        font: {color: '#8b949e', family: 'SF Pro Display'},
        xaxis: {
            gridcolor: 'rgba(48,54,61,0.5)',
            linecolor: '#30363d',
            rangeslider: {visible: false},
            tickfont: {size: 10}
        },
        yaxis: {
            gridcolor: 'rgba(48,54,61,0.5)',
            linecolor: '#30363d',
            side: 'right',
            tickfont: {size: 10},
            tickformat: '.5f'
        },
        margin: {l: 10, r: 70, t: 30, b: 40},
        showlegend: false,
        shapes: [],
        annotations: []
    };

    // Add SL/TP zone for active trade
    if (data.trade_zone) {
        const tz = data.trade_zone;
        layout.shapes.push(
            {type: 'rect', xref: 'x', yref: 'y', x0: tz.x0, x1: tz.x1, y0: tz.sl, y1: tz.tp,
             fillcolor: 'rgba(88,166,255,0.08)', line: {width: 0}},
            {type: 'line', xref: 'x', yref: 'y', x0: tz.x0, x1: tz.x1, y0: tz.original_sl, y1: tz.original_sl,
             line: {color: '#6e7681', width: 1, dash: 'dot'}},
            {type: 'line', xref: 'x', yref: 'y', x0: tz.x0, x1: tz.x1, y0: tz.sl, y1: tz.sl,
             line: {color: '#f85149', width: 2, dash: tz.sl_changed ? 'solid' : 'dash'}},
            {type: 'line', xref: 'x', yref: 'y', x0: tz.x0, x1: tz.x1, y0: tz.tp, y1: tz.tp,
             line: {color: '#3fb950', width: 2, dash: 'dash'}},
            {type: 'line', xref: 'x', yref: 'y', x0: tz.x0, x1: tz.x1, y0: tz.entry_price, y1: tz.entry_price,
             line: {color: '#58a6ff', width: 1, dash: 'dot'}}
        );
        layout.annotations.push(
            {x: tz.x1, y: tz.tp, text: 'TP', showarrow: false, font: {color: '#3fb950', size: 10}, xanchor: 'left'},
            {x: tz.x1, y: tz.sl, text: tz.sl_changed ? 'SL (modified)' : 'SL', showarrow: false,
             font: {color: '#f85149', size: 10}, xanchor: 'left'}
        );
    }

    Plotly.react('candlestick-chart', traces, layout);
}

function updateCurrentTradePanel(trade) {
    const container = document.getElementById('current-trade-container');
    if (!trade) {
        container.innerHTML = '<div class="no-trade">No active trade</div>';
        return;
    }

    const t = trade;
    const dirClass = t.direction === 1 ? 'long' : 'short';
    const dirText = t.direction === 1 ? 'LONG' : 'SHORT';
    const pnlColor = t.pnl_pips >= 0 ? '#3fb950' : '#f85149';

    const ac = t.action_counts || {};
    const actionParts = [];
    if (ac.HOLD > 0) actionParts.push(`H:${ac.HOLD}`);
    if (ac.TIGHTEN_SL > 0) actionParts.push(`<span style="color:#d29922">T:${ac.TIGHTEN_SL}</span>`);
    if (ac.TRAIL_BE > 0) actionParts.push(`<span style="color:#58a6ff">BE:${ac.TRAIL_BE}</span>`);
    if (ac.EXIT > 0) actionParts.push(`<span style="color:#f85149">X:${ac.EXIT}</span>`);
    if (ac.PARTIAL > 0) actionParts.push(`<span style="color:#a371f7">P:${ac.PARTIAL}</span>`);
    const actionSummary = actionParts.join(' ') || 'None';

    const slTightened = t.sl_tightened ? '<span style="color:#d29922;">[SL TIGHTENED]</span>' : '';
    const trailedBE = t.trailed_to_be ? '<span style="color:#58a6ff;">[TRAILING BE]</span>' : '';

    const posSize = t.position_size !== undefined ? t.position_size : 1.0;
    const posSizePercent = (posSize * 100).toFixed(0);
    const posValueM = t.position_value_m !== undefined ? t.position_value_m.toFixed(1) : '2.0';
    const posColor = posSize < 1.0 ? '#a371f7' : '#3fb950';
    const partialInfo = t.partial_exits > 0 ? `<span style="color:#a371f7;">(${t.partial_exits} partial${t.partial_exits > 1 ? 's' : ''})</span>` : '';

    const realizedPnl = t.realized_pnl !== undefined ? t.realized_pnl : 0;
    const unrealizedPnl = t.unrealized_pnl !== undefined ? t.unrealized_pnl : t.pnl_dollars;

    container.innerHTML = `
        <div class="current-trade">
            <div class="current-trade-header">
                <span class="current-trade-direction ${dirClass}">${dirText}</span>
                <span style="color:${posColor}; font-size:12px; font-weight:600;">${posValueM}M (${posSizePercent}%)</span>
            </div>
            <div class="current-trade-pnl" style="color:${pnlColor}">
                ${t.pnl_pips >= 0 ? '+' : ''}${t.pnl_pips.toFixed(1)} pips
            </div>
            <div style="color:${pnlColor}; font-size:16px; font-weight:600; font-family:'SF Mono',monospace;">
                ${t.pnl_dollars >= 0 ? '+' : ''}$${t.pnl_dollars.toFixed(0)} ${partialInfo}
            </div>
            ${realizedPnl !== 0 ? `
            <div style="display:flex; justify-content:space-between; font-size:10px; margin-top:4px;">
                <span style="color:#a371f7;">Realized: $${realizedPnl.toFixed(0)}</span>
                <span style="color:#8b949e;">Unrealized: $${unrealizedPnl.toFixed(0)}</span>
            </div>` : ''}
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:10px;">
                <div class="feature">
                    <div class="feature-name">Entry</div>
                    <div class="feature-value neutral">${t.entry_price.toFixed(5)}</div>
                </div>
                <div class="feature">
                    <div class="feature-name">Current</div>
                    <div class="feature-value neutral">${t.current_price.toFixed(5)}</div>
                </div>
                <div class="feature">
                    <div class="feature-name">MFE</div>
                    <div class="feature-value positive">+${t.mfe.toFixed(1)} pips</div>
                </div>
                <div class="feature">
                    <div class="feature-name">MAE</div>
                    <div class="feature-value negative">${t.mae.toFixed(1)} pips</div>
                </div>
            </div>
            <div style="margin-top:10px; padding-top:8px; border-top:1px solid #30363d;">
                <div style="display:flex; justify-content:space-between; font-size:11px; color:#8b949e;">
                    <span>SL: ${t.current_sl ? t.current_sl.toFixed(5) : 'N/A'}</span>
                    <span>${slTightened}${trailedBE}</span>
                </div>
                <div style="margin-top:6px; font-size:11px;">
                    <span style="color:#8b949e;">Actions:</span> ${actionSummary}
                </div>
            </div>
        </div>
    `;
}

function updateActionDisplay(data) {
    const actionName = actionNames[data.action];
    const actionEl = document.getElementById('action-name');
    actionEl.textContent = actionName;
    actionEl.className = 'action-name action-' + actionName;

    for (let i = 0; i < 5; i++) {
        const prob = data.probs[i] * 100;
        document.getElementById('prob-' + i).style.width = prob + '%';
        document.getElementById('prob-val-' + i).textContent = prob.toFixed(1) + '%';
    }

    if (data.model_output) {
        document.getElementById('model-value').textContent = data.model_output.value.toFixed(3);
        document.getElementById('model-entropy').textContent = data.model_output.entropy.toFixed(3);
        document.getElementById('model-confidence').textContent = (data.model_output.confidence * 100).toFixed(0) + '%';
    }
}

function updateActivationsChart(activations) {
    Plotly.react('activations-chart', [{
        type: 'heatmap',
        z: activations,
        colorscale: [[0, '#0d1117'], [0.3, '#1f6feb'], [0.6, '#a371f7'], [1, '#f85149']],
        showscale: false
    }], {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: {l: 5, r: 5, t: 5, b: 5},
        xaxis: {visible: false},
        yaxis: {visible: false}
    });
}

function updateTimelineChart(timeline) {
    // timeline = [{bar: 0, activations: [128 floats], action: 0}, ...]
    if (!timeline || timeline.length === 0) return;

    // Build 2D array: rows=bars, cols=neurons
    const z = timeline.map(t => t.activations);

    // Color-code by action taken at each bar
    const actionMarkers = timeline.map(t => actionNames[t.action] || 'HOLD');

    Plotly.react('timeline-chart', [{
        type: 'heatmap',
        z: z,
        colorscale: [[0, '#0d1117'], [0.3, '#1f6feb'], [0.6, '#a371f7'], [1, '#f85149']],
        showscale: false,
        hovertemplate: 'Bar %{y}<br>Neuron %{x}<br>Activation: %{z:.3f}<extra></extra>'
    }], {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(13,17,23,0.5)',
        margin: {l: 35, r: 5, t: 5, b: 25},
        xaxis: {
            title: 'Neuron',
            titlefont: {size: 9, color: '#6e7681'},
            tickfont: {size: 8, color: '#6e7681'},
            gridcolor: 'rgba(48,54,61,0.3)'
        },
        yaxis: {
            title: 'Bar',
            titlefont: {size: 9, color: '#6e7681'},
            tickfont: {size: 8, color: '#6e7681'},
            gridcolor: 'rgba(48,54,61,0.3)',
            autorange: 'reversed'
        }
    });
}

function updateCorrelationsChart(correlations) {
    // correlations = {HOLD: {top_neurons: [...], mean_activation: float, sample_count: int}, ...}
    if (!correlations) return;

    const actions = ['HOLD', 'EXIT', 'TIGHTEN_SL', 'TRAIL_BE', 'PARTIAL'];
    const colors = ['#8b949e', '#f85149', '#d29922', '#58a6ff', '#a371f7'];

    const traces = [];
    const yLabels = [];
    const xValues = [];
    const barColors = [];

    actions.forEach((action, idx) => {
        const data = correlations[action];
        if (data && data.sample_count > 0) {
            const label = `${action} (n=${data.sample_count})`;
            yLabels.push(label);
            xValues.push(data.mean_activation);
            barColors.push(colors[idx]);
        }
    });

    if (yLabels.length === 0) {
        // No data yet - show placeholder
        Plotly.react('correlations-chart', [], {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(13,17,23,0.5)',
            margin: {l: 70, r: 10, t: 10, b: 30},
            annotations: [{
                text: 'Collecting data...',
                xref: 'paper',
                yref: 'paper',
                x: 0.5,
                y: 0.5,
                showarrow: false,
                font: {size: 12, color: '#6e7681'}
            }]
        });
        return;
    }

    Plotly.react('correlations-chart', [{
        type: 'bar',
        orientation: 'h',
        y: yLabels,
        x: xValues,
        marker: {
            color: barColors,
            line: {color: 'rgba(255,255,255,0.1)', width: 1}
        },
        hovertemplate: '%{y}<br>Mean: %{x:.4f}<extra></extra>'
    }], {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(13,17,23,0.5)',
        margin: {l: 110, r: 10, t: 10, b: 30},
        xaxis: {
            title: 'Mean Activation',
            titlefont: {size: 9, color: '#6e7681'},
            tickfont: {size: 8, color: '#6e7681'},
            gridcolor: 'rgba(48,54,61,0.3)'
        },
        yaxis: {
            title: '',
            tickfont: {size: 9, color: '#8b949e'},
            automargin: true
        },
        showlegend: false
    });
}

function updateMetricsPanel(m, s) {
    document.getElementById('metric-sharpe').textContent = m.sharpe_ratio.toFixed(2);
    document.getElementById('metric-winrate').textContent = m.win_rate + '%';

    const pnlEl = document.getElementById('metric-pnl');
    pnlEl.textContent = (s.total_pnl >= 0 ? '+$' : '-$') + Math.abs(s.total_pnl).toFixed(0);
    pnlEl.className = 'metric-value ' + (s.total_pnl >= 0 ? 'positive' : 'negative');

    const retEl = document.getElementById('metric-return');
    retEl.textContent = (m.return_pct >= 0 ? '+' : '') + m.return_pct.toFixed(3) + '%';
    retEl.className = 'metric-value ' + (m.return_pct >= 0 ? 'positive' : 'negative');

    document.getElementById('metric-trades').textContent = s.trades;
    document.getElementById('metric-pf').textContent = m.profit_factor.toFixed(2);
    document.getElementById('metric-avgwin').textContent = '+$' + m.avg_win.toFixed(0);
    document.getElementById('metric-avgloss').textContent = '-$' + m.avg_loss.toFixed(0);

    document.getElementById('metric-longpct').textContent = m.long_pct + '%';
    document.getElementById('metric-shortpct').textContent = m.short_pct + '%';
    document.getElementById('metric-longwr').textContent = m.long_win_rate + '%';
    document.getElementById('metric-shortwr').textContent = m.short_win_rate + '%';

    document.getElementById('metric-avgdur').textContent = m.avg_duration.toFixed(1);
    document.getElementById('metric-maxdd').textContent = '-$' + m.max_drawdown.toFixed(0);
    document.getElementById('metric-maxwin').textContent = '+$' + m.max_win.toFixed(0);
    document.getElementById('metric-maxloss').textContent = '-$' + Math.abs(m.max_loss).toFixed(0);

    document.getElementById('metric-rlexit').textContent = m.rl_exit_pct + '%';
    document.getElementById('metric-slhit').textContent = m.sl_hit_pct + '%';
    document.getElementById('metric-partials').textContent = m.partial_per_trade.toFixed(2);

    const pipsEl = document.getElementById('metric-pips');
    pipsEl.textContent = (s.total_pips >= 0 ? '+' : '') + s.total_pips.toFixed(1);
    pipsEl.className = 'metric-value ' + (s.total_pips >= 0 ? 'positive' : 'negative');
}

function updateTradeLog(tradeHistory) {
    const logBody = document.getElementById('trade-log-body');

    logBody.innerHTML = tradeHistory.map((t, idx) => {
        const isActive = t.is_active === true;
        const dirClass = t.direction === 1 ? 'long' : 'short';
        const dirText = t.direction === 1 ? '🟢 LONG' : '🔴 SHORT';
        const pnlClass = t.pnl_dollars >= 0 ? 'positive' : 'negative';

        const entryTime = t.entry_time_pretty || t.entry_time || '-';
        const entryPrice = t.entry_price ? t.entry_price.toFixed(5) : '-';
        const positionM = isActive ? (t.position_size_pct / 100 * 2).toFixed(1) : '2.0';

        const slPrice = t.sl_price ? t.sl_price.toFixed(5) : '-';
        const tpPrice = t.tp_price ? t.tp_price.toFixed(5) : '-';
        const slChanged = t.sl_tightened || t.trailed_to_be;

        const pnlSign = t.pnl_dollars >= 0 ? '+' : '';
        const pipsSign = t.pnl_pips >= 0 ? '+' : '';

        const actions = t.action_history || [];
        let actionsHtml = '';

        if (actions.length === 0) {
            actionsHtml = '<div class="no-actions">No significant actions yet...</div>';
        } else {
            actionsHtml = actions.map(a => {
                const actionType = a.action.toLowerCase().replace('_', '');
                const actionClass = actionType === 'tighten_sl' ? 'tighten' :
                                   actionType === 'trail_be' ? 'trail' : actionType;
                const aPnlSign = a.pnl_dollars >= 0 ? '+' : '';
                const aPnlClass = a.pnl_dollars >= 0 ? 'positive' : 'negative';

                // For PARTIAL/EXIT, show Buy/Sell based on direction
                // SHORT position: closing = Buy, LONG position: closing = Sell
                let actionDisplayName = a.action;
                if (a.action === 'PARTIAL') {
                    actionDisplayName = t.direction === 1 ? 'PARTIAL (Sell)' : 'PARTIAL (Buy)';
                } else if (a.action === 'EXIT') {
                    actionDisplayName = t.direction === 1 ? 'EXIT (Sell)' : 'EXIT (Buy)';
                }

                let actionContent = '';
                if (a.action === 'PARTIAL' || a.action === 'EXIT') {
                    // position_size_pct and position_m now show the SIZE BEING CLOSED
                    const closeLabel = a.action === 'EXIT' ? 'Close' : 'Close';
                    actionContent = `
                        <div class="action-time">${a.time_pretty || ''}</div>
                        <div class="action-price">@ ${a.price.toFixed(5)}</div>
                        <div class="action-size">${closeLabel} ${a.position_size_pct.toFixed(0)}% (${a.position_m.toFixed(1)}M)</div>
                        <div class="action-pnl ${aPnlClass}">${aPnlSign}${a.pnl_pips.toFixed(1)} pips</div>
                        <div class="action-pnl ${aPnlClass}">${aPnlSign}$${a.pnl_dollars.toFixed(0)}</div>
                    `;
                } else if (a.action === 'TIGHTEN_SL' || a.action === 'TRAIL_BE') {
                    actionContent = `
                        <div class="action-time">${a.time_pretty || ''}</div>
                        <div class="action-price">@ ${a.price.toFixed(5)}</div>
                        <div class="action-note">${a.note || ''}</div>
                        <div class="action-pnl ${aPnlClass}">PnL: ${aPnlSign}$${a.pnl_dollars.toFixed(0)}</div>
                    `;
                } else {
                    actionContent = `
                        <div class="action-time">${a.time_pretty || ''}</div>
                        <div class="action-price">@ ${a.price.toFixed(5)}</div>
                        <div class="action-pnl ${aPnlClass}">PnL: ${aPnlSign}$${a.pnl_dollars.toFixed(0)}</div>
                    `;
                }

                return `
                    <div class="action-item ${actionClass}">
                        <div class="action-header">
                            <span class="action-type ${actionClass}">${actionDisplayName}</span>
                            <span class="action-bar">Bar ${a.bar}</span>
                        </div>
                        ${actionContent}
                    </div>
                `;
            }).join('');

            // Add TOTAL summary for completed trades with PnL breakdown
            if (!isActive && t.pnl_dollars !== undefined) {
                const totalPnl = t.pnl_dollars;
                const totalClass = totalPnl >= 0 ? 'positive' : 'negative';
                const totalSign = totalPnl >= 0 ? '+' : '';
                const totalPipsVal = t.pnl_pips || 0;
                const totalPipsSign = totalPipsVal >= 0 ? '+' : '';
                const totalItemClass = totalPnl >= 0 ? 'total-win' : 'total-loss';

                // Build PnL breakdown from actions (partials and exits)
                let breakdownHtml = '';
                const exitActions = actions.filter(a => a.action === 'PARTIAL' || a.action === 'EXIT');
                if (exitActions.length > 0) {
                    const breakdownItems = exitActions.map(a => {
                        const sign = a.pnl_dollars >= 0 ? '+' : '';
                        const pSign = a.pnl_pips >= 0 ? '+' : '';
                        const valClass = a.pnl_dollars >= 0 ? 'positive' : 'negative';
                        return `<div class="pnl-breakdown-item">
                            <span class="label">${a.position_m.toFixed(1)}M × ${pSign}${a.pnl_pips.toFixed(1)}p</span>
                            <span class="value ${valClass}">${sign}$${a.pnl_dollars.toFixed(0)}</span>
                        </div>`;
                    }).join('');

                    breakdownHtml = `<div class="pnl-breakdown">${breakdownItems}</div>`;
                }

                actionsHtml += `
                    <div class="action-item ${totalItemClass}">
                        <div class="action-header">
                            <span class="action-type" style="color: ${totalPnl >= 0 ? '#3fb950' : '#f85149'}; font-size: 10px;">TOTAL</span>
                            <span class="action-bar" style="background: ${totalPnl >= 0 ? 'rgba(63, 185, 80, 0.3)' : 'rgba(248, 81, 73, 0.3)'}">${t.exit_reason || 'Closed'}</span>
                        </div>
                        <div style="font-size: 10px; color: #8b949e; margin-bottom: 2px;">Full Trade (2.0M)</div>
                        <div class="action-pnl ${totalClass}" style="font-size: 13px;">${totalPipsSign}${totalPipsVal.toFixed(1)} pips</div>
                        <div class="action-pnl ${totalClass}" style="font-size: 16px; font-weight: 800;">${totalSign}$${Math.abs(totalPnl).toFixed(0)}</div>
                        ${breakdownHtml}
                    </div>
                `;
            }
        }

        return `
            <div class="trade-card ${isActive ? 'active' : ''}">
                <div class="trade-card-header">
                    <div class="trade-entry-info">
                        <span class="trade-direction ${dirClass}">${dirText}</span>
                        <div class="trade-details">
                            <span class="label">Entry:</span>
                            <span class="value">${positionM}M @ ${entryPrice}</span>
                            <span style="color:#6e7681;margin-left:8px;">${entryTime}</span>
                        </div>
                        <div class="trade-levels">
                            <span class="sl" style="${slChanged ? 'font-weight:600;' : ''}">SL: ${slPrice}${slChanged ? ' ✓' : ''}</span>
                            <span class="tp">TP: ${tpPrice}</span>
                        </div>
                    </div>
                    <div class="trade-result">
                        ${isActive ? '<span class="active-badge">⚡ ACTIVE</span>' : ''}
                        <div class="pnl ${pnlClass}">${pnlSign}$${Math.abs(t.pnl_dollars).toFixed(0)}</div>
                        <div class="pips">${pipsSign}${t.pnl_pips.toFixed(1)} pips</div>
                    </div>
                </div>
                <div class="action-timeline">
                    ${actionsHtml}
                </div>
            </div>
        `;
    }).join('');
}

// ============================================================================
// Controls
// ============================================================================

function togglePlay() {
    isPlaying = !isPlaying;
    const btn = document.getElementById('play-btn');
    btn.innerHTML = isPlaying ? '<span>&#10074;&#10074;</span> Pause' : '<span>&#9658;</span> Play';
    ws.send(JSON.stringify({command: isPlaying ? 'play' : 'pause'}));
}

function stepForward() {
    ws.send(JSON.stringify({command: 'step'}));
}

function nextTrade() {
    ws.send(JSON.stringify({command: 'next_trade'}));
}

function resetPlayback() {
    ws.send(JSON.stringify({command: 'reset'}));
    isPlaying = false;
    document.getElementById('play-btn').innerHTML = '<span>&#9658;</span> Play';
}

// Speed slider
document.getElementById('speed-slider').addEventListener('input', (e) => {
    const speed = e.target.value;
    document.getElementById('speed-value').textContent = speed + 'x';
    ws.send(JSON.stringify({command: 'speed', value: parseInt(speed)}));
});

// ============================================================================
// Initialize
// ============================================================================

window.onload = () => {
    initCharts();
    restorePanelStates();
    connectWebSocket();
};
