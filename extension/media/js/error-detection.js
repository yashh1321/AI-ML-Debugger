/**
 * Error Detection Panel JavaScript
 * Handles UI interactions for error detection and smart alerts
 */

(function() {
    const vscode = acquireVsCodeApi();
    let isMonitoring = false;
    let activeAlerts = [];
    let alertHistory = [];
    let chart = null;

    // Initialize when DOM is loaded
    document.addEventListener('DOMContentLoaded', function() {
        initializeUI();
        setupEventListeners();
        loadConfiguration();
    });

    function initializeUI() {
        // Initialize alert chart
        const canvas = document.getElementById('alertChart');
        if (canvas) {
            const ctx = canvas.getContext('2d');
            initializeChart(ctx);
        }

        // Set initial states
        updateMonitoringStatus('stopped');
        updateAlertCounts();
    }

    function setupEventListeners() {
        // Toggle monitoring button
        const toggleBtn = document.getElementById('toggleMonitoring');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', toggleMonitoring);
        }

        // Clear alerts button
        const clearBtn = document.getElementById('clearAlerts');
        if (clearBtn) {
            clearBtn.addEventListener('click', clearActiveAlerts);
        }

        // Refresh history button
        const refreshBtn = document.getElementById('refreshHistory');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => {
                vscode.postMessage({ command: 'getAlertHistory' });
            });
        }

        // Export history button
        const exportBtn = document.getElementById('exportHistory');
        if (exportBtn) {
            exportBtn.addEventListener('click', exportAlertHistory);
        }

        // Configuration change listeners
        const configInputs = document.querySelectorAll('.config-item input');
        configInputs.forEach(input => {
            input.addEventListener('change', updateConfiguration);
        });
    }

    function toggleMonitoring() {
        const toggleBtn = document.getElementById('toggleMonitoring');

        if (isMonitoring) {
            vscode.postMessage({ command: 'stopMonitoring' });
            toggleBtn.textContent = 'Start Monitoring';
            toggleBtn.classList.remove('danger');
            toggleBtn.classList.add('primary');
            isMonitoring = false;
        } else {
            vscode.postMessage({ command: 'startMonitoring' });
            toggleBtn.textContent = 'Stop Monitoring';
            toggleBtn.classList.remove('primary');
            toggleBtn.classList.add('danger');
            isMonitoring = true;
        }
    }

    function clearActiveAlerts() {
        activeAlerts = [];
        updateActiveAlertsList();
        updateAlertCounts();
    }

    function updateMonitoringStatus(status) {
        const statusElement = document.getElementById('monitoringStatus');
        if (statusElement) {
            statusElement.textContent = status === 'active' ? 'Active' : 'Stopped';
            statusElement.className = `value ${status}`;
        }
        isMonitoring = status === 'active';
    }

    function updateAlertCounts() {
        const totalElement = document.getElementById('totalAlerts');
        const highPriorityElement = document.getElementById('highPriorityAlerts');

        if (totalElement) {
            totalElement.textContent = activeAlerts.length;
        }

        if (highPriorityElement) {
            const highPriorityCount = activeAlerts.filter(alert => alert.level === 'high').length;
            highPriorityElement.textContent = highPriorityCount;
        }
    }

    function updateActiveAlertsList() {
        const container = document.getElementById('active-alerts-list');
        if (!container) return;

        container.innerHTML = '';
        if (activeAlerts.length === 0) {
            const noAlerts = document.createElement('div');
            noAlerts.className = 'no-alerts';
            noAlerts.textContent = 'No active alerts';
            container.appendChild(noAlerts);
            return;
        }

        activeAlerts.forEach(alert => {
            container.appendChild(createAlertElement(alert));
        });

        // Add dismiss handlers
        container.querySelectorAll('.alert-dismiss').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const alertId = parseInt(e.target.dataset.alertId);
                dismissAlert(alertId);
            });
        });
    }

    function createAlertElement(alert) {
        const timeFormatted = new Date(alert.timestamp).toLocaleTimeString();
        const recommendations = alert.combined_recommendations || [];

        const div = document.createElement('div');
        div.className = `alert ${escapeHtml(alert.level)}`;
        div.dataset.alertId = alert.id;

        const header = document.createElement('div');
        header.className = 'alert-header';

        const titleSpan = document.createElement('span');
        titleSpan.className = 'alert-title';
        titleSpan.textContent = getAlertTitle(alert);

        const timeSpan = document.createElement('span');
        timeSpan.className = 'alert-time';
        timeSpan.textContent = timeFormatted;

        const dismissBtn = document.createElement('button');
        dismissBtn.className = 'alert-dismiss';
        dismissBtn.dataset.alertId = alert.id;
        dismissBtn.textContent = '×';

        header.appendChild(titleSpan);
        header.appendChild(timeSpan);
        header.appendChild(dismissBtn);
        div.appendChild(header);

        const contentDiv = document.createElement('div');
        contentDiv.className = 'alert-content';
        contentDiv.textContent = getAlertDescription(alert);
        div.appendChild(contentDiv);

        if (recommendations.length > 0) {
            const recDiv = document.createElement('div');
            recDiv.className = 'recommendations';
            const strong = document.createElement('strong');
            strong.textContent = 'Recommendations:';
            recDiv.appendChild(strong);

            recommendations.forEach(rec => {
                recDiv.appendChild(createRecommendationElement(rec));
            });
            div.appendChild(recDiv);
        }

        return div;
    }

    function getAlertTitle(alert) {
        if (alert.combined_recommendations && alert.combined_recommendations.length > 0) {
            return alert.combined_recommendations[0].title;
        }
        return `${alert.level.toUpperCase()} Alert`;
    }

    function getAlertDescription(alert) {
        if (alert.combined_recommendations && alert.combined_recommendations.length > 0) {
            return alert.combined_recommendations[0].description;
        }
        return 'Training issue detected';
    }

    function createRecommendationElement(recommendation) {
        const div = document.createElement('div');
        div.className = 'recommendation';

        const title = document.createElement('div');
        title.className = 'recommendation-title';
        title.textContent = recommendation.title;
        div.appendChild(title);

        const ul = document.createElement('ul');
        ul.className = 'recommendation-suggestions';
        recommendation.suggestions.forEach(s => {
            const li = document.createElement('li');
            li.textContent = s;
            ul.appendChild(li);
        });
        div.appendChild(ul);

        if (recommendation.code_example) {
            const code = document.createElement('div');
            code.className = 'recommendation-code';
            code.textContent = recommendation.code_example;
            div.appendChild(code);
        }

        return div;
    }

    function dismissAlert(alertId) {
        activeAlerts = activeAlerts.filter(alert => alert.id !== alertId);
        vscode.postMessage({ command: 'dismissAlert', alertId: alertId });
        updateActiveAlertsList();
        updateAlertCounts();
    }

    function renderAlertHistory() {
        const container = document.getElementById('alertHistory');
        if (!container) return;

        container.innerHTML = '';
        if (alertHistory.length === 0) {
            const noHist = document.createElement('div');
            noHist.className = 'no-history';
            noHist.textContent = 'No alert history';
            container.appendChild(noHist);
            return;
        }

        alertHistory.forEach(alert => {
            const timeFormatted = new Date(alert.timestamp).toLocaleString();

            const item = document.createElement('div');
            item.className = 'history-item';

            const summary = document.createElement('div');
            summary.className = 'summary';

            const strong = document.createElement('strong');
            strong.textContent = getAlertTitle(alert);

            const br = document.createElement('br');

            const small = document.createElement('small');
            small.textContent = timeFormatted;

            summary.appendChild(strong);
            summary.appendChild(br);
            summary.appendChild(small);

            const badge = document.createElement('span');
            badge.className = `level-badge ${escapeHtml(alert.level)}`;
            badge.textContent = alert.level.toUpperCase();

            item.appendChild(summary);
            item.appendChild(badge);
            container.appendChild(item);
        });
    }

    function updateRealTimeMetrics(metrics) {
        if (metrics.gradientNorm !== undefined) {
            const element = document.getElementById('gradientNorm');
            if (element) {
                element.textContent = typeof metrics.gradientNorm === 'number'
                    ? metrics.gradientNorm.toFixed(4)
                    : metrics.gradientNorm;
            }
        }

        if (metrics.learningRate !== undefined) {
            const element = document.getElementById('learningRate');
            if (element) {
                element.textContent = typeof metrics.learningRate === 'number'
                    ? metrics.learningRate.toExponential(2)
                    : metrics.learningRate;
            }
        }

        if (metrics.lossTrend !== undefined) {
            const element = document.getElementById('lossTrend');
            if (element) {
                const trend = metrics.lossTrend;
                const trendText = trend > 0.01 ? 'Increasing' :
                                 trend < -0.01 ? 'Decreasing' : 'Stable';
                element.textContent = trendText;
                element.className = `metric-value ${trendText.toLowerCase()}`;
            }
        }
    }

    function initializeChart(ctx) {
        // Simple chart implementation for alert frequency
        chart = {
            ctx: ctx,
            data: [],
            draw: function() {
                if (this.data.length === 0) {
                    ctx.fillStyle = '#666';
                    ctx.font = '14px Arial';
                    ctx.textAlign = 'center';
                    ctx.fillText('No data to display', ctx.canvas.width / 2, ctx.canvas.height / 2);
                    return;
                }

                // Clear canvas
                ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

                // Draw simple line chart
                const padding = 40;
                const width = ctx.canvas.width - 2 * padding;
                const height = ctx.canvas.height - 2 * padding;

                if (this.data.length < 2) return;

                const maxValue = Math.max(...this.data);
                const minValue = Math.min(...this.data);
                const range = maxValue - minValue || 1;

                ctx.strokeStyle = '#007acc';
                ctx.lineWidth = 2;
                ctx.beginPath();

                this.data.forEach((value, index) => {
                    const x = padding + (index / (this.data.length - 1)) * width;
                    const y = padding + height - ((value - minValue) / range) * height;

                    if (index === 0) {
                        ctx.moveTo(x, y);
                    } else {
                        ctx.lineTo(x, y);
                    }
                });

                ctx.stroke();
            }
        };
    }

    function updateConfiguration() {
        const config = {
            gradientThreshold: parseFloat(document.getElementById('gradientThreshold').value),
            vanishingThreshold: parseFloat(document.getElementById('vanishingThreshold').value),
            enableLrAnalysis: document.getElementById('enableLrAnalysis').checked,
            enableArchAnalysis: document.getElementById('enableArchAnalysis').checked
        };

        vscode.postMessage({ command: 'updateConfiguration', config: config });
    }

    function loadConfiguration() {
        vscode.postMessage({ command: 'getConfiguration' });
    }

    function exportAlertHistory() {
        const data = {
            exportDate: new Date().toISOString(),
            alertHistory: alertHistory,
            configuration: {
                gradientThreshold: document.getElementById('gradientThreshold').value,
                vanishingThreshold: document.getElementById('vanishingThreshold').value
            }
        };

        vscode.postMessage({
            command: 'exportHistory',
            data: data
        });
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Handle messages from the extension
    window.addEventListener('message', event => {
        const message = event.data;

        switch (message.command) {
            case 'updateMonitoringStatus':
                updateMonitoringStatus(message.status);
                break;

            case 'newAlert':
                activeAlerts.push(message.alert);
                updateActiveAlertsList();
                updateAlertCounts();

                // Update chart data
                if (chart) {
                    chart.data.push(activeAlerts.length);
                    if (chart.data.length > 20) {
                        chart.data = chart.data.slice(-20);
                    }
                    chart.draw();
                }
                break;

            case 'alertHistory':
                alertHistory = message.history;
                renderAlertHistory();
                break;

            case 'alertDismissed':
                // Alert already removed from activeAlerts
                break;

            case 'loadConfiguration':
                const config = message.config;
                if (config) {
                    document.getElementById('gradientThreshold').value = config.gradientThreshold;
                    document.getElementById('vanishingThreshold').value = config.vanishingThreshold;
                    document.getElementById('enableLrAnalysis').checked = config.learningRateAnalysis;
                    document.getElementById('enableArchAnalysis').checked = config.enableSmartAlerts;
                }
                break;

            case 'updateMetrics':
                updateRealTimeMetrics(message.metrics);
                break;
        }
    });

    // Export functions for testing
    window.ErrorDetectionPanel = {
        toggleMonitoring,
        clearActiveAlerts,
        dismissAlert,
        updateConfiguration,
        updateRealTimeMetrics
    };
})();