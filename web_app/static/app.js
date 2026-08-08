/**
 * Stock Price Prediction using 3-Layer LSTM - Frontend JavaScript Engine
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const stockSelector = document.getElementById("stockSelector");
    const epochInput = document.getElementById("epochInput");
    const runTrainBtn = document.getElementById("runTrainBtn");
    const dbStatusBadge = document.getElementById("dbTypeName");
    const loadingOverlay = document.getElementById("loadingOverlay");

    // KPI Elements
    const scaledRmse = document.getElementById("scaledRmse");
    const actualRmse = document.getElementById("actualRmse");
    const r2Score = document.getElementById("r2Score");
    const dirAccuracy = document.getElementById("dirAccuracy");

    // Forecast Elements
    const forecastPrice = document.getElementById("forecastPrice");
    const lastPrice = document.getElementById("lastPrice");
    const expectedReturn = document.getElementById("expectedReturn");
    const signalBadge = document.getElementById("signalBadge");
    const signalText = document.getElementById("signalText");
    const signalConfidence = document.getElementById("signalConfidence");
    const chartTitle = document.getElementById("chartTitle");
    const assetSubtitle = document.getElementById("assetSubtitle");

    let stockChart = null;

    // Initialize Chart.js Context
    const ctx = document.getElementById("stockChart").getContext("2d");

    /**
     * Initializes or updates Chart.js instance with smooth gradients and glass styling.
     */
    function renderChart(dates, actualPrices, predictedPrices, assetLabel) {
        if (stockChart) {
            stockChart.destroy();
        }

        // Create Gradients
        const gradientActual = ctx.createLinearGradient(0, 0, 0, 400);
        gradientActual.addColorStop(0, "rgba(56, 239, 125, 0.35)");
        gradientActual.addColorStop(1, "rgba(56, 239, 125, 0.0)");

        const gradientPred = ctx.createLinearGradient(0, 0, 0, 400);
        gradientPred.addColorStop(0, "rgba(0, 242, 254, 0.35)");
        gradientPred.addColorStop(1, "rgba(0, 242, 254, 0.0)");

        stockChart = new Chart(ctx, {
            type: "line",
            data: {
                labels: dates,
                datasets: [
                    {
                        label: "Actual Market Price (INR)",
                        data: actualPrices,
                        borderColor: "#38ef7d",
                        backgroundColor: gradientActual,
                        borderWidth: 2,
                        fill: true,
                        tension: 0.2,
                        pointRadius: 0,
                        pointHoverRadius: 6
                    },
                    {
                        label: "3-Layer LSTM Prediction",
                        data: predictedPrices,
                        borderColor: "#00f2fe",
                        borderDash: [5, 5],
                        backgroundColor: gradientPred,
                        borderWidth: 2,
                        fill: false,
                        tension: 0.2,
                        pointRadius: 0,
                        pointHoverRadius: 6
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: "index",
                    intersect: false
                },
                plugins: {
                    legend: {
                        display: false // Using custom legend in header
                    },
                    tooltip: {
                        backgroundColor: "rgba(11, 15, 25, 0.9)",
                        titleFont: { family: "'Outfit', sans-serif", size: 14 },
                        bodyFont: { family: "'JetBrains Mono', monospace", size: 13 },
                        borderColor: "rgba(255,255,255,0.1)",
                        borderWidth: 1,
                        padding: 12,
                        displayColors: true
                    }
                },
                scales: {
                    x: {
                        grid: { color: "rgba(255, 255, 255, 0.04)" },
                        ticks: { color: "#94a3b8", maxTicksLimit: 10 }
                    },
                    y: {
                        grid: { color: "rgba(255, 255, 255, 0.04)" },
                        ticks: {
                            color: "#94a3b8",
                            callback: (val) => "₹" + val.toLocaleString()
                        }
                    }
                }
            }
        });
    }

    /**
     * Fetches stock price prediction payload from FastAPI backend.
     */
    async function loadPredictionData(ticker, epochs = 20, forceRetrain = false) {
        loadingOverlay.classList.add("active");
        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ ticker: ticker, epochs: epochs, force_retrain: forceRetrain })
            });

            const data = await response.json();

            if (data.status !== "success") {
                alert("Error: " + data.message);
                return;
            }

            // Update Database Engine Badge
            dbStatusBadge.textContent = `${data.db_type} Engine`;

            // Update KPI Cards
            scaledRmse.textContent = data.metrics.scaled_rmse;
            actualRmse.textContent = `₹${data.metrics.actual_rmse}`;
            r2Score.textContent = data.metrics.r2_score;
            dirAccuracy.textContent = `${data.metrics.directional_accuracy}%`;

            // Update Forecast Banner
            const forecast = data.forecast;
            forecastPrice.textContent = `₹${forecast.forecasted_next_price.toLocaleString()}`;
            lastPrice.textContent = `₹${forecast.last_actual_price.toLocaleString()}`;
            
            const isPos = forecast.expected_change_percent >= 0;
            expectedReturn.textContent = `${isPos ? '+' : ''}${forecast.expected_change_percent}%`;
            expectedReturn.className = `f-val ${isPos ? 'positive' : 'negative'}`;

            signalText.textContent = forecast.trading_signal;
            signalConfidence.textContent = forecast.signal_confidence;

            if (forecast.trading_signal.includes("BUY")) {
                signalBadge.style.background = "rgba(56, 239, 125, 0.15)";
                signalBadge.style.borderColor = "rgba(56, 239, 125, 0.4)";
                signalBadge.style.color = "#38ef7d";
            } else if (forecast.trading_signal.includes("SELL")) {
                signalBadge.style.background = "rgba(255, 65, 108, 0.15)";
                signalBadge.style.borderColor = "rgba(255, 65, 108, 0.4)";
                signalBadge.style.color = "#ff416c";
            } else {
                signalBadge.style.background = "rgba(79, 172, 254, 0.15)";
                signalBadge.style.borderColor = "rgba(79, 172, 254, 0.4)";
                signalBadge.style.color = "#4facfe";
            }

            // Update Titles & Chart
            chartTitle.textContent = `${data.company_name} — Actual vs. LSTM Prediction`;
            assetSubtitle.textContent = `${data.ticker} — 60-Day Lookback Horizon (${data.dates.length} Days Evaluation Window)`;

            renderChart(data.dates, data.actual_prices, data.predicted_prices, data.ticker);

        } catch (err) {
            console.error("API Error:", err);
            alert("Failed to load prediction data. Check console for details.");
        } finally {
            loadingOverlay.classList.remove("active");
        }
    }

    // Event Listeners
    runTrainBtn.addEventListener("click", () => {
        const ticker = stockSelector.value;
        const epochs = parseInt(epochInput.value, 10) || 20;
        loadPredictionData(ticker, epochs, true); // Force Retrain on button click
    });

    stockSelector.addEventListener("change", () => {
        const ticker = stockSelector.value;
        loadPredictionData(ticker, 20, false); // Instant load pre-trained model
    });

    // Initial Fast Load for default ticker ^NSEI
    loadPredictionData("^NSEI", 20, false);
});
