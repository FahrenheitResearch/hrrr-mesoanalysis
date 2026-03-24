/* ============================================================
   HRRR Mesoanalysis — Frontend Application
   ============================================================ */

(function () {
    "use strict";

    // -----------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------

    const API_BASE = "";  // same origin when served by serve.py
    const CONUS_BOUNDS = [[-125, 24], [-66, 50]]; // [sw, ne] in [lon, lat]
    const DEFAULT_PARAM = "sbcape";

    // Parameter metadata — mirrors mesoanalysis/plotting/styles.py
    // Colors are approximate CSS representations of the matplotlib colormaps.
    const PARAM_META = {
        sbcape: {
            label: "SBCAPE",
            units: "J/kg",
            fullLabel: "SBCAPE (J/kg)",
            levels: [100, 250, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000],
            colors: ["#ffffb2","#ffeda0","#fed976","#feb24c","#fd8d3c","#fc4e2a","#e31a1c","#bd0026","#800026"],
        },
        mlcape: {
            label: "MLCAPE",
            units: "J/kg",
            fullLabel: "MLCAPE (J/kg)",
            levels: [100, 250, 500, 1000, 1500, 2000, 2500, 3000, 4000],
            colors: ["#ffffb2","#ffeda0","#fed976","#feb24c","#fd8d3c","#fc4e2a","#e31a1c","#bd0026"],
        },
        mucape: {
            label: "MUCAPE",
            units: "J/kg",
            fullLabel: "MUCAPE (J/kg)",
            levels: [100, 250, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000],
            colors: ["#ffffb2","#ffeda0","#fed976","#feb24c","#fd8d3c","#fc4e2a","#e31a1c","#bd0026","#800026"],
        },
        sb3cape: {
            label: "SB3CAPE",
            units: "J/kg",
            fullLabel: "SB3CAPE (J/kg)",
            levels: [25, 50, 100, 150, 200, 300, 400, 500],
            colors: ["#ffffb2","#ffeda0","#fed976","#feb24c","#fd8d3c","#fc4e2a","#e31a1c"],
        },
        sbcin: {
            label: "SBCIN",
            units: "J/kg",
            fullLabel: "SBCIN (J/kg)",
            levels: [-300, -250, -200, -150, -100, -75, -50, -25],
            colors: ["#4a1486","#6a51a3","#807dba","#9e9ac8","#bcbddc","#dadaeb","#f2f0f7"],
        },
        theta_e_sfc: {
            label: "Sfc \u03b8e",
            units: "K",
            fullLabel: "Sfc Theta-e (K)",
            levels: levelsFromRange(290, 370, 5),
            colors: generateRdYlBuR(levelsFromRange(290, 370, 5).length - 1),
        },
        theta_e_850: {
            label: "850mb \u03b8e",
            units: "K",
            fullLabel: "850mb Theta-e (K)",
            levels: levelsFromRange(280, 360, 5),
            colors: generateRdYlBuR(levelsFromRange(280, 360, 5).length - 1),
        },
        mixing_ratio: {
            label: "Mixing Ratio",
            units: "g/kg",
            fullLabel: "Mixing Ratio (g/kg)",
            levels: levelsFromRange(2, 24, 2),
            colors: generateGreens(levelsFromRange(2, 24, 2).length - 1),
        },
        pw: {
            label: "Precip Water",
            units: "mm",
            fullLabel: "Precipitable Water (mm)",
            levels: levelsFromRange(5, 60, 5),
            colors: generateGnBu(levelsFromRange(5, 60, 5).length - 1),
        },
        wetbulb_sfc: {
            label: "Wet Bulb",
            units: "\u00b0C",
            fullLabel: "Sfc Wet Bulb (\u00b0C)",
            levels: levelsFromRange(-25, 35, 2),
            colors: generateRdYlBuR(levelsFromRange(-25, 35, 2).length - 1),
        },
        lr_700_500: {
            label: "700-500mb LR",
            units: "\u00b0C/km",
            fullLabel: "700-500mb Lapse Rate (\u00b0C/km)",
            levels: levelsFromRange(4, 10, 0.5),
            colors: generateReds(levelsFromRange(4, 10, 0.5).length - 1),
        },
        shear_01km: {
            label: "0-1km Shear",
            units: "kt",
            fullLabel: "0-1km Shear (kt)",
            levels: [5, 10, 15, 20, 25, 30, 35, 40],
            colors: ["#0d0887","#46039f","#7201a8","#9c179e","#bd3786","#d8576b","#f0f921"],
        },
        shear_06km: {
            label: "0-6km Shear",
            units: "kt",
            fullLabel: "0-6km Shear (kt)",
            levels: [10, 20, 30, 40, 50, 60, 70, 80],
            colors: ["#0d0887","#46039f","#7201a8","#9c179e","#bd3786","#d8576b","#f0f921"],
        },
        srh_01km: {
            label: "0-1km SRH",
            units: "m\u00b2/s\u00b2",
            fullLabel: "0-1km SRH (m\u00b2/s\u00b2)",
            levels: [50, 100, 150, 200, 300, 400, 500],
            colors: ["#feebe2","#fcc5c0","#fa9fb5","#f768a1","#c51b8a","#7a0177"],
        },
        srh_03km: {
            label: "0-3km SRH",
            units: "m\u00b2/s\u00b2",
            fullLabel: "0-3km SRH (m\u00b2/s\u00b2)",
            levels: [100, 200, 300, 400, 500, 600, 800],
            colors: ["#feebe2","#fcc5c0","#fa9fb5","#f768a1","#c51b8a","#7a0177"],
        },
        stp: {
            label: "STP",
            units: "",
            fullLabel: "Sig Tornado Parameter",
            levels: [0.5, 1, 2, 3, 5, 8, 12],
            colors: ["#ffffb2","#fecc5c","#fd8d3c","#f03b20","#bd0026","#67000d"],
        },
        scp: {
            label: "SCP",
            units: "",
            fullLabel: "Supercell Composite",
            levels: [1, 2, 4, 6, 8, 10, 15],
            colors: ["#fff7ec","#fee8c8","#fdd49e","#fdbb84","#e34a33","#b30000"],
        },
        t2m_f: {
            label: "Temperature",
            units: "\u00b0F",
            fullLabel: "Temperature (\u00b0F)",
            levels: levelsFromRange(-20, 115, 5),
            colors: generateRdYlBuR(levelsFromRange(-20, 115, 5).length - 1),
        },
        td2m_f: {
            label: "Dewpoint",
            units: "\u00b0F",
            fullLabel: "Dewpoint (\u00b0F)",
            levels: levelsFromRange(-10, 80, 5),
            colors: generateBrBG(levelsFromRange(-10, 80, 5).length - 1),
        },
    };

    // -----------------------------------------------------------
    // Color scale helpers
    // -----------------------------------------------------------

    function levelsFromRange(lo, hi, step) {
        const arr = [];
        for (let v = lo; v <= hi + step / 4; v += step) {
            arr.push(Math.round(v * 100) / 100);
        }
        return arr;
    }

    function interpolateColors(colors, n) {
        if (n <= 0) return [];
        if (n === 1) return [colors[0]];
        const result = [];
        for (let i = 0; i < n; i++) {
            const t = i / (n - 1);
            const idx = t * (colors.length - 1);
            const lo = Math.floor(idx);
            const hi = Math.min(lo + 1, colors.length - 1);
            const f = idx - lo;
            const c1 = hexToRgb(colors[lo]);
            const c2 = hexToRgb(colors[hi]);
            const r = Math.round(c1.r + (c2.r - c1.r) * f);
            const g = Math.round(c1.g + (c2.g - c1.g) * f);
            const b = Math.round(c1.b + (c2.b - c1.b) * f);
            result.push(rgbToHex(r, g, b));
        }
        return result;
    }

    function hexToRgb(hex) {
        const m = hex.match(/^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i);
        return { r: parseInt(m[1], 16), g: parseInt(m[2], 16), b: parseInt(m[3], 16) };
    }

    function rgbToHex(r, g, b) {
        return "#" + [r, g, b].map(x => x.toString(16).padStart(2, "0")).join("");
    }

    // Approximate matplotlib colormap representations
    function generateRdYlBuR(n) {
        const anchors = ["#313695","#4575b4","#74add1","#abd9e9","#e0f3f8","#ffffbf","#fee090","#fdae61","#f46d43","#d73027","#a50026"];
        return interpolateColors(anchors.slice().reverse(), n);
    }

    function generateGreens(n) {
        return interpolateColors(["#f7fcf5","#c7e9c0","#74c476","#238b45","#00441b"], n);
    }

    function generateGnBu(n) {
        return interpolateColors(["#f7fcf0","#ccebc5","#7bccc4","#2b8cbe","#084081"], n);
    }

    function generateReds(n) {
        return interpolateColors(["#fff5f0","#fcbba1","#fb6a4a","#cb181d","#67000d"], n);
    }

    function generateBrBG(n) {
        return interpolateColors(["#543005","#8c510a","#d8b365","#f6e8c3","#f5f5f5","#c7eae5","#5ab4ac","#01665e","#003c30"], n);
    }

    // -----------------------------------------------------------
    // State
    // -----------------------------------------------------------

    let currentParam = DEFAULT_PARAM;
    let currentRun = null;
    let availableRuns = [];
    let leafletMap = null;
    let imageOverlay = null;
    let viewMode = "image"; // "image" or "map"
    const imageCache = {};

    // -----------------------------------------------------------
    // DOM elements
    // -----------------------------------------------------------

    const $analysisTime = document.getElementById("analysis-time");
    const $paramTitle = document.getElementById("param-title");
    const $opacitySlider = document.getElementById("opacity-slider");
    const $opacityValue = document.getElementById("opacity-value");
    const $btnImageView = document.getElementById("btn-image-view");
    const $btnMapView = document.getElementById("btn-map-view");
    const $imageViewer = document.getElementById("image-viewer");
    const $mapViewer = document.getElementById("map-viewer");
    const $paramImage = document.getElementById("param-image");
    const $imageLoading = document.getElementById("image-loading");
    const $runSelect = document.getElementById("run-select");
    const $legendLabel = document.getElementById("legend-label");
    const $legendColorbar = document.getElementById("legend-colorbar");
    const $legendUnits = document.getElementById("legend-units");
    const $pointPopup = document.getElementById("point-popup");
    const $popupContent = document.getElementById("popup-content");
    const $popupClose = document.getElementById("popup-close");

    // -----------------------------------------------------------
    // Initialization
    // -----------------------------------------------------------

    async function init() {
        setupSidebar();
        setupControls();
        await loadRuns();
        if (currentRun) {
            selectParam(DEFAULT_PARAM);
        }
    }

    // -----------------------------------------------------------
    // API calls
    // -----------------------------------------------------------

    async function loadRuns() {
        try {
            const resp = await fetch(API_BASE + "/api/latest");
            const data = await resp.json();
            availableRuns = data.runs || [];
            currentRun = data.latest || (availableRuns.length > 0 ? availableRuns[0] : null);

            $runSelect.innerHTML = "";
            for (const run of availableRuns) {
                const opt = document.createElement("option");
                opt.value = run;
                // Format: 20260324_0400 -> 2026-03-24 04:00Z
                opt.textContent = formatRunName(run);
                if (run === currentRun) opt.selected = true;
                $runSelect.appendChild(opt);
            }

            if (currentRun) {
                $analysisTime.textContent = formatRunName(currentRun);
            }
        } catch (e) {
            console.error("Failed to load runs:", e);
            $analysisTime.textContent = "No data available";
        }
    }

    function formatRunName(run) {
        // 20260324_0400 -> 2026-03-24 04:00 UTC
        if (!run || run.length < 13) return run;
        return run.slice(0,4) + "-" + run.slice(4,6) + "-" + run.slice(6,8)
            + " " + run.slice(9,11) + ":" + run.slice(11,13) + " UTC";
    }

    function getImageUrl(param) {
        return API_BASE + "/output/" + currentRun + "/" + param + ".png";
    }

    async function queryPoint(lat, lon) {
        try {
            const resp = await fetch(
                API_BASE + "/api/point?lat=" + lat.toFixed(4) + "&lon=" + lon.toFixed(4) + "&run=" + currentRun
            );
            if (resp.ok) {
                return await resp.json();
            }
        } catch (e) {
            console.warn("Point query failed:", e);
        }
        return null;
    }

    // -----------------------------------------------------------
    // Parameter selection
    // -----------------------------------------------------------

    function selectParam(param) {
        if (!PARAM_META[param]) return;
        currentParam = param;

        // Update button states
        document.querySelectorAll(".param-btn").forEach(btn => {
            btn.classList.toggle("active", btn.dataset.param === param);
        });

        // Ensure the category is open
        const activeBtn = document.querySelector('.param-btn[data-param="' + param + '"]');
        if (activeBtn) {
            const cat = activeBtn.closest(".category");
            if (cat && !cat.classList.contains("open")) {
                cat.classList.add("open");
            }
        }

        // Update displays
        const meta = PARAM_META[param];
        $paramTitle.textContent = meta.fullLabel;
        updateLegend(meta);
        loadImage(param);
    }

    function loadImage(param) {
        const url = getImageUrl(param);

        // Check cache
        if (imageCache[url]) {
            $paramImage.src = url;
            $paramImage.style.opacity = $opacitySlider.value / 100;
            return;
        }

        // Show loading
        $imageLoading.classList.add("visible");
        $paramImage.style.opacity = 0;

        const img = new Image();
        img.onload = function () {
            imageCache[url] = true;
            $paramImage.src = url;
            $paramImage.style.opacity = $opacitySlider.value / 100;
            $imageLoading.classList.remove("visible");
        };
        img.onerror = function () {
            $imageLoading.classList.remove("visible");
            console.error("Failed to load image:", url);
        };
        img.src = url;
    }

    // -----------------------------------------------------------
    // Legend
    // -----------------------------------------------------------

    function updateLegend(meta) {
        $legendLabel.textContent = meta.fullLabel;
        $legendUnits.textContent = meta.units;

        $legendColorbar.innerHTML = "";

        const levels = meta.levels;
        const colors = meta.colors;
        const n = Math.min(colors.length, levels.length - 1);

        for (let i = 0; i < n; i++) {
            const seg = document.createElement("div");
            seg.className = "legend-segment";
            seg.style.backgroundColor = colors[i];

            // Tick at left edge of each segment
            const tick = document.createElement("span");
            tick.className = "legend-tick";
            tick.textContent = formatLevel(levels[i]);
            seg.appendChild(tick);

            // Last segment also gets right-edge tick
            if (i === n - 1) {
                const tickEnd = document.createElement("span");
                tickEnd.className = "legend-tick-end";
                tickEnd.textContent = formatLevel(levels[i + 1]);
                seg.appendChild(tickEnd);
            }

            $legendColorbar.appendChild(seg);
        }
    }

    function formatLevel(v) {
        if (Number.isInteger(v)) return v.toString();
        return v.toFixed(1);
    }

    // -----------------------------------------------------------
    // Sidebar
    // -----------------------------------------------------------

    function setupSidebar() {
        // Category collapse/expand
        document.querySelectorAll(".category-header").forEach(header => {
            header.addEventListener("click", () => {
                header.parentElement.classList.toggle("open");
            });
        });

        // Open all categories by default
        document.querySelectorAll(".category").forEach(cat => {
            cat.classList.add("open");
        });

        // Parameter buttons
        document.querySelectorAll(".param-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                selectParam(btn.dataset.param);
            });
        });
    }

    // -----------------------------------------------------------
    // Controls
    // -----------------------------------------------------------

    function setupControls() {
        // Opacity slider
        $opacitySlider.addEventListener("input", () => {
            const val = $opacitySlider.value;
            $opacityValue.textContent = val + "%";
            $paramImage.style.opacity = val / 100;
            if (imageOverlay) {
                imageOverlay.setOpacity(val / 100);
            }
        });

        // View mode toggle
        $btnImageView.addEventListener("click", () => setViewMode("image"));
        $btnMapView.addEventListener("click", () => setViewMode("map"));

        // Run selector
        $runSelect.addEventListener("change", () => {
            currentRun = $runSelect.value;
            $analysisTime.textContent = formatRunName(currentRun);
            // Clear image cache for new run
            Object.keys(imageCache).forEach(k => delete imageCache[k]);
            selectParam(currentParam);
        });

        // Point query on image click
        $paramImage.addEventListener("click", handleImageClick);

        // Close popup
        $popupClose.addEventListener("click", () => {
            $pointPopup.classList.add("hidden");
        });

        // Keyboard navigation
        document.addEventListener("keydown", handleKeyboard);
    }

    function setViewMode(mode) {
        viewMode = mode;
        $btnImageView.classList.toggle("active", mode === "image");
        $btnMapView.classList.toggle("active", mode === "map");
        $imageViewer.classList.toggle("active", mode === "image");
        $mapViewer.classList.toggle("active", mode === "map");

        if (mode === "map" && !leafletMap) {
            initLeafletMap();
        }
        if (mode === "map") {
            leafletMap.invalidateSize();
            updateMapOverlay();
        }
    }

    // -----------------------------------------------------------
    // Leaflet Map
    // -----------------------------------------------------------

    function initLeafletMap() {
        leafletMap = L.map("leaflet-map", {
            center: [38.5, -97.5],
            zoom: 5,
            minZoom: 3,
            maxZoom: 10,
            zoomControl: true,
        });

        // Dark basemap
        L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
            subdomains: "abcd",
            maxZoom: 19,
        }).addTo(leafletMap);

        // Click handler for point query
        leafletMap.on("click", async function (e) {
            const lat = e.latlng.lat;
            const lon = e.latlng.lng;
            showPointQuery(lat, lon);
        });
    }

    function updateMapOverlay() {
        if (!leafletMap) return;

        const url = getImageUrl(currentParam);
        // Approximate bounds for the matplotlib figure's data area
        // These are the CONUS extent from config.py
        const bounds = [[24.0, -125.0], [50.0, -66.0]];

        if (imageOverlay) {
            imageOverlay.setUrl(url);
            imageOverlay.setOpacity($opacitySlider.value / 100);
        } else {
            imageOverlay = L.imageOverlay(url, bounds, {
                opacity: $opacitySlider.value / 100,
                interactive: false,
            }).addTo(leafletMap);
        }
    }

    // -----------------------------------------------------------
    // Point Query
    // -----------------------------------------------------------

    async function handleImageClick(e) {
        // Estimate lat/lon from click position on the image
        const rect = $paramImage.getBoundingClientRect();
        const x = (e.clientX - rect.left) / rect.width;
        const y = (e.clientY - rect.top) / rect.height;

        // Approximate mapping — the matplotlib image has margins
        // Typical matplotlib figure with bbox_inches="tight" has ~10% margins
        const marginLeft = 0.08;
        const marginRight = 0.02;
        const marginTop = 0.08;
        const marginBottom = 0.15; // colorbar takes extra space

        const dataX = (x - marginLeft) / (1 - marginLeft - marginRight);
        const dataY = (y - marginTop) / (1 - marginTop - marginBottom);

        if (dataX < 0 || dataX > 1 || dataY < 0 || dataY > 1) return;

        // Map to CONUS extent (Lambert Conformal — this is approximate)
        const lon = -125 + dataX * (125 - 66);
        const lat = 50 - dataY * (50 - 24);

        showPointQuery(lat, lon);
    }

    async function showPointQuery(lat, lon) {
        $pointPopup.classList.remove("hidden");
        $popupContent.innerHTML = '<div style="color: var(--text-muted); padding: 8px;">Loading...</div>';

        const data = await queryPoint(lat, lon);

        let html = '<div class="popup-coords">' + lat.toFixed(3) + '\u00b0N, ' + Math.abs(lon).toFixed(3) + '\u00b0W</div>';

        if (data && data.values) {
            const paramOrder = [
                "sbcape","mlcape","mucape","sb3cape","sbcin",
                "theta_e_sfc","theta_e_850","mixing_ratio","pw","wetbulb_sfc",
                "lr_700_500",
                "shear_01km","shear_06km","srh_01km","srh_03km",
                "stp","scp",
                "t2m_f","td2m_f"
            ];

            for (const key of paramOrder) {
                if (data.values[key] !== undefined && PARAM_META[key]) {
                    const meta = PARAM_META[key];
                    const val = typeof data.values[key] === "number"
                        ? data.values[key].toFixed(1)
                        : data.values[key];
                    const isActive = key === currentParam ? ' style="color: var(--highlight); font-weight: 700;"' : '';
                    html += '<div class="popup-row">'
                        + '<span class="popup-key">' + meta.label + '</span>'
                        + '<span class="popup-val"' + isActive + '>' + val + ' ' + meta.units + '</span>'
                        + '</div>';
                }
            }
        } else {
            html += '<div style="color: var(--text-muted); font-size: 12px; padding: 4px 0;">Point query unavailable. Server may not have grid data loaded.</div>';
        }

        document.getElementById("popup-title").textContent =
            "Point Query \u2014 " + lat.toFixed(2) + "\u00b0, " + lon.toFixed(2) + "\u00b0";
        $popupContent.innerHTML = html;
    }

    // -----------------------------------------------------------
    // Keyboard Navigation
    // -----------------------------------------------------------

    function handleKeyboard(e) {
        // Don't intercept if user is in an input
        if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;

        const allParams = Object.keys(PARAM_META);
        const idx = allParams.indexOf(currentParam);

        if (e.key === "ArrowDown" || e.key === "j") {
            e.preventDefault();
            const next = (idx + 1) % allParams.length;
            selectParam(allParams[next]);
        } else if (e.key === "ArrowUp" || e.key === "k") {
            e.preventDefault();
            const prev = (idx - 1 + allParams.length) % allParams.length;
            selectParam(allParams[prev]);
        } else if (e.key === "Escape") {
            $pointPopup.classList.add("hidden");
        }
    }

    // -----------------------------------------------------------
    // Start
    // -----------------------------------------------------------

    document.addEventListener("DOMContentLoaded", init);

})();
