/* ============================================================
   mesoanalysis — Windy-style Interactive Weather Map
   ============================================================ */

(function () {
    "use strict";

    // -----------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------

    const API_BASE = "";
    const CONUS_CENTER = [-97.5, 38.5];
    const CONUS_ZOOM = 4;
    const DEFAULT_PARAM = "sbcape";
    const OBS_REFRESH_MS = 5 * 60 * 1000;

    const BASEMAP_STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-nolabels-gl-style/style.json";
    const BASEMAP_RASTER = "https://basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}@2x.png";

    // -----------------------------------------------------------
    // Color helpers
    // -----------------------------------------------------------

    function hexToRgb(hex) {
        const m = hex.match(/^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i);
        return m ? [parseInt(m[1], 16), parseInt(m[2], 16), parseInt(m[3], 16)] : [0, 0, 0];
    }

    function rgbToHex(r, g, b) {
        return "#" + [r, g, b].map(x => Math.max(0, Math.min(255, Math.round(x))).toString(16).padStart(2, "0")).join("");
    }

    function interpolateColors(anchors, n) {
        if (n <= 0) return [];
        if (n === 1) return [anchors[0]];
        const result = [];
        for (let i = 0; i < n; i++) {
            const t = i / (n - 1);
            const idx = t * (anchors.length - 1);
            const lo = Math.floor(idx);
            const hi = Math.min(lo + 1, anchors.length - 1);
            const f = idx - lo;
            const c1 = hexToRgb(anchors[lo]);
            const c2 = hexToRgb(anchors[hi]);
            result.push(rgbToHex(
                c1[0] + (c2[0] - c1[0]) * f,
                c1[1] + (c2[1] - c1[1]) * f,
                c1[2] + (c2[2] - c1[2]) * f
            ));
        }
        return result;
    }

    function levelsFromRange(lo, hi, step) {
        const arr = [];
        for (let v = lo; v <= hi + step / 4; v += step) {
            arr.push(Math.round(v * 100) / 100);
        }
        return arr;
    }

    // Approximate matplotlib colormaps
    const CMAP = {
        YlOrRd:   ["#ffffb2","#ffeda0","#fed976","#feb24c","#fd8d3c","#fc4e2a","#e31a1c","#bd0026","#800026"],
        RdYlBu_r: ["#a50026","#d73027","#f46d43","#fdae61","#fee090","#ffffbf","#e0f3f8","#abd9e9","#74add1","#4575b4","#313695"].reverse(),
        RdPu:     ["#feebe2","#fcc5c0","#fa9fb5","#f768a1","#dd3497","#ae017e","#7a0177","#49006a"],
        plasma:   ["#0d0887","#46039f","#7201a8","#9c179e","#bd3786","#d8576b","#ed7953","#fca636","#f0f921"],
        OrRd:     ["#fff7ec","#fee8c8","#fdd49e","#fdbb84","#fc8d59","#ef6548","#d7301f","#b30000","#7f0000"],
        Reds:     ["#fff5f0","#fee0d2","#fcbba1","#fc9272","#fb6a4a","#ef3b2c","#cb181d","#a50f15","#67000d"],
        Greens:   ["#f7fcf5","#e5f5e0","#c7e9c0","#a1d99b","#74c476","#41ab5d","#238b45","#006d2c","#00441b"],
        GnBu:     ["#f7fcf0","#e0f3db","#ccebc5","#a8ddb5","#7bccc4","#4eb3d3","#2b8cbe","#0868ac","#084081"],
        BuPu:     ["#f7fcfd","#e0ecf4","#bfd3e6","#9ebcda","#8c96c6","#8c6bb1","#88419d","#810f7c","#4d004b"],
        BuPu_r:   ["#4d004b","#810f7c","#88419d","#8c6bb1","#8c96c6","#9ebcda","#bfd3e6","#e0ecf4","#f7fcfd"],
        BrBG:     ["#543005","#8c510a","#bf812d","#dfc27d","#f6e8c3","#f5f5f5","#c7eae5","#80cdc1","#35978f","#01665e","#003c30"],
        RdYlBu:   ["#a50026","#d73027","#f46d43","#fdae61","#fee090","#ffffbf","#e0f3f8","#abd9e9","#74add1","#4575b4","#313695"],
    };

    function cmapColors(name, n) {
        const anchors = CMAP[name];
        if (!anchors) return interpolateColors(CMAP.YlOrRd, n);
        return interpolateColors(anchors, n);
    }

    // -----------------------------------------------------------
    // Parameter metadata — mirrors styles.py
    // -----------------------------------------------------------

    const CATEGORIES = {
        severe: {
            label: "Severe",
            params: ["sbcape", "mlcape", "mucape", "stp", "scp", "ship", "ehi", "dcp"]
        },
        shear: {
            label: "Shear",
            params: ["shear_01km", "shear_06km", "shear_03km", "srh_01km", "srh_03km", "critical_angle"]
        },
        thermo: {
            label: "Thermo",
            params: ["theta_e_sfc", "theta_e_850", "lr_700_500", "lr_850_500", "dcape", "lifted_index", "k_index", "total_totals"]
        },
        moisture: {
            label: "Moisture",
            params: ["mixing_ratio", "pw", "td2m_f"]
        },
        surface: {
            label: "Surface",
            params: ["t2m_f", "heat_index", "windchill", "wbgt", "wetbulb_sfc"]
        },
        fire: {
            label: "Fire",
            params: ["ffwi", "hdw"]
        }
    };

    const PARAMS = {
        sbcape: {
            label: "SBCAPE", units: "J/kg", fullLabel: "SBCAPE (J/kg)",
            levels: [100, 250, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000],
            colors: cmapColors("YlOrRd", 9),
        },
        mlcape: {
            label: "MLCAPE", units: "J/kg", fullLabel: "MLCAPE (J/kg)",
            levels: [100, 250, 500, 1000, 1500, 2000, 2500, 3000, 4000],
            colors: cmapColors("YlOrRd", 8),
        },
        mucape: {
            label: "MUCAPE", units: "J/kg", fullLabel: "MUCAPE (J/kg)",
            levels: [100, 250, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000],
            colors: cmapColors("YlOrRd", 9),
        },
        stp: {
            label: "STP", units: "", fullLabel: "Sig Tornado Parameter",
            levels: [0.5, 1, 2, 3, 5, 8, 12],
            colors: ["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026", "#67000d"],
        },
        scp: {
            label: "SCP", units: "", fullLabel: "Supercell Composite",
            levels: [1, 2, 4, 6, 8, 10, 15],
            colors: cmapColors("OrRd", 6),
        },
        ship: {
            label: "SHIP", units: "", fullLabel: "Sig Hail Parameter",
            levels: [0.5, 1, 1.5, 2, 3, 4, 5],
            colors: ["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026", "#67000d"],
        },
        ehi: {
            label: "EHI", units: "", fullLabel: "Energy-Helicity Index",
            levels: [0.5, 1, 2, 3, 5, 8, 12],
            colors: cmapColors("OrRd", 6),
        },
        dcp: {
            label: "DCP", units: "", fullLabel: "Derecho Composite Parameter",
            levels: [0.5, 1, 2, 3, 4, 6, 8],
            colors: cmapColors("YlOrRd", 6),
        },
        shear_01km: {
            label: "0-1km Shear", units: "kt", fullLabel: "0-1km Shear (kt)",
            levels: [5, 10, 15, 20, 25, 30, 35, 40],
            colors: cmapColors("plasma", 7),
        },
        shear_06km: {
            label: "0-6km Shear", units: "kt", fullLabel: "0-6km Shear (kt)",
            levels: [10, 20, 30, 40, 50, 60, 70, 80],
            colors: cmapColors("plasma", 7),
        },
        shear_03km: {
            label: "0-3km Shear", units: "kt", fullLabel: "0-3km Shear (kt)",
            levels: [5, 10, 15, 20, 25, 30, 35, 40, 50],
            colors: cmapColors("plasma", 8),
        },
        srh_01km: {
            label: "0-1km SRH", units: "m\u00b2/s\u00b2", fullLabel: "0-1km SRH (m\u00b2/s\u00b2)",
            levels: [50, 100, 150, 200, 300, 400, 500],
            colors: cmapColors("RdPu", 6),
        },
        srh_03km: {
            label: "0-3km SRH", units: "m\u00b2/s\u00b2", fullLabel: "0-3km SRH (m\u00b2/s\u00b2)",
            levels: [100, 200, 300, 400, 500, 600, 800],
            colors: cmapColors("RdPu", 6),
        },
        critical_angle: {
            label: "Critical Angle", units: "\u00b0", fullLabel: "Critical Angle (\u00b0)",
            levels: [30, 45, 60, 70, 80, 90, 100, 110, 120],
            colors: cmapColors("RdYlBu_r", 8),
        },
        theta_e_sfc: {
            label: "Sfc \u03b8e", units: "K", fullLabel: "Sfc Theta-e (K)",
            levels: levelsFromRange(290, 370, 5),
            colors: cmapColors("RdYlBu_r", levelsFromRange(290, 370, 5).length - 1),
        },
        theta_e_850: {
            label: "850mb \u03b8e", units: "K", fullLabel: "850mb Theta-e (K)",
            levels: levelsFromRange(280, 360, 5),
            colors: cmapColors("RdYlBu_r", levelsFromRange(280, 360, 5).length - 1),
        },
        lr_700_500: {
            label: "700-500 LR", units: "\u00b0C/km", fullLabel: "700-500mb Lapse Rate (\u00b0C/km)",
            levels: levelsFromRange(4, 10, 0.5),
            colors: cmapColors("Reds", levelsFromRange(4, 10, 0.5).length - 1),
        },
        lr_850_500: {
            label: "850-500 LR", units: "\u00b0C/km", fullLabel: "850-500mb Lapse Rate (\u00b0C/km)",
            levels: levelsFromRange(4, 10, 0.5),
            colors: cmapColors("Reds", levelsFromRange(4, 10, 0.5).length - 1),
        },
        dcape: {
            label: "DCAPE", units: "J/kg", fullLabel: "DCAPE (J/kg)",
            levels: [200, 400, 600, 800, 1000, 1500, 2000],
            colors: cmapColors("BuPu", 6),
        },
        lifted_index: {
            label: "Lifted Index", units: "\u00b0C", fullLabel: "Lifted Index (\u00b0C)",
            levels: [-10, -8, -6, -4, -2, 0, 2, 4, 6],
            colors: cmapColors("RdYlBu", 8),
        },
        k_index: {
            label: "K-Index", units: "", fullLabel: "K-Index",
            levels: [15, 20, 25, 30, 35, 40],
            colors: cmapColors("YlOrRd", 5),
        },
        total_totals: {
            label: "Total Totals", units: "", fullLabel: "Total Totals Index",
            levels: [40, 44, 48, 50, 52, 55, 60],
            colors: cmapColors("YlOrRd", 6),
        },
        mixing_ratio: {
            label: "Mixing Ratio", units: "g/kg", fullLabel: "Mixing Ratio (g/kg)",
            levels: levelsFromRange(2, 24, 2),
            colors: cmapColors("Greens", levelsFromRange(2, 24, 2).length - 1),
        },
        pw: {
            label: "Precip Water", units: "mm", fullLabel: "Precipitable Water (mm)",
            levels: levelsFromRange(5, 60, 5),
            colors: cmapColors("GnBu", levelsFromRange(5, 60, 5).length - 1),
        },
        td2m_f: {
            label: "Dewpoint", units: "\u00b0F", fullLabel: "Dewpoint (\u00b0F)",
            levels: levelsFromRange(-10, 80, 5),
            colors: cmapColors("BrBG", levelsFromRange(-10, 80, 5).length - 1),
        },
        t2m_f: {
            label: "Temperature", units: "\u00b0F", fullLabel: "Temperature (\u00b0F)",
            levels: levelsFromRange(-20, 115, 5),
            colors: cmapColors("RdYlBu_r", levelsFromRange(-20, 115, 5).length - 1),
        },
        heat_index: {
            label: "Heat Index", units: "\u00b0F", fullLabel: "Heat Index (\u00b0F)",
            levels: levelsFromRange(80, 130, 5),
            colors: cmapColors("YlOrRd", levelsFromRange(80, 130, 5).length - 1),
        },
        windchill: {
            label: "Wind Chill", units: "\u00b0F", fullLabel: "Wind Chill (\u00b0F)",
            levels: levelsFromRange(-50, 40, 5),
            colors: cmapColors("BuPu_r", levelsFromRange(-50, 40, 5).length - 1),
        },
        wbgt: {
            label: "WBGT", units: "\u00b0F", fullLabel: "WBGT (\u00b0F)",
            levels: levelsFromRange(70, 110, 2),
            colors: cmapColors("YlOrRd", levelsFromRange(70, 110, 2).length - 1),
        },
        wetbulb_sfc: {
            label: "Wet Bulb", units: "\u00b0C", fullLabel: "Sfc Wet Bulb (\u00b0C)",
            levels: levelsFromRange(-25, 35, 2),
            colors: cmapColors("RdYlBu_r", levelsFromRange(-25, 35, 2).length - 1),
        },
        ffwi: {
            label: "FFWI", units: "", fullLabel: "Fosberg Fire Weather Index",
            levels: [10, 20, 30, 40, 50, 60, 70, 80, 100],
            colors: cmapColors("YlOrRd", 8),
        },
        hdw: {
            label: "HDW", units: "", fullLabel: "Hot-Dry-Windy Index",
            levels: [50, 100, 200, 400, 600, 800, 1000, 1500],
            colors: cmapColors("YlOrRd", 7),
        },
    };

    // -----------------------------------------------------------
    // Temperature color scale for station dots
    // -----------------------------------------------------------

    const TEMP_SCALE = [
        [-20, [100, 130, 230]],
        [0,   [70,  130, 210]],
        [20,  [60,  170, 220]],
        [32,  [80,  200, 200]],
        [50,  [80,  200, 120]],
        [60,  [140, 210, 80]],
        [70,  [220, 220, 50]],
        [80,  [240, 180, 40]],
        [90,  [230, 110, 40]],
        [100, [210, 50,  50]],
        [115, [170, 30,  80]],
    ];

    function tempToColor(f) {
        if (f <= TEMP_SCALE[0][0]) return TEMP_SCALE[0][1];
        if (f >= TEMP_SCALE[TEMP_SCALE.length - 1][0]) return TEMP_SCALE[TEMP_SCALE.length - 1][1];
        for (let i = 1; i < TEMP_SCALE.length; i++) {
            if (f <= TEMP_SCALE[i][0]) {
                const t = (f - TEMP_SCALE[i - 1][0]) / (TEMP_SCALE[i][0] - TEMP_SCALE[i - 1][0]);
                const a = TEMP_SCALE[i - 1][1];
                const b = TEMP_SCALE[i][1];
                return [
                    Math.round(a[0] + (b[0] - a[0]) * t),
                    Math.round(a[1] + (b[1] - a[1]) * t),
                    Math.round(a[2] + (b[2] - a[2]) * t),
                ];
            }
        }
        return [200, 200, 200];
    }

    function tempToHex(f) {
        const c = tempToColor(f);
        return rgbToHex(c[0], c[1], c[2]);
    }

    // -----------------------------------------------------------
    // State
    // -----------------------------------------------------------

    let map = null;
    let currentParam = DEFAULT_PARAM;
    let currentCategory = "severe";
    let currentModel = "hrrr";
    let currentRunId = null;
    let overlayOpacity = 0.7;
    let overlayBounds = null; // [[west, south], [east, north]]
    let obsData = null;
    let obsTimer = null;
    let activePopup = null;
    let overlaySourceId = "param-overlay";
    let overlayLayerId = "param-overlay-layer";
    let prevOverlaySourceId = null;
    let prevOverlayLayerId = null;
    let fadeCounter = 0;

    // -----------------------------------------------------------
    // DOM references
    // -----------------------------------------------------------

    const $runTime = document.getElementById("run-time");
    const $modelSelect = document.getElementById("model-select");
    const $opacitySlider = document.getElementById("opacity-slider");
    const $legendLabel = document.getElementById("legend-label");
    const $legendGradient = document.getElementById("legend-gradient");
    const $legendTicks = document.getElementById("legend-ticks");
    const $paramPills = document.getElementById("param-pills");
    const $tooltip = document.getElementById("station-tooltip");
    const $popup = document.getElementById("station-popup");
    const $popupContent = document.getElementById("station-popup-content");
    const $popupClose = document.getElementById("station-popup-close");

    // -----------------------------------------------------------
    // Initialization
    // -----------------------------------------------------------

    async function init() {
        setupCategoryTabs();
        setupControls();
        renderParamPills("severe");
        await loadRuns();
        initMap();
    }

    function initMap() {
        // Try vector style first, fall back to raster
        map = new maplibregl.Map({
            container: "map",
            style: {
                version: 8,
                sources: {
                    "carto-dark": {
                        type: "raster",
                        tiles: [BASEMAP_RASTER],
                        tileSize: 256,
                        attribution: "&copy; OpenStreetMap &copy; CARTO",
                    }
                },
                layers: [{
                    id: "carto-dark-layer",
                    type: "raster",
                    source: "carto-dark",
                    minzoom: 0,
                    maxzoom: 19,
                }]
            },
            center: CONUS_CENTER,
            zoom: CONUS_ZOOM,
            minZoom: 3,
            maxZoom: 12,
            pitchWithRotate: false,
            dragRotate: false,
            touchZoomRotate: true,
        });

        // Disable rotation
        map.touchZoomRotate.disableRotation();

        // Add navigation control (zoom only)
        map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");

        map.on("load", function () {
            // Select default parameter
            selectParam(DEFAULT_PARAM);
            // Load observations
            loadObs();
            obsTimer = setInterval(loadObs, OBS_REFRESH_MS);
        });
    }

    // -----------------------------------------------------------
    // API
    // -----------------------------------------------------------

    async function loadRuns() {
        try {
            const resp = await fetch(API_BASE + "/api/runs");
            if (!resp.ok) throw new Error("HTTP " + resp.status);
            const data = await resp.json();
            // Find latest run: check data.latest first, then scan model keys
            let latest = data.latest;
            if (!latest) {
                for (const m of ["hrrr", "rap", "nam"]) {
                    const runs = data[m];
                    if (runs && runs.length > 0) {
                        latest = { model: m, run_id: runs[0] };
                        break;
                    }
                }
            }
            if (latest) {
                currentModel = latest.model || currentModel;
                currentRunId = latest.run_id || latest;
                $modelSelect.value = currentModel;
                updateRunTimeDisplay();
            }
        } catch (e) {
            console.warn("Failed to load runs:", e);
            $runTime.textContent = "No data";
        }
    }

    function updateRunTimeDisplay() {
        if (!currentRunId) {
            $runTime.textContent = "--";
            return;
        }
        // Format: 20260324_0400 -> 2026-03-24 04:00Z
        const s = String(currentRunId);
        if (s.length >= 13) {
            $runTime.textContent = s.slice(0, 4) + "-" + s.slice(4, 6) + "-" + s.slice(6, 8) +
                " " + s.slice(9, 11) + ":" + s.slice(11, 13) + "Z";
        } else {
            $runTime.textContent = s;
        }
    }

    function tileUrl(param) {
        return API_BASE + "/api/tiles/" + currentModel + "/" + currentRunId + "/" + param + ".png";
    }

    async function loadManifest() {
        if (!currentRunId) return;
        try {
            const resp = await fetch(API_BASE + "/api/manifest/" + currentModel + "/" + currentRunId);
            if (!resp.ok) return;
            const data = await resp.json();
            if (data.bounds) {
                // bounds: {west, south, east, north}
                overlayBounds = data.bounds;
            }
        } catch (e) {
            console.warn("Manifest load failed:", e);
        }
    }

    async function loadObs() {
        try {
            const resp = await fetch(API_BASE + "/api/obs?window=20");
            if (!resp.ok) return;
            const data = await resp.json();
            if (data && data.type === "FeatureCollection") {
                obsData = data;
                renderStations();
            }
        } catch (e) {
            console.warn("Obs load failed:", e);
        }
    }

    async function loadPointData(lat, lon) {
        try {
            const resp = await fetch(
                API_BASE + "/api/point/" + currentModel + "/" + currentRunId +
                "?lat=" + lat.toFixed(4) + "&lon=" + lon.toFixed(4)
            );
            if (!resp.ok) return null;
            return await resp.json();
        } catch (e) {
            return null;
        }
    }

    // -----------------------------------------------------------
    // Parameter overlay
    // -----------------------------------------------------------

    function addOverlay(param) {
        if (!map || !map.loaded()) return;
        if (!currentRunId) return;

        const url = tileUrl(param);

        // Default CONUS bounds if no manifest
        const bounds = overlayBounds || { west: -125, south: 24, east: -66, north: 50 };

        // MapLibre image source wants coordinates as [[topleft], [topright], [bottomright], [bottomleft]]
        const coordinates = [
            [bounds.west, bounds.north],  // top-left
            [bounds.east, bounds.north],  // top-right
            [bounds.east, bounds.south],  // bottom-right
            [bounds.west, bounds.south],  // bottom-left
        ];

        const newSourceId = "param-overlay-" + (++fadeCounter);
        const newLayerId = "param-overlay-layer-" + fadeCounter;

        // Add new source + layer
        map.addSource(newSourceId, {
            type: "image",
            url: url,
            coordinates: coordinates,
        });

        map.addLayer({
            id: newLayerId,
            type: "raster",
            source: newSourceId,
            paint: {
                "raster-opacity": 0,
                "raster-opacity-transition": { duration: 0 },
                "raster-fade-duration": 0,
            },
        });

        // Move new layer below station layers if they exist
        if (map.getLayer("stations-circle")) {
            map.moveLayer(newLayerId, "stations-circle");
        }

        // Crossfade: fade in new, fade out old
        const oldSourceId = overlaySourceId;
        const oldLayerId = overlayLayerId;
        const hasOld = map.getLayer(oldLayerId);

        overlaySourceId = newSourceId;
        overlayLayerId = newLayerId;

        // Use requestAnimationFrame for smooth transition
        requestAnimationFrame(function () {
            // Fade in new layer
            if (map.getLayer(newLayerId)) {
                map.setPaintProperty(newLayerId, "raster-opacity", overlayOpacity);
                map.setPaintProperty(newLayerId, "raster-opacity-transition", { duration: 400 });
            }

            // Fade out old layer
            if (hasOld && map.getLayer(oldLayerId)) {
                map.setPaintProperty(oldLayerId, "raster-opacity", 0);
                map.setPaintProperty(oldLayerId, "raster-opacity-transition", { duration: 400 });

                // Remove old layer/source after transition
                setTimeout(function () {
                    try {
                        if (map.getLayer(oldLayerId)) map.removeLayer(oldLayerId);
                        if (map.getSource(oldSourceId)) map.removeSource(oldSourceId);
                    } catch (e) { /* ignore */ }
                }, 500);
            }
        });
    }

    function updateOverlayOpacity() {
        if (map && map.getLayer(overlayLayerId)) {
            map.setPaintProperty(overlayLayerId, "raster-opacity", overlayOpacity);
        }
    }

    // -----------------------------------------------------------
    // Station rendering
    // -----------------------------------------------------------

    function renderStations() {
        if (!map || !map.loaded() || !obsData) return;

        // Build GeoJSON with color properties
        const features = obsData.features.map(function (f) {
            const props = f.properties || {};
            const temp = props.temperature_f != null ? props.temperature_f :
                         props.temp_f != null ? props.temp_f :
                         props.temperature != null ? props.temperature : null;
            const color = temp != null ? tempToHex(temp) : "#888888";
            return {
                type: "Feature",
                geometry: f.geometry,
                properties: Object.assign({}, props, { _color: color, _temp: temp }),
            };
        });

        const geojson = { type: "FeatureCollection", features: features };

        if (map.getSource("stations")) {
            map.getSource("stations").setData(geojson);
        } else {
            map.addSource("stations", { type: "geojson", data: geojson });

            map.addLayer({
                id: "stations-circle",
                type: "circle",
                source: "stations",
                paint: {
                    "circle-radius": 4,
                    "circle-color": ["get", "_color"],
                    "circle-stroke-width": 1,
                    "circle-stroke-color": "rgba(0,0,0,0.6)",
                    "circle-opacity": 0.9,
                },
            });

            // Hover: show tooltip
            map.on("mouseenter", "stations-circle", function (e) {
                map.getCanvas().style.cursor = "pointer";
            });

            map.on("mousemove", "stations-circle", function (e) {
                if (e.features && e.features.length > 0) {
                    const props = e.features[0].properties;
                    const name = props.station_id || props.station || props.id || "Station";
                    const temp = props._temp;
                    let text = name;
                    if (temp != null) text += "  " + Math.round(temp) + "\u00b0F";
                    $tooltip.textContent = text;
                    $tooltip.classList.remove("hidden");
                    $tooltip.style.left = (e.originalEvent.clientX + 14) + "px";
                    $tooltip.style.top = (e.originalEvent.clientY - 8) + "px";
                }
            });

            map.on("mouseleave", "stations-circle", function () {
                map.getCanvas().style.cursor = "";
                $tooltip.classList.add("hidden");
            });

            // Click: show popup
            map.on("click", "stations-circle", function (e) {
                if (e.features && e.features.length > 0) {
                    const f = e.features[0];
                    showStationPopup(f.properties, f.geometry.coordinates);
                    e.preventDefault();
                }
            });
        }
    }

    // -----------------------------------------------------------
    // Station popup
    // -----------------------------------------------------------

    function showStationPopup(props, coords) {
        const name = props.station_name || props.name || props.station_id || props.station || "Unknown";
        const id = props.station_id || props.station || props.id || "";
        const temp = props.temperature_f != null ? props.temperature_f :
                     props.temp_f != null ? props.temp_f :
                     props.temperature != null ? props.temperature : null;
        const dewp = props.dewpoint_f != null ? props.dewpoint_f :
                     props.dew_f != null ? props.dew_f :
                     props.dewpoint != null ? props.dewpoint : null;
        const windSpd = props.wind_speed_kt != null ? props.wind_speed_kt :
                        props.wind_speed != null ? props.wind_speed : null;
        const windDir = props.wind_direction != null ? props.wind_direction :
                        props.wind_dir != null ? props.wind_dir : null;
        const pres = props.pressure_hpa != null ? props.pressure_hpa :
                     props.pressure != null ? props.pressure :
                     props.slp != null ? props.slp :
                     props.mslp != null ? props.mslp : null;
        const obsTime = props.observation_time || props.time || props.obs_time || null;

        let html = '<div class="popup-station-name">' + escHtml(name) + '</div>';
        if (id) html += '<div class="popup-station-id">' + escHtml(id) + '</div>';

        if (temp != null) {
            html += '<div class="popup-row"><span class="popup-row-label">Temperature</span>' +
                    '<span class="popup-row-value">' + round1(temp) + '\u00b0F</span></div>';
        }
        if (dewp != null) {
            html += '<div class="popup-row"><span class="popup-row-label">Dewpoint</span>' +
                    '<span class="popup-row-value">' + round1(dewp) + '\u00b0F</span></div>';
        }
        if (windDir != null || windSpd != null) {
            let windStr = "";
            if (windDir != null) {
                windStr += '<span class="popup-wind-arrow" style="transform:rotate(' + (windDir + 180) + 'deg)">\u2191</span>';
                windStr += degToCompass(windDir) + " ";
            }
            if (windSpd != null) {
                windStr += Math.round(windSpd) + " kt";
            }
            html += '<div class="popup-row"><span class="popup-row-label">Wind</span>' +
                    '<span class="popup-row-value">' + windStr + '</span></div>';
        }
        if (pres != null) {
            html += '<div class="popup-row"><span class="popup-row-label">Pressure</span>' +
                    '<span class="popup-row-value">' + round1(pres) + ' hPa</span></div>';
        }
        if (obsTime) {
            html += '<div class="popup-obs-time">' + escHtml(obsTime) + '</div>';
        }

        $popupContent.innerHTML = html;
        $popup.classList.remove("hidden");

        // Also show MapLibre popup on map
        if (activePopup) activePopup.remove();
        activePopup = new maplibregl.Popup({ closeOnClick: true, maxWidth: "280px" })
            .setLngLat(coords)
            .setHTML(html)
            .addTo(map);

        activePopup.on("close", function () {
            $popup.classList.add("hidden");
        });
    }

    function degToCompass(deg) {
        const dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                      "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"];
        return dirs[Math.round(deg / 22.5) % 16];
    }

    function round1(v) {
        return Math.round(v * 10) / 10;
    }

    function escHtml(s) {
        const el = document.createElement("span");
        el.textContent = s;
        return el.innerHTML;
    }

    // -----------------------------------------------------------
    // Legend
    // -----------------------------------------------------------

    function updateLegend(param) {
        const meta = PARAMS[param];
        if (!meta) return;

        $legendLabel.textContent = meta.fullLabel;

        const levels = meta.levels;
        const colors = meta.colors;
        const n = Math.min(colors.length, levels.length - 1);

        // Build gradient
        const stops = [];
        for (let i = 0; i < colors.length; i++) {
            const pct = (i / (colors.length - 1)) * 100;
            stops.push(colors[i] + " " + pct.toFixed(1) + "%");
        }
        $legendGradient.style.background = "linear-gradient(to right, " + stops.join(", ") + ")";

        // Build ticks
        $legendTicks.innerHTML = "";
        const tickCount = Math.min(levels.length, 12); // limit ticks for readability
        const step = Math.max(1, Math.floor(levels.length / tickCount));
        for (let i = 0; i < levels.length; i += step) {
            const tick = document.createElement("span");
            tick.className = "legend-tick";
            const pct = (i / (levels.length - 1)) * 100;
            tick.style.left = pct.toFixed(1) + "%";
            tick.textContent = formatLevel(levels[i]);
            $legendTicks.appendChild(tick);
        }
        // Always show last tick
        if ((levels.length - 1) % step !== 0) {
            const tick = document.createElement("span");
            tick.className = "legend-tick";
            tick.style.left = "100%";
            tick.textContent = formatLevel(levels[levels.length - 1]);
            $legendTicks.appendChild(tick);
        }
    }

    function formatLevel(v) {
        if (Number.isInteger(v)) return v.toString();
        if (Math.abs(v) >= 10) return Math.round(v).toString();
        return v.toFixed(1);
    }

    // -----------------------------------------------------------
    // Parameter selection
    // -----------------------------------------------------------

    function selectParam(param) {
        if (!PARAMS[param]) return;
        currentParam = param;

        // Update pill states
        document.querySelectorAll(".param-pill").forEach(function (pill) {
            pill.classList.toggle("active", pill.dataset.param === param);
        });

        // Scroll active pill into view
        const activePill = document.querySelector('.param-pill[data-param="' + param + '"]');
        if (activePill) {
            activePill.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
        }

        updateLegend(param);
        addOverlay(param);
    }

    // -----------------------------------------------------------
    // Category tabs & parameter pills
    // -----------------------------------------------------------

    function setupCategoryTabs() {
        document.querySelectorAll(".cat-tab").forEach(function (tab) {
            tab.addEventListener("click", function () {
                const cat = tab.dataset.cat;
                if (cat === currentCategory) return;
                currentCategory = cat;

                document.querySelectorAll(".cat-tab").forEach(function (t) {
                    t.classList.toggle("active", t.dataset.cat === cat);
                });

                renderParamPills(cat);
            });
        });
    }

    function renderParamPills(cat) {
        $paramPills.innerHTML = "";
        const catDef = CATEGORIES[cat];
        if (!catDef) return;

        catDef.params.forEach(function (p) {
            const meta = PARAMS[p];
            if (!meta) return;

            const pill = document.createElement("button");
            pill.className = "param-pill" + (p === currentParam ? " active" : "");
            pill.dataset.param = p;
            pill.textContent = meta.label;
            pill.addEventListener("click", function () {
                selectParam(p);
            });
            $paramPills.appendChild(pill);
        });
    }

    // -----------------------------------------------------------
    // Controls
    // -----------------------------------------------------------

    function setupControls() {
        $opacitySlider.addEventListener("input", function () {
            overlayOpacity = $opacitySlider.value / 100;
            updateOverlayOpacity();
        });

        $modelSelect.addEventListener("change", function () {
            currentModel = $modelSelect.value;
            loadRuns().then(function () {
                selectParam(currentParam);
            });
        });

        $popupClose.addEventListener("click", function () {
            $popup.classList.add("hidden");
            if (activePopup) {
                activePopup.remove();
                activePopup = null;
            }
        });

        // Keyboard navigation
        document.addEventListener("keydown", function (e) {
            if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;

            const catDef = CATEGORIES[currentCategory];
            if (!catDef) return;
            const params = catDef.params.filter(function (p) { return !!PARAMS[p]; });
            const idx = params.indexOf(currentParam);

            if (e.key === "ArrowRight" || e.key === "l") {
                e.preventDefault();
                const next = idx < 0 ? 0 : (idx + 1) % params.length;
                selectParam(params[next]);
            } else if (e.key === "ArrowLeft" || e.key === "h") {
                e.preventDefault();
                const prev = idx < 0 ? 0 : (idx - 1 + params.length) % params.length;
                selectParam(params[prev]);
            } else if (e.key === "Escape") {
                $popup.classList.add("hidden");
                if (activePopup) {
                    activePopup.remove();
                    activePopup = null;
                }
            }
        });
    }

    // -----------------------------------------------------------
    // Start
    // -----------------------------------------------------------

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }

})();
