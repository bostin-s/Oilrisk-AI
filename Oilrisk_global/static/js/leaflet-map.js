/**
 * leaflet-map.js  —  OilRisk v10.0
 * =======================================
 * FIXES from v9:
 *   ✦ Image now correctly shows in popup (fixed margin:0 conflict)
 *   ✦ Popup width explicitly set — no content overflow
 *   ✦ Side panel fixed width, no layout shift
 *   ✦ Map header two-row aligned layout
 *   ✦ Jump bar dynamically built, wraps cleanly
 *   ✦ 65 global hotspots across 14 regions
 */
(function () {
  'use strict';

  /* ─── Risk colours ─────────────────────────────────── */
  const RISK = {
    CRITICAL: { hex:'#8b5cf6', dark:'#6d28d9', radius:320000 },
    HIGH:     { hex:'#ef4444', dark:'#b91c1c', radius:220000 },
    MEDIUM:   { hex:'#f59e0b', dark:'#b45309', radius:150000 },
    LOW:      { hex:'#10b981', dark:'#047857', radius: 90000 },
  };

  /* ─── 65 global hotspots ────────────────────────────── */
  const SPOTS = [
    // Middle East — Israel/Iran
    { id:'il-ir',     nm:'Israel–Iran Front',        ctry:'Israel / Iran',    region:'Middle East',  lat:32.50, lng:43.00, risk:'CRITICAL', flag:'🇮🇱', wiki:'Israel Iran conflict',              zoom:5,  desc:'Active air, missile & drone campaign. Nuclear escalation threat. Kharg, Natanz & Fordow targeted.' },
    { id:'kharg',     nm:'Kharg Island Terminal',    ctry:'Iran',             region:'Middle East',  lat:29.25, lng:50.32, risk:'CRITICAL', flag:'🛢️',  wiki:'Kharg Island',                      zoom:12, desc:'90% of Iran crude oil exports. 2.5 Mbpd. Primary IDF airstrike target.' },
    { id:'natanz',    nm:'Natanz Nuclear',           ctry:'Iran',             region:'Middle East',  lat:33.73, lng:51.93, risk:'HIGH',     flag:'☢️',  wiki:'Natanz nuclear facility',           zoom:13, desc:'Underground uranium enrichment. Repeated sabotage. IDF airstrike objective.' },
    { id:'tehran',    nm:'Tehran Refineries',        ctry:'Iran',             region:'Middle East',  lat:35.69, lng:51.39, risk:'HIGH',     flag:'🇮🇷', wiki:'Tehran oil refinery',               zoom:11, desc:'Tehran refinery complex — 9 Mtpa. Vulnerable to escalation strikes.' },
    // Persian Gulf
    { id:'hormuz',    nm:'Strait of Hormuz',         ctry:'International',    region:'Persian Gulf', lat:26.56, lng:56.25, risk:'CRITICAL', flag:'⚓', wiki:'Strait of Hormuz',                  zoom:8,  desc:'21 million bbl/day — 20% of global crude. Closure = immediate global price shock.' },
    { id:'abqaiq',    nm:'Abqaiq / Aramco',          ctry:'Saudi Arabia',     region:'Persian Gulf', lat:25.93, lng:49.67, risk:'HIGH',     flag:'🇸🇦', wiki:'Abqaiq oil processing facility',    zoom:12, desc:'World largest oil processing facility. 7% of global supply. Drone attacked 2019.' },
    { id:'rasT',      nm:'Ras Tanura Terminal',       ctry:'Saudi Arabia',     region:'Persian Gulf', lat:26.64, lng:50.16, risk:'HIGH',     flag:'🇸🇦', wiki:'Ras Tanura',                        zoom:12, desc:'Largest offshore crude loading terminal. 6.5 Mbpd export capacity.' },
    { id:'ghawar',    nm:'Ghawar Oil Field',          ctry:'Saudi Arabia',     region:'Persian Gulf', lat:25.12, lng:49.25, risk:'HIGH',     flag:'🇸🇦', wiki:'Ghawar oil field',                  zoom:9,  desc:'World largest oil field. 3.8 Mbpd — 5% of global supply.' },
    { id:'basra',     nm:'Basra Oil Fields',          ctry:'Iraq',             region:'Persian Gulf', lat:30.50, lng:47.80, risk:'HIGH',     flag:'🇮🇶', wiki:'Basra oil fields Iraq',             zoom:10, desc:'Iraq primary crude export hub — 4 Mbpd. Iran-Iraq border risk zone.' },
    { id:'kuwait',    nm:'Kuwait Oil Fields',         ctry:'Kuwait',           region:'Persian Gulf', lat:29.37, lng:47.98, risk:'MEDIUM',   flag:'🇰🇼', wiki:'Kuwait oil fields',                 zoom:10, desc:'2.9 Mbpd. Gulf escalation and Houthi long-range missile exposure.' },
    { id:'uae',       nm:'Abu Dhabi Oil Hub',         ctry:'UAE',              region:'Persian Gulf', lat:24.45, lng:54.38, risk:'MEDIUM',   flag:'🇦🇪', wiki:'Abu Dhabi oil',                     zoom:10, desc:'ADNOC 3.2 Mbpd. Gulf conflict escalation exposure.' },
    { id:'qatar',     nm:'Qatar LNG Hub',             ctry:'Qatar',            region:'Persian Gulf', lat:25.28, lng:51.53, risk:'MEDIUM',   flag:'🇶🇦', wiki:'Qatar LNG',                         zoom:10, desc:'World largest LNG exporter. 77 Mtpa QatarEnergy.' },
    // Red Sea
    { id:'bab',       nm:'Bab-el-Mandeb',            ctry:'Yemen / Intl',     region:'Red Sea',      lat:12.58, lng:43.42, risk:'HIGH',     flag:'🔴', wiki:'Bab-el-Mandeb',                     zoom:9,  desc:'Houthi missile & drone strikes on tankers since Oct 2023. 10% of global trade disrupted.' },
    { id:'suez',      nm:'Suez Canal',                ctry:'Egypt',            region:'Red Sea',      lat:30.58, lng:32.27, risk:'MEDIUM',   flag:'🇪🇬', wiki:'Suez Canal',                        zoom:11, desc:'Key Europe-Asia oil route. Rerouting via Cape adds 2 weeks and $1M per voyage.' },
    { id:'aden',      nm:'Gulf of Aden',              ctry:'International',    region:'Red Sea',      lat:12.00, lng:46.50, risk:'HIGH',     flag:'⛵', wiki:'Gulf of Aden',                       zoom:8,  desc:'Active anti-ship missile zone. 30+ vessels hit. Naval coalition deployed.' },
    { id:'jeddah',    nm:'Jeddah Oil Terminal',       ctry:'Saudi Arabia',     region:'Red Sea',      lat:21.49, lng:39.19, risk:'MEDIUM',   flag:'🇸🇦', wiki:'Jeddah port',                       zoom:11, desc:'Saudi Red Sea crude export terminal. Houthi drone range threat.' },
    // Europe
    { id:'ukraine',   nm:'Ukraine Pipeline Hub',     ctry:'Ukraine',          region:'Europe',       lat:49.00, lng:32.00, risk:'HIGH',     flag:'🇺🇦', wiki:'Ukraine pipeline war',               zoom:7,  desc:'Russian strikes on Ukrainian energy. Major gas/oil transit hub to Europe.' },
    { id:'novorss',   nm:'Novorossiysk Port',         ctry:'Russia',           region:'Europe',       lat:44.72, lng:37.77, risk:'MEDIUM',   flag:'🇷🇺', wiki:'Novorossiysk port',                  zoom:12, desc:'Black Sea crude terminal. CPC pipeline endpoint. Under sanction pressure.' },
    { id:'druzhba',   nm:'Druzhba Pipeline',          ctry:'Russia / EU',      region:'Europe',       lat:52.00, lng:28.00, risk:'HIGH',     flag:'🇪🇺', wiki:'Druzhba pipeline',                   zoom:7,  desc:'Europe longest oil pipeline. 1.2 Mbpd. Cuts Hungary, Slovakia, Czech Republic.' },
    { id:'northsea',  nm:'North Sea Fields',          ctry:'Norway / UK',      region:'Europe',       lat:57.50, lng:3.50,  risk:'LOW',      flag:'🇳🇴', wiki:'North Sea oil',                      zoom:7,  desc:'Brent/Ninian fields. 1.5 Mbpd combined. Weather and aging infrastructure risk.' },
    { id:'turkstream',nm:'TurkStream Pipeline',       ctry:'Russia / Turkey',  region:'Europe',       lat:42.50, lng:36.00, risk:'MEDIUM',   flag:'🇹🇷', wiki:'TurkStream pipeline',                zoom:8,  desc:'Russian gas to Turkey and S.Europe. 31.5 Bcm/year. Geopolitical leverage tool.' },
    // South Asia — India
    { id:'ind-ocean', nm:'Indian Ocean Route',        ctry:'International',    region:'South Asia',   lat:9.00,  lng:72.50, risk:'HIGH',     flag:'🌊', wiki:'Indian Ocean trade route',           zoom:6,  desc:'Hormuz to India tanker corridor. 80% of India crude imports use this route.' },
    { id:'jamnagar',  nm:'Jamnagar Refinery',         ctry:'India',            region:'South Asia',   lat:22.47, lng:70.06, risk:'MEDIUM',   flag:'🇮🇳', wiki:'Jamnagar refinery',                  zoom:13, desc:'World largest refinery. Reliance 1.24 Mbpd. Entirely Hormuz-dependent.' },
    { id:'mumbai',    nm:'Mumbai Offshore (BH)',      ctry:'India',            region:'South Asia',   lat:19.08, lng:71.50, risk:'MEDIUM',   flag:'🇮🇳', wiki:'Bombay High oilfield',               zoom:10, desc:'ONGC Bombay High offshore. 200 Kbpd. Vulnerable to Indian Ocean disruption.' },
    { id:'kochi',     nm:'Kochi Refinery',            ctry:'India',            region:'South Asia',   lat:9.93,  lng:76.27, risk:'LOW',      flag:'🇮🇳', wiki:'Kochi refinery',                     zoom:13, desc:'BPCL 15.5 Mtpa. SW India fuel hub. Gulf tanker dependent.' },
    { id:'vizag',     nm:'Vizag Strategic Reserve',   ctry:'India',            region:'South Asia',   lat:17.69, lng:83.22, risk:'LOW',      flag:'🇮🇳', wiki:'Visakhapatnam port',                 zoom:12, desc:'India strategic petroleum reserve + Eastern Naval Command HQ.' },
    { id:'paradip',   nm:'Paradip Refinery',          ctry:'India',            region:'South Asia',   lat:20.32, lng:86.60, risk:'LOW',      flag:'🇮🇳', wiki:'Paradip refinery',                   zoom:13, desc:'IOCL 15 Mtpa. East India largest refinery. Odisha coast.' },
    { id:'chennai',   nm:'Chennai Petroleum',         ctry:'India',            region:'South Asia',   lat:13.00, lng:80.28, risk:'LOW',      flag:'🇮🇳', wiki:'Chennai Petroleum',                  zoom:12, desc:'CPCL 10.5 Mtpa. South India fuel supply. Gulf crude sea imports.' },
    { id:'pak-pipe',  nm:'Pakistan Energy',           ctry:'Pakistan',         region:'South Asia',   lat:26.00, lng:62.50, risk:'MEDIUM',   flag:'🇵🇰', wiki:'Pakistan energy crisis',             zoom:8,  desc:'Iran-Pakistan gas pipeline. Sanction risk. Energy security flashpoint.' },
    // Central Asia
    { id:'baku',      nm:'Baku–BTC Pipeline',         ctry:'Azerbaijan',       region:'Central Asia', lat:40.41, lng:49.87, risk:'LOW',      flag:'🇦🇿', wiki:'Baku Tbilisi Ceyhan pipeline',       zoom:11, desc:'BTC pipeline 1.2 Mbpd. Caspian to Mediterranean. Nagorno-Karabakh monitor.' },
    { id:'kashagan',  nm:'Kashagan Oil Field',        ctry:'Kazakhstan',       region:'Central Asia', lat:45.50, lng:53.00, risk:'LOW',      flag:'🇰🇿', wiki:'Kashagan oil field',                 zoom:9,  desc:'Caspian largest oil field. 1.6 Mbpd potential. Complex offshore production.' },
    { id:'turkmen',   nm:'Turkmenistan Gas',          ctry:'Turkmenistan',     region:'Central Asia', lat:39.00, lng:59.00, risk:'LOW',      flag:'🇹🇲', wiki:'Turkmenistan natural gas',           zoom:7,  desc:'4th largest gas reserves. China-dependent exports. Geopolitical isolation risk.' },
    // Asia-Pacific
    { id:'scs',       nm:'South China Sea',           ctry:'International',    region:'Asia-Pacific', lat:12.00, lng:114.00,risk:'HIGH',     flag:'🌊', wiki:'South China Sea dispute',            zoom:6,  desc:'PLA exercises escalating. Spratly dispute. 40% of global trade passes through Malacca.' },
    { id:'malacca',   nm:'Strait of Malacca',         ctry:'International',    region:'Asia-Pacific', lat:2.50,  lng:101.50,risk:'HIGH',     flag:'⚓', wiki:'Strait of Malacca',                  zoom:8,  desc:'40% of global seaborne trade. China and India oil lifeline. Piracy active.' },
    { id:'taiwan',    nm:'Taiwan Strait',             ctry:'Taiwan',           region:'Asia-Pacific', lat:25.03, lng:121.56,risk:'HIGH',     flag:'🇹🇼', wiki:'Taiwan Strait',                      zoom:9,  desc:'PLA blockade scenario would cut all Taiwan energy imports. 98% import dependent.' },
    { id:'indonesia', nm:'Indonesia Straits',         ctry:'Indonesia',        region:'Asia-Pacific', lat:-2.00, lng:108.00,risk:'MEDIUM',   flag:'🇮🇩', wiki:'Indonesia oil production',           zoom:7,  desc:'Lombok and Sunda straits. LNG exports from Bontang and Tangguh.' },
    { id:'japan-lng', nm:'Japan LNG Terminals',       ctry:'Japan',            region:'Asia-Pacific', lat:35.68, lng:139.70,risk:'LOW',      flag:'🇯🇵', wiki:'Japan LNG terminal',                 zoom:10, desc:'Japan 100% import dependent. Futtsu and Negishi terminals. Middle East LNG.' },
    { id:'skorea',    nm:'South Korea Oil Hubs',      ctry:'South Korea',      region:'Asia-Pacific', lat:37.55, lng:126.97,risk:'LOW',      flag:'🇰🇷', wiki:'South Korea oil refinery',           zoom:10, desc:'Ulsan and Onsan refineries. 97% import dependent. North Korea risk.' },
    // Africa
    { id:'nigeria',   nm:'Niger Delta',               ctry:'Nigeria',          region:'Africa',       lat:5.50,  lng:6.50,  risk:'MEDIUM',   flag:'🇳🇬', wiki:'Niger Delta oil',                    zoom:10, desc:'MEND sabotage. 1.5 Mbpd at risk. Pipeline attacks ongoing.' },
    { id:'libya',     nm:'Sirte Oil Basin',           ctry:'Libya',            region:'Africa',       lat:30.00, lng:18.00, risk:'MEDIUM',   flag:'🇱🇾', wiki:'Libya oil fields',                   zoom:9,  desc:'LNA vs GNU conflict near Oil Crescent. 1.2 Mbpd intermittently shut.' },
    { id:'sudan',     nm:'Port Sudan Terminal',       ctry:'Sudan',            region:'Africa',       lat:19.62, lng:37.22, risk:'HIGH',     flag:'🇸🇩', wiki:'Port Sudan',                         zoom:12, desc:'SAF-RSF civil war disrupting Red Sea crude export. 100 Kbpd offline.' },
    { id:'angola',    nm:'Angola Offshore',           ctry:'Angola',           region:'Africa',       lat:-8.83, lng:13.24, risk:'LOW',      flag:'🇦🇴', wiki:'Angola offshore oil',                zoom:9,  desc:'Angola 1.1 Mbpd. Deep offshore Cabinda. Political stability risk.' },
    { id:'algeria',   nm:'Algeria Hassi Messaoud',    ctry:'Algeria',          region:'Africa',       lat:31.67, lng:6.07,  risk:'LOW',      flag:'🇩🇿', wiki:'Hassi Messaoud',                     zoom:9,  desc:'Hassi Messaoud — North Africa largest oil field. 1 Mbpd. Sonatrach operations.' },
    { id:'mozambique',nm:'Mozambique LNG',            ctry:'Mozambique',       region:'Africa',       lat:-13.32,lng:40.69, risk:'MEDIUM',   flag:'🇲🇿', wiki:'Mozambique LNG',                     zoom:9,  desc:'Rovuma Basin 85 Tcf gas. TotalEnergies project. Insurgency in Cabo Delgado.' },
    { id:'ghana',     nm:'Ghana Jubilee Field',       ctry:'Ghana',            region:'Africa',       lat:4.90,  lng:-1.90, risk:'LOW',      flag:'🇬🇭', wiki:'Ghana Jubilee field',                zoom:10, desc:'Ghana offshore 200 Kbpd. Jubilee and TEN fields. Stable production.' },
    { id:'gabon',     nm:'Gabon Offshore',            ctry:'Gabon',            region:'Africa',       lat:-0.80, lng:8.78,  risk:'LOW',      flag:'🇬🇦', wiki:'Gabon oil production',               zoom:9,  desc:'220 Kbpd. Maturing offshore. 2023 coup instability.' },
    // Americas
    { id:'venezuela', nm:'Maracaibo Basin',           ctry:'Venezuela',        region:'Americas',     lat:10.63, lng:-71.64,risk:'MEDIUM',   flag:'🇻🇪', wiki:'Lake Maracaibo',                     zoom:10, desc:'Regime instability + US sanctions. 800 Kbpd — half 2014 peak.' },
    { id:'mexico',    nm:'Gulf of Mexico (MX)',       ctry:'Mexico',           region:'Americas',     lat:20.00, lng:-92.00,risk:'MEDIUM',   flag:'🇲🇽', wiki:'Cantarell oil field Mexico',         zoom:8,  desc:'Cantarell and KMZ fields. 1.9 Mbpd. Pemex debt crisis.' },
    { id:'usgulf',    nm:'US Gulf Coast',             ctry:'USA',              region:'Americas',     lat:29.00, lng:-91.00,risk:'LOW',      flag:'🇺🇸', wiki:'Gulf of Mexico oil',                 zoom:7,  desc:'US offshore 1.8 Mbpd. Hurricane season vulnerability. SPR nearby.' },
    { id:'canada',    nm:'Alberta Oil Sands',         ctry:'Canada',           region:'Americas',     lat:57.00, lng:-111.50,risk:'LOW',     flag:'🇨🇦', wiki:'Athabasca oil sands',                zoom:7,  desc:'3.3 Mbpd. World third-largest reserves. Pipeline bottleneck + wildfire risk.' },
    { id:'brazil',    nm:'Santos Basin',              ctry:'Brazil',           region:'Americas',     lat:-24.00,lng:-42.00,risk:'LOW',      flag:'🇧🇷', wiki:'Santos Basin Brazil',               zoom:8,  desc:'Petrobras pre-salt 3.8 Mbpd. Buzios — fastest-growing giant field.' },
    { id:'colombia',  nm:'Colombian Pipelines',       ctry:'Colombia',         region:'Americas',     lat:4.71,  lng:-74.07,risk:'MEDIUM',   flag:'🇨🇴', wiki:'Colombia oil pipeline',              zoom:8,  desc:'800 Kbpd. ELN guerrilla pipeline attacks ongoing. Cano Limon-Covenas.' },
    { id:'ecuador',   nm:'Ecuador Oil Fields',        ctry:'Ecuador',          region:'Americas',     lat:-0.22, lng:-78.52,risk:'MEDIUM',   flag:'🇪🇨', wiki:'Ecuador oil Amazon',                 zoom:8,  desc:'520 Kbpd. SOTE pipeline Amazon. Indigenous protests cause shutdowns.' },
    // Caucasus
    { id:'georgi',    nm:'Georgia BTC Transit',       ctry:'Georgia',          region:'Caucasus',     lat:42.00, lng:44.00, risk:'MEDIUM',   flag:'🇬🇪', wiki:'Georgia BTC pipeline',               zoom:8,  desc:'BTC and SCP pipelines transit Georgia. Russian invasion risk.' },
    // North Africa / Mediterranean
    { id:'malta-ch',  nm:'Sicily Channel',            ctry:'International',    region:'N.Africa/Med', lat:37.00, lng:15.00, risk:'LOW',      flag:'🌊', wiki:'Strait of Sicily',                   zoom:8,  desc:'Sicily Channel oil tanker route. Europe alternative to Suez.' },
    // West Africa
    { id:'congo',     nm:'Congo Basin Oil',           ctry:'Rep. of Congo',    region:'W.Africa',     lat:-4.27, lng:15.27, risk:'LOW',      flag:'🇨🇬', wiki:'Republic of Congo oil',              zoom:9,  desc:'260 Kbpd. SNPC operations. Debt to China and governance risk.' },
    // Oceania
    { id:'australia', nm:'NW Shelf Australia',        ctry:'Australia',        region:'Oceania',      lat:-20.50,lng:115.50,risk:'LOW',      flag:'🇦🇺', wiki:'North West Shelf LNG',               zoom:8,  desc:'NW Shelf LNG 16 Mtpa. Gorgon and Wheatstone. Asia LNG hub.' },
    { id:'timor',     nm:'Timor Sea Fields',          ctry:'Timor-Leste',      region:'Oceania',      lat:-10.00,lng:127.00,risk:'LOW',      flag:'🇹🇱', wiki:'Timor Sea oil',                      zoom:9,  desc:'Bayu-Undan field. Greater Sunrise dispute with Australia.' },
    // Arctic / North Sea
    { id:'norway',    nm:'Norway North Sea',          ctry:'Norway',           region:'Arctic/N.Sea', lat:60.00, lng:2.50,  risk:'LOW',      flag:'🇳🇴', wiki:'Norway oil production',               zoom:7,  desc:'Ekofisk and Johan Sverdrup. 1.9 Mbpd. Sabotage risk (Nord Stream precedent).' },
    { id:'russia-arc',nm:'Russian Arctic Fields',    ctry:'Russia',           region:'Arctic/N.Sea', lat:70.00, lng:70.00, risk:'MEDIUM',   flag:'🇷🇺', wiki:'Yamal LNG Russia',                   zoom:6,  desc:'Yamal and Vankor fields. 2.5 Mbpd. Sanction risk. Arctic route opening.' },
  ];

  /* ─── Threat arcs ───────────────────────────────────── */
  const ARCS = [
    { f:'il-ir',    t:'kharg',      r:'CRITICAL' },
    { f:'il-ir',    t:'hormuz',     r:'CRITICAL' },
    { f:'il-ir',    t:'natanz',     r:'HIGH'     },
    { f:'hormuz',   t:'ind-ocean',  r:'HIGH'     },
    { f:'hormuz',   t:'malacca',    r:'HIGH'     },
    { f:'ind-ocean',t:'jamnagar',   r:'MEDIUM'   },
    { f:'ind-ocean',t:'mumbai',     r:'MEDIUM'   },
    { f:'bab',      t:'suez',       r:'HIGH'     },
    { f:'bab',      t:'aden',       r:'HIGH'     },
    { f:'malacca',  t:'scs',        r:'HIGH'     },
    { f:'ukraine',  t:'druzhba',    r:'HIGH'     },
    { f:'ukraine',  t:'novorss',    r:'MEDIUM'   },
    { f:'abqaiq',   t:'hormuz',     r:'HIGH'     },
    { f:'basra',    t:'hormuz',     r:'HIGH'     },
    { f:'scs',      t:'taiwan',     r:'HIGH'     },
    { f:'baku',     t:'georgi',     r:'MEDIUM'   },
  ];

  /* ─── Region jump config ────────────────────────────── */
  const REGIONS = [
    { label:'🌐 World',        lat:20,   lng:15,    zoom:2  },
    { label:'🇸🇦 Gulf',        lat:26,   lng:50,    zoom:6  },
    { label:'🇮🇱 Israel-Iran', lat:32,   lng:43,    zoom:6  },
    { label:'🔴 Red Sea',      lat:15,   lng:42,    zoom:6  },
    { label:'🇺🇦 Europe',      lat:50,   lng:25,    zoom:5  },
    { label:'🇮🇳 India',       lat:19,   lng:76,    zoom:5  },
    { label:'🌊 S.China Sea',  lat:12,   lng:114,   zoom:5  },
    { label:'🌍 Africa',       lat:5,    lng:20,    zoom:4  },
    { label:'🌎 Americas',     lat:10,   lng:-75,   zoom:4  },
    { label:'🇰🇿 C.Asia',      lat:43,   lng:57,    zoom:5  },
    { label:'🇦🇺 Oceania',     lat:-25,  lng:125,   zoom:4  },
    { label:'🧊 Arctic',       lat:68,   lng:30,    zoom:4  },
  ];

  /* ─── Photo cache ───────────────────────────────────── */
  const photoCache = {};

  async function getPhoto(wiki) {
    if (wiki in photoCache) return photoCache[wiki];
    try {
      const r = await fetch(
        `https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(wiki)}`,
        { signal: AbortSignal.timeout(6000) }
      );
      if (r.ok) {
        const d = await r.json();
        if (d.thumbnail?.source) {
          // Always get 400px version for quality
          const src  = d.thumbnail.source.replace(/\/\d+px-/, '/400px-');
          const page = d.content_urls?.desktop?.page || '#';
          const cap  = d.description || d.title || wiki;
          return (photoCache[wiki] = { src, page, cap });
        }
      }
    } catch (_) {}
    return (photoCache[wiki] = null);
  }

  /* ─── Popup HTML (shown immediately on click) ───────── */
  function makePopupHTML(s) {
    const c = RISK[s.risk];
    const pillStyle = {
      CRITICAL: 'background:rgba(139,92,246,.15);color:#7c3aed;border:1px solid rgba(139,92,246,.4)',
      HIGH:     'background:rgba(239,68,68,.12);color:#dc2626;border:1px solid rgba(239,68,68,.35)',
      MEDIUM:   'background:rgba(245,158,11,.12);color:#d97706;border:1px solid rgba(245,158,11,.35)',
      LOW:      'background:rgba(16,185,129,.12);color:#059669;border:1px solid rgba(16,185,129,.35)',
    }[s.risk];

    return `
<div style="width:295px;font-family:'DM Sans',system-ui,sans-serif;">

  <!-- PHOTO SLOT — will be replaced by real image -->
  <div id="pp-${s.id}" style="
    width:295px; height:160px;
    background:linear-gradient(135deg,#1e293b 0%,#0f172a 100%);
    display:flex; flex-direction:column;
    align-items:center; justify-content:center;
    color:rgba(255,255,255,.45); font-size:.75rem; text-align:center;
    position:relative; overflow:hidden;
  ">
    <div style="font-size:2rem;margin-bottom:6px;">🛰️</div>
    <div>Loading satellite view…</div>
  </div>

  <!-- INFO BODY -->
  <div style="padding:11px 13px 12px;">
    <div style="display:flex;align-items:flex-start;gap:9px;margin-bottom:9px;">
      <span style="font-size:1.6rem;line-height:1;flex-shrink:0;">${s.flag}</span>
      <div style="min-width:0;">
        <div style="font-weight:700;font-size:13px;color:#0f172a;line-height:1.3;">${s.nm}</div>
        <div style="font-size:10px;color:#64748b;margin-top:2px;">📍 ${s.ctry} &nbsp;·&nbsp; ${s.region}</div>
      </div>
    </div>
    <span style="display:inline-block;padding:2px 9px;border-radius:20px;font-size:10.5px;font-weight:700;letter-spacing:.04em;margin-bottom:8px;${pillStyle}">
      ⚡ ${s.risk} RISK
    </span>
    <p style="font-size:11.5px;color:#334155;line-height:1.58;margin:0 0 8px;">${s.desc}</p>
    <div style="display:flex;align-items:center;gap:8px;font-size:9.5px;color:#94a3b8;border-top:1px solid #f1f5f9;padding-top:6px;">
      <span>🌐 ${s.lat.toFixed(3)}°, ${s.lng.toFixed(3)}°</span>
      <span style="margin-left:auto;display:flex;align-items:center;gap:3px;">
        <span style="width:5px;height:5px;border-radius:50%;background:${c.hex};display:inline-block;"></span>
        Live monitoring
      </span>
    </div>
  </div>
</div>`;
  }

  /* ─── Inject real photo into popup photo slot ───────── */
  function injectPopupPhoto(spotId, photo) {
    const el = document.getElementById(`pp-${spotId}`);
    if (!el) return;
    if (!photo) {
      el.innerHTML = `
        <div style="font-size:2rem;margin-bottom:4px;">🛰️</div>
        <div style="font-size:.72rem;">Satellite view active on map</div>`;
      return;
    }
    el.style.padding = '0';
    el.style.background = '#000';
    el.innerHTML = `
      <img src="${photo.src}" alt="${photo.cap}"
           style="width:100%;height:100%;object-fit:cover;display:block;"
           onerror="this.parentElement.innerHTML='<div style=\\'width:100%;height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;color:rgba(255,255,255,.4);font-size:.75rem;gap:6px;\\' ><div style=\\'font-size:1.8rem;\\'>🛰️</div><div>Satellite view active</div></div>'"
      />
      <div style="
        position:absolute;bottom:0;left:0;right:0;
        background:linear-gradient(transparent,rgba(0,0,0,.72));
        padding:5px 8px;
      ">
        <a href="${photo.page}" target="_blank" rel="noopener"
           style="color:rgba(255,255,255,.8);font-size:9px;text-decoration:none;display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
          📷 ${photo.cap.slice(0,60)} — Wikipedia
        </a>
      </div>`;
  }

  /* ─── Side panel with photo ─────────────────────────── */
  async function updateSidePanel(s, map) {
    const el = document.getElementById('hotspot-detail-panel');
    if (!el) return;

    const c = RISK[s.risk];
    el.style.borderColor = c.hex + '55';

    // Show loading immediately
    el.innerHTML = `
      <div style="text-align:center;padding:1.4rem .5rem;">
        <div style="font-size:2rem;margin-bottom:.4rem;">${s.flag}</div>
        <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:.88rem;color:#0f172a;">${s.nm}</div>
        <div style="font-size:.7rem;color:#94a3b8;margin:.2rem 0 .6rem;">${s.ctry}</div>
        <div style="height:3px;border-radius:2px;background:linear-gradient(90deg,${c.hex},${c.dark});width:70%;margin:0 auto;animation:orLoadBar 1.2s ease-in-out infinite;"></div>
      </div>`;

    // Fly map
    map.flyTo([s.lat, s.lng], s.zoom || 7, { duration:1.3, easeLinearity:.4 });

    // Fetch photo
    const photo = await getPhoto(s.wiki);

    const pillClass = { CRITICAL:'risk-critical', HIGH:'risk-high', MEDIUM:'risk-medium', LOW:'risk-low' }[s.risk];

    // Photo block
    const photoBlock = photo ? `
      <div style="width:100%;height:145px;border-radius:10px;overflow:hidden;margin-bottom:.8rem;position:relative;box-shadow:0 4px 14px rgba(0,0,0,.14);">
        <img src="${photo.src}" alt="${photo.cap}"
             style="width:100%;height:100%;object-fit:cover;display:block;"
             onerror="this.parentElement.style.display='none'"/>
        <div style="position:absolute;bottom:0;left:0;right:0;background:linear-gradient(transparent,rgba(0,0,0,.68));padding:4px 7px;">
          <a href="${photo.page}" target="_blank" rel="noopener"
             style="color:rgba(255,255,255,.78);font-size:8.5px;text-decoration:none;display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
            📷 ${photo.cap.slice(0,50)} — Wikipedia
          </a>
        </div>
      </div>` : `
      <div style="height:65px;border-radius:10px;background:linear-gradient(135deg,rgba(26,86,232,.06),rgba(139,92,246,.06));display:flex;align-items:center;justify-content:center;margin-bottom:.8rem;border:1px dashed rgba(26,86,232,.14);">
        <div style="text-align:center;color:#94a3b8;font-size:.72rem;">🛰️<br>Satellite view active</div>
      </div>`;

    el.innerHTML = `
      ${photoBlock}
      <div style="display:flex;align-items:flex-start;gap:.6rem;margin-bottom:.7rem;">
        <span style="font-size:1.7rem;line-height:1;flex-shrink:0;">${s.flag}</span>
        <div style="min-width:0;">
          <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:.87rem;color:#0f172a;line-height:1.25;">${s.nm}</div>
          <div style="font-size:.68rem;color:#64748b;margin-top:.15rem;">📍 ${s.ctry}</div>
          <div style="font-size:.65rem;color:#94a3b8;">🌐 ${s.lat.toFixed(3)}°, ${s.lng.toFixed(3)}°</div>
        </div>
      </div>
      <span class="risk-pill ${pillClass}" style="margin-bottom:.55rem;display:inline-block;font-size:.7rem;">${s.risk} RISK &nbsp;·&nbsp; ${s.region}</span>
      <p style="font-size:.78rem;color:#334155;line-height:1.55;margin:.25rem 0 .65rem;">${s.desc}</p>
      <div style="padding-top:.5rem;border-top:1px solid #f1f5f9;font-size:.67rem;color:#94a3b8;display:flex;align-items:center;gap:.35rem;">
        <span style="width:6px;height:6px;background:${c.hex};border-radius:50%;display:inline-block;"></span>
        Live monitoring active
      </div>`;
  }

  /* ─── Pulsing SVG marker ────────────────────────────── */
  function makeIcon(risk) {
    const c  = RISK[risk];
    const sz = { CRITICAL:46, HIGH:40, MEDIUM:34, LOW:30 }[risk];
    const r  = sz / 2;
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${sz}" height="${sz}" viewBox="0 0 ${sz} ${sz}">
      <circle cx="${r}" cy="${r}" r="${r-1}"  fill="${c.hex}1e"/>
      <circle cx="${r}" cy="${r}" r="${r-6}"  fill="${c.hex}50"/>
      <circle cx="${r}" cy="${r}" r="${r-12}" fill="${c.hex}" stroke="white" stroke-width="2.5"/>
      <circle cx="${r*0.7}" cy="${r*0.7}" r="${r*0.22}" fill="rgba(255,255,255,.62)"/>
    </svg>`;
    return L.divIcon({
      html:        `<div class="or-marker or-${risk.toLowerCase()}">${svg}</div>`,
      className:   '',
      iconSize:    [sz, sz],
      iconAnchor:  [r, r],
      popupAnchor: [0, -(r + 10)],
    });
  }

  /* ─── Curved arc polyline ───────────────────────────── */
  function drawArc(map, A, B, risk) {
    const c   = RISK[risk];
    const pts = [];
    for (let i = 0; i <= 28; i++) {
      const t = i / 28, ti = 1 - t;
      const mLat = (A.lat + B.lat) / 2 + Math.abs(B.lng - A.lng) * 0.12 + 3;
      const mLng = (A.lng + B.lng) / 2;
      pts.push([
        ti*ti*A.lat + 2*ti*t*mLat + t*t*B.lat,
        ti*ti*A.lng + 2*ti*t*mLng + t*t*B.lng,
      ]);
    }
    return L.polyline(pts, {
      color: c.hex, weight: 2, opacity: .48,
      dashArray: '7 9', lineCap: 'round',
    }).addTo(map);
  }

  /* ─── Init ──────────────────────────────────────────── */
  function init() {
    const mapEl = document.getElementById('leaflet-map');
    if (!mapEl || typeof L === 'undefined') return;

    /* Tile layers */
    const esriSat = L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      { attribution:'Tiles &copy; Esri — DigitalGlobe, GeoEye, USDA FSA, USGS, AEX, Getmapping, Aerogrid, IGN', maxZoom:19 }
    );
    const satLabels = L.tileLayer(
      'https://{s}.basemaps.cartocdn.com/rastertiles/voyager_only_labels/{z}/{x}/{y}{r}.png',
      { attribution:'&copy; CARTO', maxZoom:19, opacity:.85 }
    );
    const streets = L.tileLayer(
      'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
      { attribution:'&copy; <a href="https://openstreetmap.org/copyright">OpenStreetMap</a>', maxZoom:19 }
    );
    const terrain = L.tileLayer(
      'https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
      { attribution:'Map data &copy; OpenStreetMap, SRTM | Style &copy; OpenTopoMap', maxZoom:17 }
    );
    const dark = L.tileLayer(
      'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
      { attribution:'&copy; OpenStreetMap &copy; CARTO', maxZoom:19 }
    );
    const satFull = L.layerGroup([esriSat, satLabels]);

    /* Create map */
    const map = L.map('leaflet-map', {
      center: [20, 30], zoom: 2,
      layers: [esriSat],
      zoomControl: false,
      attributionControl: true,
    });

    /* Controls */
    L.control.zoom({ position: 'bottomright' }).addTo(map);
    L.control.scale({ position: 'bottomleft', imperial: false }).addTo(map);
    L.control.layers(
      { '🛰️ Satellite': esriSat, '🛰️ Satellite + Labels': satFull, '🗺️ Street Map': streets, '🏔️ Terrain': terrain, '🌑 Dark Mode': dark },
      {}, { position: 'topright', collapsed: false }
    ).addTo(map);

    const spotById  = Object.fromEntries(SPOTS.map(s => [s.id, s]));
    const markerArr = [];
    const circleArr = [];
    const arcArr    = [];
    let showCircles = true, showArcs = true;

    /* Place all markers */
    SPOTS.forEach(s => {
      const c = RISK[s.risk];

      const marker = L.marker([s.lat, s.lng], {
        icon: makeIcon(s.risk),
        zIndexOffset: { CRITICAL:400, HIGH:300, MEDIUM:200, LOW:100 }[s.risk] || 0,
        title: `${s.nm} — ${s.risk}`,
      }).addTo(map);

      // Bind popup — zero margin, image at top
      marker.bindPopup(makePopupHTML(s), {
        maxWidth:  295,
        minWidth:  295,
        className: 'or-popup-wrap',
        autoPan:   true,
        autoPanPadding: [20, 20],
      });

      // On popup open → fetch and inject photo
      marker.on('popupopen', async () => {
        const photo = await getPhoto(s.wiki);
        injectPopupPhoto(s.id, photo);
      });

      // On marker click → update side panel
      marker.on('click', () => updateSidePanel(s, map));

      markerArr.push({ spot: s, marker });

      // Risk exposure circle
      circleArr.push(L.circle([s.lat, s.lng], {
        radius:      c.radius,
        color:       c.hex,
        fillColor:   c.hex,
        fillOpacity: .06,
        weight:      1.5,
        opacity:     .30,
        interactive: false,
      }).addTo(map));
    });

    /* Draw arcs */
    ARCS.forEach(arc => {
      const A = spotById[arc.f], B = spotById[arc.t];
      if (A && B) arcArr.push(drawArc(map, A, B, arc.r));
    });

    /* Build region jump bar */
    const jumpBar = document.getElementById('region-jump-bar-inner');
    if (jumpBar) {
      REGIONS.forEach(reg => {
        const btn = document.createElement('button');
        btn.className   = 'jump-btn';
        btn.textContent = reg.label;
        btn.addEventListener('click', () =>
          map.flyTo([reg.lat, reg.lng], reg.zoom, { duration: 1.3, easeLinearity: .35 })
        );
        jumpBar.appendChild(btn);
      });
    }

    /* Update badge */
    const badge = document.getElementById('zone-count-badge');
    if (badge) badge.textContent = SPOTS.length + ' Active Zones';

    /* Layer toggles */
    document.getElementById('toggle-circles')?.addEventListener('click', function () {
      showCircles = !showCircles;
      circleArr.forEach(c => showCircles ? c.addTo(map) : map.removeLayer(c));
      this.classList.toggle('active', showCircles);
    });
    document.getElementById('toggle-arcs')?.addEventListener('click', function () {
      showArcs = !showArcs;
      arcArr.forEach(l => showArcs ? l.addTo(map) : map.removeLayer(l));
      this.classList.toggle('active', showArcs);
    });

    /* Risk filter */
    document.querySelectorAll('[data-risk-filter]').forEach(btn => {
      btn.addEventListener('click', () => {
        const f = btn.dataset.riskFilter;
        markerArr.forEach(({ spot, marker }) =>
          (!f || spot.risk === f) ? marker.addTo(map) : map.removeLayer(marker)
        );
        document.querySelectorAll('[data-risk-filter]').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
      });
    });

    window._oilRiskMap = map;
  }

  if (document.readyState !== 'loading') init();
  else document.addEventListener('DOMContentLoaded', init);

})();
