# Lokaler LLM-Assistent — Technisches Konzept

> **Ziel:** Ein vollständig lokal laufender persönlicher Assistent mit drei
> Kernfähigkeiten: Informationen abrufen, eine selbst gewartete Wissensbasis
> (LLM-Wiki) und Coding-Unterstützung. Kein Cloud-Zwang, vollständige
> Datenkontrolle. Englisch primär, Deutsch unterstützt.

---

## 1. Infrastruktur-Übersicht

```
┌──────────────────────────────────────┐         ┌──────────────────────────────────────────┐
│   INFERENCE NODE  ✅ ALWAYS-ON        │   LAN   │   PROXMOX VE CLUSTER  ✅ ALWAYS-ON      │
│   (Power-Cycling zurückgestellt)     │◄───────►│   Alle Services als LXC-Container       │
│                                      │         │                                          │
│  • Ollama  :11434  (GPU)             │         │  jarvis-orchestrator  :8000  ✅          │
│                                      │         │  jarvis-mcp-hub       :8080  ✅          │
│  GPU: GTX 1080 Ti, 11 GB VRAM        │         │  jarvis-webui         :3000  ✅          │
│  RAM: 32 GB                          │         │  jarvis-wake-manager  :8090  (deferred)  │
│  OS:  Ubuntu 24.04 LTS               │         │  wyoming-whisper      ← Phase 6         │
│  PBS-Client → PBS im Cluster         │         │  wyoming-piper        ← Phase 6         │
└──────────────────────────────────────┘         │                                          │
                                                  │  Bestehend:                              │
                                                  │    Reverse Proxy  :80/:443               │
                                                  │    TriliumNext (Wiki-Backend)            │
                                                  │    PBS (Proxmox Backup Server)           │
                                                  └──────────────────────────────────────────┘
                                                                  ▲
                                                                  │ Wyoming-Protokoll (Phase 6)
┌──────────────────────────────────────┐         ┌───────────────┴──────────────────────────┐
│   PI ZERO 2W + ReSpeaker  Phase 6   │────────►│   wyoming-whisper  :10300               │
│   wyoming-satellite (lokal)          │         │   wyoming-piper    :10200               │
│   Wake Word: OpenWakeWord (lokal)    │         └──────────────────────────────────────────┘
│   Mic-Array mit Beamforming          │
└──────────────────────────────────────┘
```

---

## 2. Hardware-Details

### Inference Node — GTX 1080 Ti (11 GB VRAM)

| Eigenschaft | Detail | Konsequenz |
|---|---|---|
| Architektur | Pascal (CUDA 6.1) | Kein Flash Attention (erfordert CC 8.0+) |
| Tensor Cores | Keine | Kein natives INT8/INT4 — reine CUDA-Quantisierung |
| VRAM | 11 GB GDDR5X | Modellgrenze ~10 GB nach CUDA-Overhead |
| RAM | 32 GB | RAM-Offloading möglich (langsamere Layer) |

**Gemessene Performance (Q4_K_M, auf dieser Hardware):**

| Modell | Token/s | VRAM | Bewertung |
|---|---|---|---|
| 8B | ~46 T/s | ~5.0 GB | Sehr schnell — ideal als Routing-Modell |
| 12B | ~32 T/s | ~8.1 GB | Bestes Qualitäts-/Geschwindigkeits-Verhältnis ✅ |
| Reasoning 14B | ~29 T/s | ~9.0 GB | Für Wiki-Synthese, Chain-of-Thought |
| 14B | ~27 T/s | ~8.5 GB | Knapper VRAM, kaum Qualitätsgewinn ggü. 12B |
| 30B MoE | ~19 T/s | ~17 GB | Passt nicht ins VRAM-Budget (RAM-Offloading nötig) |
| 22B+ Dense | ≤ 7 T/s | > 12 GB | Zu langsam für interaktiven Betrieb ❌ |

> Pascal ohne Tensor Cores — ca. 20–30% langsamer als RTX-Karten gleicher VRAM-Klasse.
> Ohne Flash Attention: Kontextfenster praktisch auf ≤ 8192 Token begrenzen.

**Praktische VRAM-Konfiguration (empfohlen):**

```
Option A — 12B Q4_K_M   ≈ 8.1 GB  ← Hauptmodell (bestes Verhältnis)
           KV-Cache      0.9 GB
           ──────────────────────────────
           Gesamt        9.0 GB  ← komfortabel

Option B — 8B Q4_K_M    ≈ 5.0 GB  ← Hauptmodell (schneller)
           KV-Cache      4.0 GB  ← Kontextfenster auf 8192 erweiterbar
           ──────────────────────────────
           Gesamt        9.0 GB  ← komfortabel

Option C — 8B + Routing-Modell gleichzeitig geladen
           8B   Q4_K_M  ≈ 5.0 GB
           3-4B Q4      ≈ 2.5 GB  ← kein Modell-Swap beim Routing
           KV-Cache     ≈ 2.0 GB
           ──────────────────────────────
           Gesamt       ≈ 9.5 GB  ← kein Swap-Overhead

Whisper medium  →  CPU  (~3–5 s Latenz, akzeptabel)
Kokoro TTS      →  CPU  (<0.5 s)
```

### PVE-Cluster

| Ressource | Verfügbar |
|---|---|
| Nodes | 2 |
| RAM gesamt | ≥ 20 GB |
| Bestehende Services | Reverse Proxy, TriliumNext, PBS |

**LXC-Container RAM-Budget:**

```
jarvis-orchestrator   1.5 GB
jarvis-mcp-hub        1.0 GB  (inkl. Wiki-Storage)
jarvis-webui          0.8 GB  (Docker-in-LXC)
jarvis-wake-manager   0.2 GB
──────────────────────────────
Gesamt                3.5 GB  — problemlos innerhalb 20 GB
```

---

## 3. Modellauswahl

### Primärmodelle (GTX 1080 Ti, Q4_K_M)

| Modell | VRAM | Token/s¹ | Stärke |
|---|---|---|---|
| **Gemma 3 12B Q4_K_M** *(Empfehlung, gemessen)* | ~8.1 GB | ~32 | Bestes Verhältnis auf Pascal |
| Qwen3 14B Q4_K_M | ~8.5 GB | ~27 | Gute Qualität, etwas knapper |
| Qwen3 8B Q4_K_M | ~5.0 GB | ~46 | Sehr schnell, mehr KV-Cache |
| Phi-4 14B Q4_K_M | ~8.9 GB | ~28 | Microsoft, starkes Reasoning |
| Llama 4 Scout Q4 | ~10 GB | ~18 | MoE (17B aktiv/109B gesamt) |

¹ *Gemessene Werte auf GTX 1080 Ti (Pascal, kein Flash Attention)*

### Routing / Klassifikation

| Modell | VRAM | Stärke |
|---|---|---|
| **Qwen3 8B Q4_K_M** *(Routing, gemessen ~46 T/s)* | ~5.0 GB | Schnell genug für Intent-Klassifikation |
| Phi-4-mini 3.8B Q4 | ~2.5 GB | Schlägt kleinere Modelle bei Reasoning |
| Qwen3 1.7B Q4 | ~1.2 GB | Sehr schnell, ausreichend für einfaches Routing |

### Reasoning / Wiki-Synthese

| Modell | VRAM | Stärke |
|---|---|---|
| **DeepSeek-R1-Distill 14B Q4** *(gemessen ~29 T/s)* | ~9.0 GB | Chain-of-Thought, ideal für wiki_ingest / wiki_query |

### Code-Modelle

| Modell | VRAM | Stärke |
|---|---|---|
| **Qwen2.5-Coder 7B Q4_K_M** *(Empfehlung)* | ~4.8 GB | #1 HumanEval in der 7B-Klasse |
| DeepSeek-Coder V2 Lite Q4 | ~5.0 GB | Stärker bei Multi-File-Coding |

### Spezialmodelle

| Modell | Aufgabe | Läuft auf |
|---|---|---|
| Whisper medium | Speech-to-Text, DE+EN | Inference Node, CPU |
| Kokoro / Piper | Text-to-Speech, DE+EN | Inference Node, CPU |
| OpenWakeWord | Wake-Word-Erkennung | Pi Zero 2W, CPU |
| nomic-embed-text | Embeddings (optional) | Inference Node, Ollama |

### Quantisierungs-Kurzreferenz

```
Q4_K_M  → ~75% VRAM-Ersparnis, ~95% Qualität  ← Standard
Q5_K_M  → ~70% VRAM-Ersparnis, ~97% Qualität  ← wenn Platz reicht
Q8_0    → ~50% VRAM-Ersparnis, ~99% Qualität  ← nur für RAM-Offloading
```

---

## 4. Die drei Kernfähigkeiten

### 4.1 Informationen abrufen

Über den MCP-Hub Zugriff auf:
- **Web-Suche** (SearXNG self-hosted oder Brave Search MCP)
- **Wetter** (offene APIs, kein API-Key nötig)
- **Kalender / Erinnerungen** (lokale ICS-Datei oder Nextcloud)
- **News** (RSS-Feed-MCP)

### 4.2 LLM-Wiki (Karpathy-Muster)

Selbst gewartete Markdown-Wiki statt klassischem RAG. Details in Abschnitt 5.

### 4.3 Coding-Assistent

Über den MCP-Hub Zugriff auf:
- **Filesystem-MCP** (Dateien lesen/schreiben/suchen)
- **Shell-MCP** (Befehle ausführen, Tests laufen lassen)
- **Git-MCP** (Commits, Diffs, Log)
- Optimales Modell: **Qwen2.5-Coder 7B Q4_K_M**

---

## 5. LLM-Wiki (Karpathy-Muster) — Kernarchitektur

### Konzept: Wiki statt RAG

| | Klassisches RAG | LLM-Wiki |
|---|---|---|
| Speicher | Vektordatenbank (Embeddings) | Markdown-Dateien (lesbar) |
| Abfrage | Embedding-Suche → Chunk-Rückgabe | LLM liest Index → wählt Seiten → synthetisiert |
| Wartung | Keine | LLM aktualisiert Seiten bei jedem Ingest |
| Konsistenz | Keine (Chunks isoliert) | Cross-References, Widerspruchs-Erkennung |
| Token-Effizienz | ~70× mehr Tokens pro Query | Kompakte Wiki-Seiten, wenige Tokens |

### Die Herausforderung: kleines Kontextfenster

Das ist die zentrale Einschränkung beim Betrieb auf Consumer-Hardware ohne Flash Attention:
das effektive Kontextfenster liegt bei **≤ 8 192 Token**. Das formt die gesamte Wiki-Architektur:

- **Atomare Seiten**: Eine Seite = ein Konzept, angestrebt ≤ 400 Tokens — das LLM kann viele davon auf einmal überblicken
- **Index-first**: Das LLM liest zuerst den kompakten Index, wählt dann gezielt 1–3 Seiten
  statt alle Wiki-Inhalte blind in den Kontext zu laden
- **Kompakter Index**: Format `path | title` (~50 Zeichen/Eintrag) hält den Index unter ~300 Zeilen nutzbar;
  der vollständige Seiteninhalt wird erst auf Anfrage geladen
- **Lint-Modus**: Prüft Konsistenz auf maximal 20 Seiten gleichzeitig (begrenzt durch das Kontextfenster)
- **Langfristig (>300 Seiten)**: Relevanz-basiertes Index-Filtering nötig,
  ähnlich wie `_select_pages_from_index` — sonst läuft der Index selbst über das Kontextfenster

> Bei einem größeren Kontextfenster (z.B. durch Flash Attention auf RTX-Karten) ließe sich der
> vollständige Index und mehrere Wiki-Seiten gleichzeitig laden — das würde die Index-first-Architektur
> vereinfachen, ist auf Pascal-Hardware aber nicht realistisch.

### Verzeichnisstruktur

```
/wiki-data/
├── raw/                    ← Quelldokumente (unveränderlich)
│   ├── artikel_2026-05.pdf
│   └── ...
├── wiki/                   ← LLM-gepflegte Seiten (Markdown)
│   ├── index.md            ← Inhaltsverzeichnis aller Seiten
│   ├── log.md              ← Append-only Protokoll (Ingests, Queries)
│   ├── python_asyncio.md
│   └── ...
└── schema/
    └── AGENTS.md           ← Konventionen & Formatregeln für das LLM
```

### Drei Operationen

**`wiki ingest <quelldatei>`**
```
1. LLM liest Quelldokument vollständig
2. Extrahiert Konzepte, Entities, Kernaussagen
3. Erstellt oder aktualisiert 10–15 Wiki-Seiten
4. Fügt Wikilinks [[andere-seite]] ein
5. Aktualisiert index.md und log.md
```

**`wiki query "<frage>"`**
```
1. LLM liest index.md  (Übersicht aller Seiten)
2. Wählt 3–7 relevante Seitennamen
3. Lädt diese Seiten in den Kontext
4. Synthetisiert Antwort (optional: speichert als neue Seite)
```

**`wiki lint`**
```
Periodisch: findet verwaiste Seiten, Widersprüche,
fehlende Cross-References — LLM bereinigt selbst
```

### Implementierung als MCP-Tool

```python
# mcp_hub/tools/wiki.py
from pathlib import Path
from mcp.server import Server
from mcp.types import Tool, TextContent

WIKI_PATH = Path("/data/wiki")

@server.call_tool()
async def wiki_ingest(source_path: str) -> str:
    source = Path(source_path).read_text()
    index = (WIKI_PATH / "wiki/index.md").read_text()

    prompt = f"""Du bist der Wiki-Maintainer. Hier ist ein neues Dokument:

{source}

Aktuelle Wiki-Seiten (index.md):
{index}

Erstelle oder aktualisiere die relevanten Wiki-Seiten.
Format: === wiki/seitenname.md ===\n<inhalt>"""

    response = await ollama_generate(model="<reasoning-model>", prompt=prompt)
    _write_wiki_pages(response)
    _append_log(f"INGEST {source_path}")
    return f"Wiki aktualisiert: {source_path}"


@server.call_tool()
async def wiki_query(question: str) -> str:
    index = (WIKI_PATH / "wiki/index.md").read_text()

    # Schritt 1: Relevante Seiten identifizieren (kleines Routing-Modell)
    pages_prompt = f"index.md:\n{index}\n\nFrage: {question}\n\nWelche 3–5 Seiten sind relevant? Nur Dateinamen."
    relevant = await ollama_generate(model="<routing-model>", prompt=pages_prompt)

    # Schritt 2: Seiten laden und Antwort synthetisieren
    pages_content = _load_pages(relevant)
    answer_prompt = f"Wiki-Inhalt:\n{pages_content}\n\nFrage: {question}\n\nAntwort:"
    return await ollama_generate(model="<main-model>", prompt=answer_prompt)
```

---

## 6. Vollständige System-Architektur

```
 NUTZER (Sprache oder Browser)
  │
  ▼
┌──────────────────────────────────────────────────────────────────┐
│  INFERENCE NODE (bare metal, Ubuntu 24.04)                       │
│                                                                  │
│  [OpenWakeWord] ──► "Hey Jarvis" ──► [Whisper medium / CPU]      │
│                                              │ Text              │
│                                              ▼                   │
│              HTTP POST ──────────────────────────────────────► PVE│
│                                                                  │
│  PVE-Response ──► [TTS / CPU] ──► Lautsprecher                   │
│                                                                  │
│  [Ollama :11434 / GPU]                                           │
│    <main-model>     (Haupt-LLM)                                  │
│    <code-model>     (Code)                                       │
│    <routing-model>  (Intent-Routing)                             │
│    nomic-embed-text (Embeddings, optional)                       │
└──────────────────────────────────────────────────────────────────┘
                              LAN ▲▼
┌──────────────────────────────────────────────────────────────────┐
│  PVE CLUSTER                                                     │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  jarvis-orchestrator (LangGraph, Python 3.12)              │ │
│  │                                                            │ │
│  │  Eingabe → Intent Router (<routing-model> via Ollama)      │ │
│  │  → "Info?"    → Web-Search MCP                             │ │
│  │  → "Wiki?"    → Wiki-Query MCP                             │ │
│  │  → "Code?"    → Ollama <code-model> + Filesystem MCP       │ │
│  │  → "Direkt?"  → Ollama <main-model>                        │ │
│  └───────────────────────┬────────────────────────────────────┘ │
│                           │                                      │
│         ┌─────────────────┼──────────────────┐                  │
│         ▼                 ▼                  ▼                  │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ jarvis-mcp  │  │  jarvis-     │  │  jarvis-wake │          │
│  │    -hub     │  │  webui       │  │  -manager    │          │
│  │             │  │  :3000       │  │  :8090       │          │
│  │ Tools:      │  │  Chat-UI     │  │              │          │
│  │ • wiki_*    │  └──────────────┘  └──────────────┘          │
│  │ • web_search│                                               │
│  │ • filesystem│                                               │
│  │ • shell     │                                               │
│  │ • git       │                                               │
│  │ • calendar  │                                               │
│  │ • rss_feed  │                                               │
│  └─────────────┘                                               │
│                                                                  │
│  [Bestehend]  Reverse Proxy → <assistant>.home, api.<assistant>.home │
└──────────────────────────────────────────────────────────────────┘
```

---

## 7. Power-Management

### 7.1 Hinweis: WoL vs. Smart Plug

Wake-on-LAN ist die elegante Lösung — setzt aber voraus, dass NIC und BIOS es wirklich unterstützen. Das ist nicht selbstverständlich. Vor der Planung prüfen:

```bash
ethtool <iface> | grep "Wake-on"   # muss "g" zeigen, nicht "d"
```

Wenn WoL nicht verfügbar ist: **Smart Plug als Alternative.**
Im BIOS `Restore AC Power Loss → Power On` setzen. Der Rechner startet automatisch, sobald die Steckdose Strom gibt. Herunterfahren per Shutdown-Agent (API-Call).

**WoL (wenn verfügbar) — Ubuntu, persistent via udev:**

```bash
# Sofort aktivieren
sudo ethtool -s <iface> wol g

# Permanent
echo 'ACTION=="add", SUBSYSTEM=="net", NAME=="<iface>", RUN+="/sbin/ethtool -s <iface> wol g"' \
     | sudo tee /etc/udev/rules.d/81-wol.rules
```

> Feste IP oder DHCP-Reservation für den Inference Node ist in beiden Fällen nötig.

### 7.2 Wake-Manager (PVE LXC, always-on)

```
jarvis-wake-manager  :8090
├── POST /wake          → startet Inference Node, wartet bis bereit
├── POST /shutdown      → fährt Inference Node sauber herunter
├── GET  /status        → "online" | "offline" | "booting"
└── GET  /health        → Liveness-Check des Wake-Managers selbst
```

```python
# wake_manager/main.py
import asyncio, httpx
from fastapi import FastAPI, HTTPException
from wakeonlan import send_magic_packet  # oder: Smart-Plug-API-Call

app = FastAPI()

INFERENCE_MAC  = "<MAC-Adresse>"          # für WoL
INFERENCE_IP   = "<Inference-Node-IP>"    # feste IP / DHCP-Reservation
OLLAMA_URL     = f"http://{INFERENCE_IP}:11434"
BOOT_TIMEOUT   = 120   # Sekunden
IDLE_SHUTDOWN  = 1800  # 30 min Inaktivität → Auto-Shutdown


async def ollama_reachable() -> bool:
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get(f"{OLLAMA_URL}/api/tags")
            return r.status_code == 200
    except Exception:
        return False


@app.post("/wake")
async def wake():
    if await ollama_reachable():
        return {"status": "already_online"}

    send_magic_packet(INFERENCE_MAC)  # oder: Smart-Plug einschalten

    for _ in range(BOOT_TIMEOUT // 5):
        await asyncio.sleep(5)
        if await ollama_reachable():
            await _warmup_models()
            return {"status": "online"}

    raise HTTPException(503, "Inference Node nicht erreichbar nach Boot-Timeout")


@app.post("/shutdown")
async def shutdown():
    async with httpx.AsyncClient(timeout=5) as c:
        await c.post(f"http://{INFERENCE_IP}:8090/shutdown")
    return {"status": "shutdown_sent"}


@app.get("/status")
async def status():
    return {"status": "online" if await ollama_reachable() else "offline"}


async def _warmup_models():
    async with httpx.AsyncClient(timeout=60) as c:
        await c.post(f"{OLLAMA_URL}/api/generate", json={
            "model": "<main-model>",
            "prompt": "",
            "keep_alive": "1h"
        })
```

### 7.3 Auto-Shutdown nach Inaktivität

```bash
# /opt/jarvis/check_idle.sh
#!/bin/bash
ACTIVE=$(curl -s http://localhost:11434/api/ps | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(len(data.get('models', [])))
")
if [ "$ACTIVE" -eq "0" ]; then
    LAST_FILE="/tmp/jarvis_last_active"
    NOW=$(date +%s)
    if [ ! -f "$LAST_FILE" ]; then
        echo $NOW > "$LAST_FILE"
    else
        LAST=$(cat "$LAST_FILE")
        DIFF=$((NOW - LAST))
        if [ $DIFF -gt 1800 ]; then   # 30 Minuten
            rm "$LAST_FILE"
            logger "Jarvis: Inference Node shutting down (idle)"
            shutdown -h now
        fi
    fi
else
    date +%s > /tmp/jarvis_last_active
fi
```

```ini
# /etc/systemd/system/jarvis-idle-check.timer
[Timer]
OnBootSec=15min
OnUnitActiveSec=15min

[Install]
WantedBy=timers.target
```

### 7.4 Startup-Ablauf (Ende-zu-Ende)

```
Nutzer: "Jarvis, starte das System"   (oder API: POST /wake)
  │
  ▼
jarvis-orchestrator (PVE, always-on)
  │  erkennt "wake"-Intent
  ▼
jarvis-wake-manager POST /wake
  │
  ├─ WoL / Smart Plug → Inference Node einschalten
  │
  ├─ Inference Node bootet (~30–60 s)
  │   └─ systemd startet Ollama, Shutdown-Agent automatisch
  │
  ├─ Wake-Manager pollt GET /api/tags alle 5 s
  │
  ├─ Ollama antwortet → Modell vorwärmen (~15 s)
  │
  └─ Antwort: "System bereit."

Typische Zeiten:
  POST-Screen          ~10 s
  Ubuntu-Boot          ~20–30 s
  Ollama startet       ~5 s
  Modell in VRAM       ~10–15 s
  ─────────────────────────────
  Gesamt bis bereit    ~50–60 s
```

---

## 8. Backup-Strategie (PBS)

Alle Komponenten werden über den bestehenden **Proxmox Backup Server** gesichert.

### Inference Node: PBS-Client (Datei-Backup)

```bash
# PBS-Client-Repository (Ubuntu 24.04 / Bookworm-Packages)
wget https://enterprise.proxmox.com/debian/proxmox-release-bookworm.gpg \
     -O /etc/apt/trusted.gpg.d/proxmox-release-bookworm.gpg

echo "deb [arch=amd64] http://download.proxmox.com/debian/pbs-client bookworm main" \
     | sudo tee /etc/apt/sources.list.d/pbs-client.list

apt update && apt install -y proxmox-backup-client
```

> Kein libssl1.1-Workaround nötig bei Ubuntu 24.04 (im Gegensatz zu 22.04).

**Was gesichert wird:**

| Pfad | Archiv-Name | Sichern? | Begründung |
|---|---|---|---|
| `/etc` | `etc.pxar` | **Ja** | Ollama-Service, Netzwerk-Config |
| `/opt/jarvis` | `jarvis.pxar` | **Ja** | Voice-Pipeline, eigene Skripte |
| `/home/<user>` | `home.pxar` | **Ja** | Dotfiles, SSH-Keys |
| `<ollama-models-path>` | — | **Nein** | 5–15 GB, jederzeit mit `ollama pull` neu ladbar |

```ini
# /etc/systemd/system/pbs-backup.service
[Unit]
Description=PBS Backup Inference Node

[Service]
Type=oneshot
Environment="PBS_REPOSITORY=<user>@pbs!<token-id>@<pbs-ip>:<datastore>"
Environment="PBS_ENCRYPTION_PASSWORD=<key>"
ExecStart=/usr/bin/proxmox-backup-client backup \
  etc.pxar:/etc \
  jarvis.pxar:/opt/jarvis \
  home.pxar:/home/<user> \
  --backup-id inference-node \
  --change-detection-mode=metadata
```

```ini
# /etc/systemd/system/pbs-backup.timer
[Timer]
OnCalendar=*-*-* 03:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

**Empfohlene Retention-Policy:**

```
Täglich:     7 Versionen
Wöchentlich: 4 Versionen
Monatlich:   3 Versionen
```

---

## 9. PVE-Container (LXC)

| Container | RAM | Stack | Port | Beschreibung |
|---|---|---|---|---|
| `jarvis-orchestrator` | 1.5 GB | Python 3.12, LangGraph, FastAPI | 8000 | Kernlogik, Intent Routing |
| `jarvis-mcp-hub` | 1.0 GB | Python, MCP SDK, Wiki-Storage | 8080 | Alle Tools gebündelt |
| `jarvis-webui` | 0.8 GB | Docker-in-LXC, Open WebUI | 3000 | Chat-Frontend |
| `jarvis-wake-manager` | 0.2 GB | Python, FastAPI, wakeonlan | 8090 | Power-Management |
| `jarvis-voice` *(Phase 6)* | 1.0 GB | Docker, wyoming-whisper + wyoming-piper | 10300/10200 | STT + TTS |

### Hinweis: Docker auf ZFS-LXC

overlay2 und fuse-overlayfs schlagen auf ZFS fehl. Lösung: VFS-Storage-Driver.

```json
// /etc/docker/daemon.json
{ "storage-driver": "vfs" }
```

Runtime-Performance ist nicht betroffen — nur Image-Pull ist langsamer.
LXC muss als **privileged** laufen. Empfohlene Mindestgröße: **20 GB Disk**.

### Routing (bestehender Reverse Proxy)

```
<assistant>.home          →  jarvis-webui:3000          (Chat-UI)
api.<assistant>.home      →  jarvis-orchestrator:8000   (REST API)
# Ollama + Wyoming niemals nach außen — nur intern im LAN!
```

---

## 10. Sprach-Pipeline

### 10.1 Phase 5 — ❌ GESTRICHEN

Alexa Custom Skill wurde gestrichen. Wake Word, STT und TTS laufen über Amazon-Server —
das widerspricht dem No-Cloud-Prinzip des Projekts (vollständige Datenkontrolle).
Ersatz: Phase 6 Wyoming Satellite (vollständig lokal, ohne Cloud-Abhängigkeit).

---

### 10.2 Phase 6 — Pi Zero 2W Satellite (vollständig lokal)

```
Pi Zero 2W + ReSpeaker 2-Mic HAT
  │
OpenWakeWord "Hey Jarvis"   lokal auf Pi, CPU, <50 MB
  │
wyoming-satellite           Audio-Stream via Wyoming-Protokoll
  │  (LAN)
wyoming-faster-whisper      Docker auf PVE  :10300   ~2–4 s DE+EN
  │ Text
jarvis-orchestrator         PVE  :8000
  │
LLM (Ollama, Inference Node)     ~5–15 s
  │ Text
wyoming-piper               Docker auf PVE  :10200   <0.5 s DE-Stimme
  │ Audio-Stream
Pi → Lautsprecher

Gesamt: ~8–20 s — vergleichbar mit Alexa, 100% lokal
```

```bash
# wyoming-satellite auf dem Pi
pip install wyoming-satellite

wyoming-satellite \
  --name "jarvis-satellite" \
  --uri "tcp://0.0.0.0:10700" \
  --mic-command "arecord -r 16000 -c 1 -f S16_LE -t raw" \
  --snd-command "aplay -r 22050 -c 1 -f S16_LE -t raw" \
  --wake-word-name "hey_jarvis" \
  --wyoming-whisper-uri "tcp://<pve-ip>:10300" \
  --wyoming-piper-uri "tcp://<pve-ip>:10200"
```

```yaml
# docker-compose.yml (Wyoming-Server auf PVE)
services:
  wyoming-whisper:
    image: rhasspy/wyoming-faster-whisper
    ports: ["10300:10300"]
    volumes: ["./whisper-data:/data"]
    command: --model medium --language de --uri tcp://0.0.0.0:10300

  wyoming-piper:
    image: rhasspy/wyoming-piper
    ports: ["10200:10200"]
    volumes: ["./piper-data:/data"]
    command: >
      --piper /usr/local/bin/piper
      --voice de_DE-thorsten-high
      --uri tcp://0.0.0.0:10200
```

**Hardware-Einkaufsliste (Phase 6):**

| Teil | Modell | Preis |
|---|---|---|
| SBC | Raspberry Pi Zero 2W | ~18 € |
| Mikrofon | ReSpeaker 2-Mic HAT | ~15 € |
| SD-Karte | 16 GB microSD | ~5 € |
| Lautsprecher | Mini USB / 3.5mm Speaker | ~10 € |
| **Gesamt** | | **~48 €** |

---

## 11. Phasenplan

### Phase 1 — Inference Node & Ollama ✅

- Ubuntu 24.04 installieren, NVIDIA-Treiber + CUDA ≥ 12.1
- Ollama installieren, im LAN exponieren (`OLLAMA_HOST=0.0.0.0`)
- Modelle laden, Performance messen (`ollama run <model> --verbose`)
- Modellpfad auf separate Partition/SSD auslagern (`OLLAMA_MODELS=<pfad>`)
- Hauptmodell wählen (anhand gemessener Token/s, nicht aus Benchmarks)
- PBS-Backup einrichten, ersten Backup-Lauf verifizieren
- Power-Management (WoL oder Smart Plug) einrichten
- Shutdown-Agent als systemd-Service einrichten
- Idle-Check-Timer einrichten (Auto-Shutdown nach 30 min Inaktivität)

### Phase 2 — PVE-Infrastruktur (Open WebUI) ✅

- LXC `jarvis-webui` erstellen (privileged, ≥20 GB, Docker)
- Docker VFS-Storage-Driver konfigurieren (ZFS-Kompatibilität)
- Open WebUI gegen Ollama-IP konfigurieren
- Reverse Proxy: `<assistant>.home` → WebUI
- Browser-Chat-Test

### Phase 3 — Orchestrator & MCP-Tools ✅

- LXC `jarvis-orchestrator`: Python 3.12, LangGraph, FastAPI — Port 8000
- LXC `jarvis-mcp-hub`: FastAPI, ddgs, httpx — Port 8080
- LXC `jarvis-webui`: Open WebUI → zeigt auf Orchestrator (Modell "assistant")
- Intent-Router: Keyword-Vorfilter + Routing-Modell-Fallback (chat / web_search / wiki_query / wiki_lint / wiki_ingest / code)
- Web-Suche: `ddgs` (`from ddgs import DDGS`) — kein API-Key nötig
- Streaming: `/api/chat` Ollama-kompatibler Endpoint mit StreamingResponse
- Wake-Manager: dauerhaft zurückgestellt — Inference Node bleibt always-on (Hardware-Schutz)

### Phase 4 — LLM-Wiki ✅

- Wiki-Backend: TriliumNext ETAPI (Standalone Binary, kein Docker)
- `wiki_query` MCP-Tool: Routing-Modell extrahiert Suchbegriffe → Index-first → Hauptmodell
  - Multi-Keyword: bis zu 3 Suchbegriffe → separate Suchen → dedupliziert nach Pfad
- `wiki_ingest` MCP-Tool: POST /tools/wiki_ingest auf MCP-Hub → Trilium ETAPI
  - Karpathy-Muster: liest Index + verwandte Seiten, integriert mit [[wikilinks]], aktualisiert Index
- `wiki_lint` MCP-Tool: drei Modi (index / links / full) — alle als Chat-Intents
  - `index`: baut `schema/index` aus allen Seiten (kein LLM)
  - `links`: prüft alle [[wikilinks]] auf Existenz (kein LLM)
  - `full`: links + LLM-Konsistenzcheck (begrenzt auf 20 Seiten — Kontextfenster)
- `schema/AGENTS.md` Konventionsseite anlegen
- Code-Modell: Qwen2.5-Coder 7B für Intent "code"
- DIRECT_INTENTS: wiki_lint + wiki_ingest geben Tool-Ergebnis direkt zurück — kein LLM-Aufruf

### Phase 4.5 — Multi-Agent Architektur (geplant)

Ziel: Den Orchestrator von hardcoded Keyword-Routing auf ein dynamisches, Wiki-gesteuertes Agent-System umbauen.
Analog zu CLAUDE.md: Agent-Definitionen als Wiki-Seiten, kein Code-Change für neue Agents.

```
schema/core          ← globaler Systemprompt (Charakter, Constraints) — in jeden LLM-Call injiziert
schema/agents/wiki   ← Modell: 12B · Trigger-Keywords · Tool: wiki_*
schema/agents/code   ← Modell: Code-Modell · Trigger-Keywords
schema/agents/chat   ← Modell: Routing-Modell · Fallback
schema/agents/search ← Modell: 12B · Tool: web_search
```

- `intent.py` liest Agent-Definitionen dynamisch aus Trilium statt hardcoded Listen
- Neuen Agent hinzufügen = neue Wiki-Seite anlegen, kein Code-Change
- Dokument-Import Feature: Webpage/PDF → atomare Wiki-Notizen (neues MCP Tool)

### Phase 5 — ❌ GESTRICHEN

Alexa Custom Skill wurde gestrichen — Daten laufen über Amazon-Server (widerspricht No-Cloud-Prinzip).

### Phase 6 — Pi Voice Satellite (vollständig lokal)

- Hardware kaufen und zusammenbauen (~48 €)
- Raspberry Pi OS Lite 64-bit, SSH
- LXC `jarvis-voice`: wyoming-whisper `:10300`, wyoming-piper `:10200`
- `wyoming-satellite` auf Pi konfigurieren
- Orchestrator: Wyoming-Intent-Handler implementieren
- End-to-End-Test: „Hey Jarvis, ..." ohne Amazon Cloud
- Echo Dot ablösen

---

## 12. Technologie-Stack

| Schicht | Technologie |
|---|---|
| LLM Runtime | **Ollama** (GPU, Inference Node) |
| Hauptmodell | **12B Q4_K_M** (oder 8B für mehr Geschwindigkeit) |
| Reasoning-Modell | **14B Distill Q4** (Wiki-Synthese, Chain-of-Thought) |
| Routing-Modell | **8B Q4_K_M** (Intent-Klassifikation, schnell) |
| Code-Modell | **Qwen2.5-Coder 7B Q4_K_M** |
| Orchestrierung | **LangGraph (Python 3.12)** |
| Tool-Protokoll | **Python MCP SDK** |
| LLM-Wiki Backend | **TriliumNext** (ETAPI, Standalone Binary) |
| LLM-Wiki Logik | **Custom (Karpathy-Muster)** als MCP-Tool |
| Voice Phase 6 | **Wyoming Satellite (Pi Zero 2W + ReSpeaker)** |
| STT Phase 6 | **wyoming-faster-whisper** (Docker, PVE) |
| TTS Phase 6 | **wyoming-piper** (Docker, PVE, DE: thorsten-high) |
| Wake Word | **OpenWakeWord** (Pi Zero 2W, CPU) |
| Chat-Frontend | **Open WebUI** (Docker-in-LXC) |
| Routing | **Bestehender Reverse Proxy** (PVE) |
| Backup | **proxmox-backup-client** → PBS |
| Power-Management | **Wake-on-LAN oder Smart Plug + Wake-Manager** |

---

*Hardware-Anforderungen: GPU ≥ 8 GB VRAM (CUDA), ≥ 16 GB RAM · PVE-Cluster mit ≥ 20 GB RAM*
