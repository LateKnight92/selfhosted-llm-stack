# Local LLM Stack

Ein vollständig lokal laufender persönlicher Assistent — kein Cloud-Zwang, vollständige Datenkontrolle.

## Was ist das?

Dieser Stack verbindet einen GPU-Rechner mit einer bestehenden Proxmox-Infrastruktur zu einem
privaten KI-Assistenten. Anfragen werden lokal verarbeitet, keine Daten verlassen das Heimnetz.

Der Assistent beherrscht drei Kernfähigkeiten:

- **Informationen abrufen** — Web-Suche (DuckDuckGo, kein API-Key nötig)
- **Persönliches Wissen** — eigene Wiki-Seiten lesen und befüllen (LLM-Wiki nach Karpathy-Muster)
- **Code** — schreiben, erklären, debuggen mit einem spezialisierten Code-Modell

Der Intent-Router erkennt automatisch, welche Fähigkeit gefragt ist, und leitet die Anfrage
an das passende Modell weiter.

## Architektur

```
NUTZER (Browser oder Sprache)
  │
  ▼
┌─────────────────────────────────────┐      ┌─────────────────────────────────┐
│  INFERENCE NODE (GPU, always-on)    │ LAN  │  PROXMOX CLUSTER (always-on)   │
│                                     │◄────►│                                 │
│  Ollama :11434                      │      │  jarvis-orchestrator  :8000     │
│    Hauptmodell (12B Q4_K_M)         │      │  jarvis-mcp-hub       :8080     │
│    Code-Modell (Coder 7B Q4_K_M)    │      │  jarvis-webui         :3000     │
│    Routing-Modell (8B Q4_K_M)       │      │  TriliumNext          :8080     │
└─────────────────────────────────────┘      └─────────────────────────────────┘
```

Der Orchestrator empfängt Anfragen, klassifiziert den Intent und ruft bei Bedarf
Web-Suche oder Wiki-Abfrage über den MCP-Hub ab — bevor das Hauptmodell antwortet.

Details zur Architektur, Modellauswahl und allen Phasen: [KONZEPT.md](KONZEPT.md)

## Voraussetzungen

**Inference Node (GPU-Rechner)**
- GPU mit ≥ 8 GB VRAM und CUDA-Unterstützung
- ≥ 16 GB RAM
- Ubuntu 22.04 oder 24.04 LTS
- [Ollama](https://ollama.com) installiert und im LAN exponiert

**Proxmox-Cluster**
- ≥ 20 GB freier RAM für die LXC-Container
- Bestehender Reverse Proxy (empfohlen)
- [TriliumNext](https://github.com/TriliumNext/Notes) (optional, für LLM-Wiki)

**Modelle (werden mit `ollama pull` geladen)**
```bash
ollama pull gemma3:12b          # Hauptmodell
ollama pull qwen3:8b            # Routing-Modell
ollama pull qwen2.5-coder:7b    # Code-Modell
```

## Quick Start

Wer schnell loslegen will — ohne systemd-Services und Reverse Proxy:

```bash
# 1. Repo klonen
git clone https://github.com/<user>/selfhosted-llm-stack.git
cd selfhosted-llm-stack

# 2. Konfiguration anlegen
cp .env.example .env
nano .env   # OLLAMA_URL und MCP_HUB_URL eintragen

# 3. MCP-Hub starten (Terminal 1)
cd jarvis-mcp-hub
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8080

# 4. Orchestrator starten (Terminal 2)
cd jarvis-orchestrator
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000

# 5. Testen
curl http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Was ist der Unterschied zwischen RAG und LLM-Wiki?"}'
```

Für den Produktivbetrieb (systemd-Services, Open WebUI, Proxmox-Container) siehe die vollständige Anleitung unten.

---

## Aufsetzen

### 1. Ollama auf dem Inference Node

Ollama muss im LAN erreichbar sein:

```bash
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf << 'EOF'
[Service]
Environment="OLLAMA_HOST=0.0.0.0"
EOF
sudo systemctl daemon-reload && sudo systemctl restart ollama
```

Modelle auf eine separate Partition/SSD auslagern (empfohlen bei großen Modellen):

```bash
# In /etc/systemd/system/ollama.service.d/override.conf ergänzen:
Environment="OLLAMA_MODELS=/mnt/models/ollama"
```

### 2. LXC-Container auf Proxmox

Vier Container werden benötigt (Debian 12, unprivileged, je nach Bedarf):

| Hostname | RAM | Disk | Port |
|---|---|---|---|
| `jarvis-orchestrator` | 1.5 GB | 8 GB | 8000 |
| `jarvis-mcp-hub` | 1.0 GB | 8 GB | 8080 |
| `jarvis-webui` | 0.8 GB | 20 GB | 3000 |
| `jarvis-trilium` | 0.5 GB | 10 GB | 8080 |

> **Docker auf ZFS:** overlay2 schlägt auf ZFS fehl. In `/etc/docker/daemon.json`
> den VFS-Storage-Driver setzen: `{ "storage-driver": "vfs" }`.
> Der Container muss als **privileged** laufen. Mindestens 20 GB Disk für den WebUI-Container.

### 3. Open WebUI

Im `jarvis-webui`-Container:

```bash
curl -fsSL https://get.docker.com | sh
docker run -d --name open-webui --restart always \
  -p 3000:8080 \
  -e OLLAMA_BASE_URL=http://<orchestrator-ip>:8000 \
  -v open-webui:/app/backend/data \
  ghcr.io/open-webui/open-webui:main
```

Open WebUI zeigt auf den Orchestrator (nicht direkt auf Ollama) — so läuft
der Intent-Router für jede Anfrage.

### 4. Orchestrator & MCP-Hub

In den jeweiligen Containern (Pfad frei wählbar, hier `/opt/` als Beispiel):

```bash
apt install -y python3 python3-pip python3-venv git

# Orchestrator
cp -r jarvis-orchestrator /opt/jarvis-orchestrator
cd /opt/jarvis-orchestrator
python3 -m venv venv
venv/bin/pip install -r requirements.txt

# MCP-Hub analog
cp -r jarvis-mcp-hub /opt/jarvis-mcp-hub
```

Konfiguration: Die Dienste laden beim Start automatisch eine `.env`-Datei
im Arbeitsverzeichnis. Alternativ können Variablen direkt im systemd-Service
via `Environment=` gesetzt werden (empfohlen für den Produktivbetrieb).

```bash
# .env anlegen
cp .env.example /opt/jarvis-orchestrator/.env
nano /opt/jarvis-orchestrator/.env   # OLLAMA_URL, MCP_HUB_URL, Modelle eintragen
```

Dienste starten:

```bash
/opt/jarvis-orchestrator/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
/opt/jarvis-mcp-hub/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8080
```

Für den Produktivbetrieb empfiehlt sich ein systemd-Service (Beispiel in [KONZEPT.md](KONZEPT.md)).

### 5. TriliumNext (Wiki-Backend)

TriliumNext läuft als Standalone-Binary ohne Docker. Im `jarvis-trilium`-Container:

```bash
apt update && apt install -y wget libatomic1

# Aktuelle Version von https://github.com/TriliumNext/Notes/releases laden
# Datei: TriliumNextNotes-*-linux-x64-server.tar.xz
cd /tmp
tar xf TriliumNextNotes-*-linux-x64-server.tar.xz
mv trilium-linux-x64-server /opt/trilium

useradd -r -s /bin/false trilium
mkdir -p /opt/trilium-data
chown trilium:trilium /opt/trilium-data /opt/trilium
```

systemd-Service (`/etc/systemd/system/trilium.service`):

```ini
[Unit]
Description=TriliumNext Notes
After=network.target

[Service]
Type=simple
User=trilium
Environment="TRILIUM_DATA_DIR=/opt/trilium-data"
ExecStart=/opt/trilium/trilium.sh
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
systemctl enable --now trilium
```

Nach dem ersten Start (`http://<trilium-ip>:8080`) Admin-Passwort setzen,
dann unter *Einstellungen → API* einen ETAPI-Token generieren.
Diesen Token als `TRILIUM_API_TOKEN` im systemd-Service des MCP-Hubs eintragen:

```ini
# /etc/systemd/system/jarvis-mcp-hub.service (Auszug)
[Service]
Environment="TRILIUM_URL=http://<trilium-ip>:8080/etapi"
Environment="TRILIUM_API_TOKEN=<etapi-token>"
Environment="OLLAMA_URL=http://<inference-node-ip>:11434"
ExecStart=/opt/jarvis-mcp-hub/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8080
WorkingDirectory=/opt/jarvis-mcp-hub
Restart=always
```

## Konfiguration

Alle Adressen und Modellnamen werden über Umgebungsvariablen gesetzt.
Vorlage: [`.env.example`](.env.example)

| Variable | Beschreibung | Beispiel |
|---|---|---|
| `OLLAMA_URL` | Ollama-Adresse auf dem Inference Node | `http://192.168.1.100:11434` |
| `MCP_HUB_URL` | Adresse des MCP-Hub-Containers | `http://192.168.1.102:8080` |
| `MAIN_MODEL` | Hauptmodell (Ollama-Name) | `gemma3:12b` |
| `CODE_MODEL` | Code-Modell (Ollama-Name) | `qwen2.5-coder:7b` |
| `ROUTING_MODEL` | Routing-Modell (Ollama-Name) | `qwen3:8b` |
| `ASSISTANT_NAME` | Name des Assistenten im System-Prompt | `Assistant` |
| `TRILIUM_URL` | TriliumNext ETAPI-Endpoint (optional) | `http://192.168.1.104:8080/etapi` |
| `TRILIUM_API_TOKEN` | TriliumNext ETAPI-Token (optional) | `<token>` |

## Intent-Routing

Der Orchestrator erkennt automatisch sechs Intents:

| Intent | Auslöser | Verarbeitung |
|---|---|---|
| `chat` | allgemeine Fragen, Erklärungen | Hauptmodell |
| `web_search` | aktuelle Informationen, Nachrichten, Wetter | Hauptmodell + Web-Kontext |
| `wiki_query` | "was hab ich notiert", "mein Wiki", persönliche Notizen | Hauptmodell + Wiki-Kontext |
| `wiki_ingest` | "merke dir: ...", "notiere das" | direkt ins Wiki (kein LLM) |
| `wiki_lint` | "baue deinen Index auf", "prüfe alle Links" | direktes Tool-Ergebnis |
| `code` | Code schreiben, debuggen, erklären | Code-Modell |

`wiki_ingest` und `wiki_lint` sind **DIRECT_INTENTS**: das Tool-Ergebnis des MCP-Hubs
wird direkt zurückgegeben, ohne einen weiteren LLM-Aufruf.

Zuerst greift ein Keyword-Filter (ohne Latenz), danach ein LLM-Klassifikator
als Fallback. Eigene Keywords können in `intent.py` ergänzt werden.

## LLM-Wiki (Karpathy-Muster)

Statt einer klassischen Vektordatenbank pflegt das LLM selbst eine Markdown-Wiki.

### Warum kein RAG?

| | Klassisches RAG | LLM-Wiki |
|---|---|---|
| Speicher | Vektordatenbank (Embeddings) | Lesbare Markdown-Dateien |
| Abfrage | Embedding-Suche → Chunk-Rückgabe | LLM wählt Seiten → synthetisiert |
| Wartung | Keine | LLM aktualisiert Seiten bei jedem Ingest |
| Konsistenz | Keine (Chunks isoliert) | Cross-References, Widerspruchs-Erkennung |
| Token-Effizienz | ~70× mehr Tokens pro Query | Kompakte Wiki-Seiten, wenige Tokens |

### Die Kontextfenster-Herausforderung

Auf Consumer-Hardware ohne Flash Attention liegt das effektive Kontextfenster bei
**≤ 8 192 Token** — das ist die zentrale Einschränkung, die die gesamte Architektur formt:

- **Atomare Seiten**: Eine Seite = ein Konzept (angestrebt ≤ 400 Tokens)
- **Index-first**: Das LLM liest zuerst einen kompakten Index (`path | title`, ~50 Zeichen/Eintrag),
  wählt dann gezielt 1–3 Seiten — statt alle Inhalte blind in den Kontext zu laden
- **Lint-Modus**: Analysiert maximal 20 Seiten gleichzeitig (durch das Kontextfenster begrenzt)
- **Ab ~300 Seiten**: Relevanz-basiertes Index-Filtering nötig

### Drei Wiki-Operationen

**wiki_ingest** — neues Wissen einpflegen (per Chat: "Merke dir: ...")

Das LLM liest den Index und verwandte Seiten, integriert das neue Wissen in atomare Notizen,
fügt [[wikilinks]] ein und aktualisiert den Index:

```bash
curl -X POST http://<mcp-hub-ip>:8080/tools/wiki_ingest \
  -H "Content-Type: application/json" \
  -d '{"title": "Meine Notiz", "content": "# ..."}'
```

**wiki_query** — eigenes Wissen abfragen (per Chat: "Was weiß ich über ...?")

Das Routing-Modell extrahiert Suchbegriffe, das LLM wählt relevante Seiten aus dem Index
und synthetisiert eine Antwort:

```bash
curl -X POST http://<mcp-hub-ip>:8080/tools/wiki_query \
  -H "Content-Type: application/json" \
  -d '{"query": "Proxmox"}'
```

**wiki_lint** — Wiki-Qualität prüfen (per Chat: "Baue deinen Index auf")

Drei Modi:
- `index` — baut den `schema/index` aus allen Seiten neu auf (kein LLM-Aufruf)
- `links` — prüft alle `[[wikilinks]]` auf Existenz (kein LLM-Aufruf)
- `full` — links + LLM-Konsistenzcheck (Widersprüche, fehlende Verlinkungen)

```bash
curl -X POST http://<mcp-hub-ip>:8080/tools/wiki_lint \
  -H "Content-Type: application/json" \
  -d '{"mode": "index"}'
```

### Wiki-Backend: TriliumNext

Als Wiki-Backend kommt [TriliumNext](https://github.com/TriliumNext/Notes) über die ETAPI zum Einsatz.
Jede Wiki-Notiz trägt das Label `#jarvisWiki` und ein `wikiPath`-Attribut für die Pfad-Adressierung.

Die Note Map in Trilium zeigt automatisch alle `[[wikilinks]]` als grafisches Beziehungsdiagramm —
ein nützlicher Nebeneffekt der atomaren Seitenstruktur.

## Projektstruktur

```
.
├── jarvis-orchestrator/       ← Intent-Router + LLM-Gateway
│   ├── main.py                ← FastAPI: /chat, /api/chat (Ollama-kompatibler Endpoint)
│   ├── graph.py               ← LangGraph StateGraph
│   ├── intent.py              ← Keyword-Filter + LLM-Klassifikator
│   ├── llm.py                 ← Ollama-Client (respond + stream_respond)
│   ├── tools.py               ← web_search, wiki_query, wiki_lint, wiki_ingest via MCP-Hub
│   └── requirements.txt
├── jarvis-mcp-hub/            ← Tool-Server
│   ├── main.py                ← FastAPI: /tools/web_search, /tools/wiki_*
│   ├── wiki.py                ← TriliumNext ETAPI-Client
│   └── requirements.txt
├── KONZEPT.md                 ← Vollständiges technisches Konzept
├── .env.example               ← Konfigurationsvorlage
└── .gitignore
```

## Phasenplan

Das Projekt ist in Phasen aufgebaut — jede Phase ist eigenständig nutzbar:

| Phase | Inhalt | Stand |
|---|---|---|
| 1 | Inference Node + Ollama + Modelle | ✅ Implementiert |
| 2 | Open WebUI über Proxmox-Container | ✅ Implementiert |
| 3 | Orchestrator + MCP-Hub + Web-Suche | ✅ Implementiert |
| 4 | LLM-Wiki (Karpathy-Muster, TriliumNext) | ✅ Implementiert |
| 4.5 | Multi-Agent Architektur (dynamisches Routing aus Wiki) | Geplant |
| 6 | Pi Zero 2W Voice Satellite (100% lokal) | Geplant |

Details und Setup-Anleitungen zu jeder Phase in [KONZEPT.md](KONZEPT.md).

## Hinweise

**Modellwahl:** Die in der Konfiguration angegebenen Modelle sind Empfehlungen.
Jedes Ollama-Modell kann eingesetzt werden — entscheidend ist der verfügbare VRAM.
Gemessene Token/s auf der eigenen Hardware sind aussagekräftiger als Benchmarks.

**TriliumNext ist optional:** Wenn kein TriliumNext vorhanden ist, funktioniert alles außer
`wiki_query`, `wiki_ingest` und `wiki_lint`. Der Intent-Router erkennt Wiki-Anfragen trotzdem —
sie fallen dann auf eine Fehlermeldung zurück, bis das Backend konfiguriert ist.

**Sprachsteuerung ohne Cloud:** Ein Alexa Custom Skill wäre technisch möglich,
aber Spracheingaben würden über Amazon-Server laufen — das widerspricht dem
Grundprinzip vollständiger Datenkontrolle. Deshalb setzt das Projekt auf die
Wyoming-Satellite-Pipeline (Phase 6): Wake Word, STT und TTS laufen vollständig
lokal auf einem Raspberry Pi Zero 2W (~48 €).

## Lizenz

MIT
