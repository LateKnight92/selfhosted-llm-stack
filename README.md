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
│    Routing-Modell (8B Q4_K_M)       │      │                                 │
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
- [Wiki.js](https://js.wiki) (optional, für LLM-Wiki)

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

Drei Container werden benötigt (Debian 12, unprivileged, je nach Bedarf):

| Hostname | RAM | Disk | Port |
|---|---|---|---|
| `jarvis-orchestrator` | 1.5 GB | 8 GB | 8000 |
| `jarvis-mcp-hub` | 1.0 GB | 8 GB | 8080 |
| `jarvis-webui` | 0.8 GB | 20 GB | 3000 |

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
| `WIKI_URL` | Wiki.js GraphQL-Endpoint (optional) | `http://192.168.1.105:3000/graphql` |
| `WIKI_API_TOKEN` | Wiki.js API-Token (optional) | `<token>` |

## Intent-Routing

Der Orchestrator erkennt automatisch vier Intents:

| Intent | Auslöser | Modell |
|---|---|---|
| `chat` | allgemeine Fragen, Erklärungen | Hauptmodell |
| `web_search` | aktuelle Informationen, Nachrichten, Wetter | Hauptmodell + Web-Kontext |
| `wiki_query` | "was hab ich notiert", "mein Wiki", persönliche Notizen | Hauptmodell + Wiki-Kontext |
| `code` | Code schreiben, debuggen, erklären | Code-Modell |

Zuerst greift ein Keyword-Filter (ohne Latenz), danach ein LLM-Klassifikator
als Fallback. Eigene Keywords können in `intent.py` ergänzt werden.

## LLM-Wiki (Karpathy-Muster)

Statt einer klassischen Vektordatenbank pflegt das LLM selbst eine
Markdown-Wiki. Vorteile gegenüber RAG:

- Seiten bleiben menschenlesbar (im Browser über Wiki.js)
- LLM aktualisiert und verknüpft Seiten bei jedem Ingest
- Keine Embedding-Datenbank nötig
- Widersprüche werden erkannt und aufgelöst

Das Wiki-Backend ist Wiki.js über die GraphQL-API. Als Alternative
kann das `wiki/`-Verzeichnis direkt auf dem Dateisystem gepflegt werden
(dann Filesystem-MCP statt GraphQL).

Wiki-Seite anlegen:

```bash
curl -X POST http://<mcp-hub-ip>:8080/tools/wiki_ingest \
  -H "Content-Type: application/json" \
  -d '{"title": "Meine Notiz", "path": "notizen/meine-notiz", "content": "# ...", "locale": "en"}'
```

## Projektstruktur

```
.
├── jarvis-orchestrator/       ← Intent-Router + LLM-Gateway
│   ├── main.py                ← FastAPI: /chat, /api/chat (Ollama-kompatibler Endpoint)
│   ├── graph.py               ← LangGraph StateGraph
│   ├── intent.py              ← Keyword-Filter + LLM-Klassifikator
│   ├── llm.py                 ← Ollama-Client (respond + stream_respond)
│   ├── tools.py               ← web_search + wiki_query via MCP-Hub
│   └── requirements.txt
├── jarvis-mcp-hub/            ← Tool-Server
│   ├── main.py                ← FastAPI: /tools/web_search, /tools/wiki_query, /tools/wiki_ingest
│   ├── wiki.py                ← Wiki.js GraphQL-Client
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
| 4 | LLM-Wiki (Karpathy-Muster) | ✅ Implementiert |
| 5 | Pi Zero 2W Voice Satellite (100% lokal) | Geplant |

Details und Setup-Anleitungen zu jeder Phase in [KONZEPT.md](KONZEPT.md).

## Hinweise

**Modellwahl:** Die in der Konfiguration angegebenen Modelle sind Empfehlungen.
Jedes Ollama-Modell kann eingesetzt werden — entscheidend ist der verfügbare VRAM.
Gemessene Token/s auf der eigenen Hardware sind aussagekräftiger als Benchmarks.

**Wiki.js ist optional:** Wenn kein Wiki.js vorhanden ist, funktioniert alles außer
`wiki_query` und `wiki_ingest`. Der Intent-Router erkennt Wiki-Anfragen trotzdem —
sie fallen dann auf eine Fehlermeldung zurück, bis das Backend konfiguriert ist.

**Sprachsteuerung ohne Cloud:** Ein Alexa Custom Skill wäre technisch möglich,
aber Spracheingaben würden über Amazon-Server laufen — das widerspricht dem
Grundprinzip vollständiger Datenkontrolle. Deshalb setzt das Projekt auf die
Wyoming-Satellite-Pipeline (Phase 5): Wake Word, STT und TTS laufen vollständig
lokal auf einem Raspberry Pi Zero 2W (~48 €).

## Lizenz

MIT
