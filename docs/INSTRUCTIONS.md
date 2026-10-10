# Projektspezifikation: `stop` (Smart-OP System-One Controller)

**System-Architektur und Implementierungsrichtlinie für die Agenten-Ausführung**

---

## 1. Übersicht & Zielsetzung

`stop` ist eine fehlertolerante, latenzkritische Steuerungs- und Simulationsplattform für vernetzte Medizingeräte im OP-Saal (Laparoskopie, Kaltlicht, Insufflatoren, OP-Tisch). Das System nutzt das **JevK5-2B Decision Model** (bzw. die kompatible System-One Typed Decision HTTP API, da Modell wird auf privater Workstation laufen) für paralleles, slot-basiertes Multi-Pass Decision Making.

Die Projektsprache ist Englisch. Vergiss beim Designen und Planen der Spezifikationen weiter unten nicht darauf, dass die System-One Modells trotzdem große Latenz haben und daher die Logik non-blocking funktionieren muss.

### Kernprinzip des Multi-Pass Decision Loops

Ein einzelner chirurgischer Sprachbefehl kann mehrere simultane Aktionen implizieren (z. B. *„Licht auf 40 % dimmen und mit der Optik zwei Stufen heranfahren“*). Anstatt unstrukturiertes JSON zeichenweise per LLM zu generieren:

1. `stop` serialisiert den aktuellen OP-Raumzustand (`RoomState`) und den Benutzer-Prompt in das Eingabeformat.
2. JevK5-2B evaluiert alle Decision Slots in einem **einzigen Vorwärtsdurchlauf** parallel.
3. Die Engine wendet die primäre Aktion auf den Raumzustand an.
4. Die Engine wendet die Objekt-Entscheidungen des **einzigen Vorwärtsdurchlaufs** an: pro Raumobjekt eine Entscheidung mit `null`-Option (keine Änderung) plus absolute Zielwerte. Mehrere simultane Aktionen lösen so in einem Pass auf — kein Re-Evaluation-Loop.

```
                   +---------------------------------------------+
                   |           User Input (CLI / STT)            |
                   +---------------------------------------------+
                                          |
                                          v
+------------------+     Prompt +     +-------------------------+
|                  |   RoomState[n]   |                         |
|   stop-gui       |<-----------------|    stop-core Loop       |
|  (Vello/Winit)   |                  |                         |
|                  |----------------->|  - Build Context State  |
+------------------+  Visual Feedback |  - Call Jev Provider    |
        ^                             |  - Apply Delta to State |
        | State Delta Event           |  - Single pass only     |
        +-----------------------------+-------------------------+
                                          |
                                          v
                              +-------------------------+
                              | JevK5-2B / System-One   |
                              |   (Typed Parallel Head) |
                              +-------------------------+

```

---

## 2. Tooling & Workspace-Setup

Das Projekt wird vollständig über `mise` verwaltet und als modularer Rust-Cargo-Workspace strukturiert. Sollte aus bestehenden Projekten generiert werden. Verweis erfolgt im Agent-Chat.

### 2.1 `mise.toml`

Die mise-Verwaltung folgt dem Drei-Dateien-Modell (Details: `docs/MISE.md`):

```toml
# mise.toml (global, versioniert)
[settings]
experimental = true

[vars]
REPO_ROOT = "{{config_root}}"

[tools]
rust = "nightly"
"cargo:cargo-nextest" = "latest"

[task_config]
includes = [
    "tasks/format.toml", "tasks/linting.toml", "tasks/check.toml",
    "tasks/fixes.toml", "tasks/tests.toml", "tasks/dev.toml",
    "tasks/misc.toml", "tasks/build.toml"
]
```

```toml
# mise.dev.toml (Dev-Umgebung, versioniert)
[env]
RUST_LOG = "info,stop_core=debug,stop_gui=debug"
JEV_API_BASE_URL = "http://localhost:8080" # Lokale JevK5 GPU Instanz
OPENROUTER_API_KEY = ""                     # Für stop-dataset Generator
```

`mise.local.toml` (gitignored) hält maschinen-spezifische Overrides (`CARGO_BUILD_JOBS` etc.).

Einstiegspunkte: `mise run fix` (Lint-Fixes + Format), `mise run check` (Format-/Lint-Gates),
`mise run test` (nextest), `mise run build` (Release), `mise run dev:gui`.

### 2.2 Cargo Workspace Layout

```
stop/
├── mise.toml                 # Tools, Vars, Task-Includes
├── mise.dev.toml             # Dev-Env-Variablen (RUST_LOG, JEV_API_BASE_URL, ...)
├── mise.local.toml           # lokale Overrides (gitignored)
├── tasks/                    # Task-Definitionen: fmt, lint, check, fix, test, build, dev, misc
├── Cargo.toml
├── docs/
│   ├── INSTRUCTIONS.md       # diese Spezifikation
│   └── MISE.md               # mise-Tooling-Doku (Struktur, Tasks, Env-Runnables)
├── crates/
│   ├── stop-core/            # Domain-Modell, Jev API-Client, Multi-Pass Engine, Events
│   ├── stop-dataset/         # OpenRouter API-Client, Generator-Bin für Ground-Truth-Daten
│   ├── stop-benchmark/       # Test-Runner, Genauigkeits- & ROC/AUC-Analyse, Calibration
│   └── stop-gui/             # Vello/Winit Rendering, CLI-Input-Thread, Event-Loop
├── assets/
│   └── icons/                # SVG-Shapes für OP-Geräte (Endoskop, Lampe, Tisch)
└── data/
    └── test_suite.jsonl      # Generiertes Test-Dataset mit Ground Truth

```

### 2.3 Wurzel `Cargo.toml`

```toml
[workspace]
resolver = "2"
members = [
    "crates/stop-core",
    "crates/stop-dataset",
    "crates/stop-benchmark",
    "crates/stop-gui",
]

[workspace.package]
version = "0.1.0"
edition = "2024"
authors = ["Tim Peko <timerertim@gmail.com>"]
license-file = "LICENSE"

[workspace.dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tokio = { version = "1.40", features = ["full"] }
thiserror = "1.0"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
reqwest = { version = "0.12", features = ["json"] }

```

---

## 3. `stop-core`: Domänenmodell & Typed Decision Engine

`stop-core` enthält keine GUI- und keine Benchmarking-Logik. Es definiert den Zustand, die Aktionen, den Engine-Loop und das Typed API-Interface.

### 3.1 OP-Raumzustand (`RoomState`)

Der State muss kompakt als formatiertes JSON serialisierbar sein (für den Jev 16k-Context).

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RoomState {
    pub lighting: LightingState,
    pub endoscope: EndoscopeState,
    pub insufflator: InsufflatorState,
    pub table: TableState,
    pub safety_interlock_active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LightingState {
    pub primary_intensity_pct: u8, // 0 - 100
    pub field_mode: LightMode,     // Normal, CavityFocus, AmbientRed
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LightMode {
    Normal,
    CavityFocus,
    AmbientRed,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EndoscopeState {
    pub zoom_level: i8,            // 1 bis 5
    pub white_balance_locked: bool,
    pub irrigation_active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InsufflatorState {
    pub target_pressure_mmhg: u8,  // typisch 12 - 15 mmHg
    pub gas_flow_l_min: u8,
    pub is_active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TableState {
    pub tilt_degrees: i8,          // -15 (Trendelenburg) bis +15 (Anti-Trendelenburg)
    pub height_cm: u8,
}

```

### 3.2 Decision Slots & JevK5 Schema

Die JevK5-Modell-Köpfe geben parallele Klassifikationen aus. Ein einziger
Vorwärtsdurchlauf liefert ALLE Entscheidungen: pro Raumobjekt eine
`DeviceDecision` mit `null`-Option (keine Änderung) plus absolute Zielwerte
für wertsetzende Aktionen. Typnamen sind provider-agnostisch (`ActionDecision`,
kein `Jev`-Prefix — der Jev-/System-One-Client ist austauschbar):

```rust
/// Objekt-Entscheidung eines einzelnen Vorwärtsdurchlaufs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DeviceDecision {
    pub target_device: TargetDevice,
    /// `None`, wenn das Modell `null` gewählt hat: keine Änderung an diesem Objekt.
    pub action: Option<ActionKind>,
    /// Absoluter Zielwert (Helligkeit %, Zoom-Stufe, mmHg, Grad, cm, Modus-Code).
    pub absolute: Option<i16>,
    pub confidence: f32,
    pub absolute_confidence: f32,
}

/// Vollständige Entscheidungsmenge eines Vorwärtsdurchlaufs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UtteranceDecision {
    /// Abgeschlossene Objektmenge: Licht, Kamera, Insufflator, Tisch.
    pub devices: Vec<DeviceDecision>,
    pub emergency_stop: bool,
    pub requires_sterile_confirm: bool,
}

/// Deterministische State-Delta-Einheit (`apply_action_to_state`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionDecision {
    /// Zielgerät der Aktion
    pub target_device: TargetDevice,

    /// Konkrete Operation auf dem Zielgerät
    pub action_kind: ActionKind,

    /// Relativer Schrittwert / Diskrete Anpassung
    pub step_value: StepValue,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum TargetDevice {
    None,
    SurgicalLight,
    EndoscopeCamera,
    Insufflator,
    OperatingTable,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ActionKind {
    Idle,
    IncreaseBrightness,
    DecreaseBrightness,
    SetLightMode,
    ZoomIn,
    ZoomOut,
    ToggleIrrigation,
    ToggleWhiteBalanceLock,
    AdjustPressure,
    AdjustGasFlow,
    ToggleInsufflation,
    TiltTable,
    SetTableHeight,
    EngageSafetyInterlock,
    EmergencyStop,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum StepValue {
    Zero,
    PlusOne,
    PlusTwo,
    MinusOne,
    MinusTwo,
    AbsoluteValue(i16),
}

```

### 3.3 Der Single-Pass Orchestrator

Die Engine führt **genau einen** Inferenz-Call pro Utterance aus — der
Antwort-Decode liefert bereits alle Objekt-Entscheidungen, daher existiert
kein Re-Evaluation-Loop und kein Safety-Guard mehr:

```rust
pub struct SinglePassExecutor<I> {
    inference: I,
}

impl<I: InferencePort> SinglePassExecutor<I> {
    /// Ein Utterance -> exakt ein Inferenz-Call.
    pub async fn process_utterance(
        &self,
        current_state: &RoomState,
        utterance: &str,
    ) -> Result<ExecutionResult, ExecutionError> {

        // 1. Genau ein Inferenz-Payload: RoomState (JSON) + Utterance,
        //    ohne Pass-History (Token-Minimierung).
        let outcome = self.inference.single_pass(&InferenceInput {
            room_state: current_state,
            utterance,
        })
        .await?;

        // 2. EmergencyStop zuerst (Interlock, Insufflation/Irrigation aus),
        //    dann jede Objekt-Entscheidung einzeln: `null` oder Konfidenz
        //    < 0.5 bedeutet keine Änderung; wertsetzende Aktionen ohne
        //    confidenten Zielwert überspringen.

        // 3. State-Deltas deterministisch anwenden
        //    (Safety-Caps in den State-Structs: siehe Abschnitt 3.1).

        Ok(ExecutionResult {
            new_room,
            report: UtteranceReport { /* ... */ },
        })
    }
}

```

Jede Einzelaktion wird über die Setter der jeweiligen State-Struct erzwungen
(`set_intensity_pct`, `set_zoom_level`, `set_target_pressure_mmhg`,
`set_tilt_degrees`, `set_height_cm` — alle clamped intern): Helligkeit
0-100 %, Zoom 1-5, Druck hart gedeckelt bei `MAX_PRESSURE_MMHG = 25` mmHg,
Tischneigung -15..+15 Grad, Tischhöhe 70-130 cm. `EmergencyStop` aktiviert
den Safety-Interlock und schaltet Insufflation sowie Irrigation ab. Die
Konstanten liegen in `state.rs`. Die Objektmenge ist abgeschlossen (Licht,
Kamera, Insufflator, Tisch): mehrere simultane Aktionen eines Befehls lösen
im selben Pass auf, ohne gegeneinander zu konkurrieren.

---

## 4. `stop-dataset`: Synthetische Datenbeschaffung via OpenRouter

Um eine Ground Truth ohne manuelle Annotation zu erhalten, generiert eine eigenständige Binary (`generate-dataset`) strukturierte chirurgische Szenarien über OpenRouter (z. B. via `anthropic/claude-3.5-sonnet` oder `openai/gpt-4o`).

Die Generierung läuft über einen Zwischenschritt (Abschnitt 4.3): zuerst wird pro Szenario ein STT-nahes Transkript inklusive Rausch-Sprechakten erzeugt, daraus werden die eigentlichen Gerätebefehle extrahiert und schrittweise gegen `apply_action_to_state` ausgeführt — die Ground Truth entsteht also aus demselben Codepfad wie die Runtime.

### 4.1 CLI-Aufruf

```bash
cargo run -p stop-dataset --bin generate-data -- \
  --count 250 \
  --output data/test_suite.jsonl \
  --scenarios "laparoscopic_cholecystectomy,laparoscopic_hernia_repair,laparoscopic_appendectomy,laparoscopic_sleeve_gastrectomy,laparoscopic_fundoplication" \
  --include-noise

```

### 4.2 Datenformat (`data/test_suite.jsonl`)

Jede Zeile beschreibt ein komplettes Operations-Szenario und enthält:
- den initialen Zustand des Raumes (`initial_state`),
- eine Serie von realistischen, englischsprachigen STT-Eingaben (`raw_utterance`), wie sie von Chirurg:innen, OP-Personal oder Assistenz in Alltagssprache mit typischen Unterbrechungen und Füllwörtern gesprochen werden,
- sowie für jede Eingabe den erwarteten Zielzustand nach Anwendung aller inferierten Aktionen (`expected_output_state`).

Enthält der Eintrag Rauschen (Kapitel 4.3), ist das Feld `kind` gesetzt:
`"kind": "noise"` markiert handlungsneutrale Zwischen-Sprechakte, deren
`expected_output_state` identisch zum vorherigen Zustand ist (kein
State-Delta). Feld fehlt oder `"command"`: handlungsrelevant, Delta wird
angewendet. Marker `"self_correction"` kennzeichnet Sprechakte, die einen
vorherigen Befehl zurücknehmen; das korrigierte Delta wird bei der Auswertung
nicht mitgezählt — der Endzustand bleibt die Referenz.

So lassen sich komplexe, mehrstufige Operationsverläufe und authentische Dialog-Sequenzen abbilden. Damit prüfen wir nicht nur Einzelaktionen, sondern ganze Abfolgen, z. B. mehrere Kommandos und Gerätezustandsänderungen pro Fall.

```json
{
  "id": "case_chole_042",
  "procedure": "cholecystectomy",
  "initial_state": {
    "lighting": { "primary_intensity_pct": 80, "field_mode": "Normal" },
    "endoscope": { "zoom_level": 2, "white_balance_locked": true, "irrigation_active": false },
    "insufflator": { "target_pressure_mmhg": 12, "gas_flow_l_min": 10, "is_active": true },
    "table": { "tilt_degrees": 0, "height_cm": 100 },
    "safety_interlock_active": false
  },
  "history": [
    {
      "raw_utterance": "uhm nurse uh could you uh dim the overhead OR lights by like two steps and uh wait no uh also can you bring the endoscope a little closer? yeah, thanks",
      "expected_output_state": {
        "lighting": { "primary_intensity_pct": 60, "field_mode": "Normal" },
        "endoscope": { "zoom_level": 3, "white_balance_locked": true, "irrigation_active": false },
        "insufflator": { "target_pressure_mmhg": 12, "gas_flow_l_min": 10, "is_active": true },
        "table": { "tilt_degrees": 0, "height_cm": 100 },
        "safety_interlock_active": false
      }
    },
    {
      "kind": "noise",
      "raw_utterance": "yeah can someone check the CO2 canister after this, ah and uh what time is it even... anyway",
      "expected_output_state": {
        "lighting": { "primary_intensity_pct": 60, "field_mode": "Normal" },
        "endoscope": { "zoom_level": 3, "white_balance_locked": true, "irrigation_active": false },
        "insufflator": { "target_pressure_mmhg": 12, "gas_flow_l_min": 10, "is_active": true },
        "table": { "tilt_degrees": 0, "height_cm": 100 },
        "safety_interlock_active": false
      }
    },
    {
      ...
    }
  ]
}

```

### 4.3 Rauschtranskripte als Zwischenschritt

Rein befehlshaltige Sequenzen sind unrealistisch: echte STT-Transkripte enthalten
vor, zwischen und nach den Befehlen handlungsneutrale Sprache. Der Generator
erzeugt deshalb pro Szenario in einem ersten OpenRouter-Zwischenschritt ein
Transkript-Mikrosegment mit:

* **Smalltalk / Kommentare** ohne Gerätebezug („did you sleep okay?", Uhrzeit,
  Abläufe), Team-Kommentare, nachträgliche Fragen der Pflege;
* **Füllwörter, Selbstkorrekturen, Verwerfungen** („wait no — actually dim it
  less"), Abbrüche mitten im Satz;
* **bereits ausgeführte Befehle**, die nur bestätigt werden („yep, that's at
  40 now"), und
* **Antworten auf Zwischenfragen**, die nicht an die OP-Geräte gerichtet sind.

Aus diesem Transkript extrahiert ein zweiter Schritt die eigentlichen
Gerätebefehle (`kind: "command"`), wirft die restlichen Sprechakte als
`kind: "noise"` ein und erzeugt pro Schritt das erwartete Ergebnis, indem
`apply_action_to_state` schrittweise über den Raumzustand läuft. Rausch-
Einträge behalten den vorherigen Zustand unverändert; Selbstkorrekturen
(`self_correction`) neutralisieren sich über den Sequenzverlauf.

Eval-Metrik: `eval-accuracy` wertet Noise-Einträge als Null-Erwartung —
**jede** gegen einen Noise-Eintrag vorhergesagte State-Änderung zählt als
Falsch-Positiv (alle Objekt-Entscheidungen sollen hier `null` sein).

---

## 5. `stop-benchmark`: Evaluierung & Statistische Analyse

Um wiederholte und unnötige API-Latenz zu vermeiden, trennt die Benchmark-Crate strikt zwischen Datengenerierung (SystemOne Requests) und Auswertung:
- **Ein zentrales Main-Binary** (`run-benchmark`) führt einmalig alle Fälle aus dem Dataset vollständig gegen die lokale JevK5-Instanz aus und speichert die Roh-Ausgaben (inkl. pro Pass/Slot Rückgaben, Latenzen, Konfidenzen) in einer Output-Datei (z.B. `benchmark_results.jsonl`).
- **Alle spezialisierten Analyse-Binaries** (`eval-accuracy`, `eval-roc`, `eval-latency` etc.) operieren rein auf diesen gespeicherten Rohdaten und NICHT auf dem Dataset direkt. Dadurch kann beliebig viele Auswertungen fahren, ohne das Model unnötig mehrfach zu befragen.

### 5.1 Ablauf

1. **Durchlauf und Speicherung**
    - `cargo run -p stop-benchmark --bin run-benchmark -- --input data/test_suite.jsonl --output data/benchmark_results.jsonl`
    - Führt alle Fälle mit SystemOne aus, persistiert sämtliche Model-Raw-Outputs, Latenzen und Metadaten.
    - Pro Utterance werden **zwei Vorhersage-Varianten** aufgenommen:
        - `fresh_prediction`: Input ist der vorherige **erwartete** Zustand
          (Ground-Truth-Verkettung; isoliert Einzelfehler von Rollout-Drift).
        - `rolling_prediction`: Input ist der vorherige **vorhergesagte** Zustand
          (selbstverkettender Rollout ab `initial_state`; misst Fehlerakkumulation).
          Eine fehlgeschlagene Rolling-Vorhersage lässt alle folgenden des Falls
          ebenfalls fehlschlagen (`Err` ohne weitere Inferenz-Calls).
    - Beide Varianten tragen je `state`, `inference_passes` und `wall_latency_ms`.
    - `expected_output_state` heißt jetzt `expected_state`.

2. **Analyse-Binaries:**
    - `cargo run -p stop-benchmark --bin eval-accuracy -- --input data/benchmark_results.jsonl`
        - Slot-spezifische Genauigkeit, **Sequence Exact Match** (Durchgänge vollständig korrekt). Noise-Einträge (`kind: "noise"`) zählen als Null-Erwartung: jede vorhergesagte State-Änderung dagegen ist Falsch-Positiv.
        - Splits: Overall, **pro State-Feld** (`light_brightness`, `light_mode`, `zoom_level`, ...), **pro Szenario**, **pro Modell**, plus Per-Case-Detail (`failed_entry_indices`) — jeweils als **Fresh-vs-Rolling-Gruppen** (zwei Tabellenzeilen pro Schlüssel).
        - `--out <pfad>` schreibt den kompletten Breakdown (`AccuracyBreakdown`) als Pretty-JSON-Datei.
    - `cargo run -p stop-benchmark --bin eval-correlation -- --input data/benchmark_results.jsonl`
        - Fehlerrate nach `entry_index` (Position in der Fall-History) und nach Fall-Länge (Anzahl Einträge), plus Pearson-Korrelationen (`null` in JSON bei fehlender Varianz) — jeweils pro Variante (fresh, rolling).
        - `--out <pfad>` schreibt den kompletten Breakdown (`CorrelationBreakdown`) als Pretty-JSON-Datei.
    - `cargo run -p stop-benchmark --bin eval-latency -- --input data/benchmark_results.jsonl`
        - Latenz-Verteilung (P50, P95, P99) auf Basis der zuvor gemessenen Pass-Latenzen, **pro Modell** (Tabelle je `model_name`) plus Overall — Pass- und Utterance-Latenz getrennt für fresh und rolling.
        - `--out <pfad>` schreibt den kompletten Breakdown (`LatencyBreakdown`) als Pretty-JSON-Datei.

### 5.2 Konsolen-Reporting

Die Auswertung erzeugt für jede Analyse gut lesbare Tabellen im Terminal, je Split
eine eigene überschriebene Sektion; mit `--out` wird zusätzlich der komplette
Breakdown als JSON-Datei geschrieben:

```
Overall
+-----------------------+----------+---------+-------+
| Metric / Slot         | Accuracy | Matched | Total |
+-----------------------+----------+---------+-------+
| Overall accuracy      | 0.982    | 1964    | 2000  |
| Action-entry accuracy | 0.941    | 941     | 1000  |
| No-change accuracy    | 0.980    | 1023    | 1044  |
+-----------------------+----------+---------+-------+
Sequence Exact Match (SEM): 88.4%
No-change false positives (predicted state change on unchanged expected state): 12

Per field
+-------------------------+----------+---------+-------+
| Field                   | Accuracy | Matched | Total |
+-------------------------+----------+---------+-------+
| light_brightness        | 0.982    | 1964    | 2000  |
| light_mode              | 0.991    | 1982    | 2000  |
| ...                     | ...      | ...     | ...   |
+-------------------------+----------+---------+-------+

Per scenario
+-----------------+----------+---------+-------+------+
| Scenario        | Accuracy | Matched | Total | SEM  |
+-----------------+----------+---------+-------+------+
| cholecystectomy | 0.990    | 495     | 500   | 96%  |
| ...             | ...      | ...     | ...   | ...  |
+-----------------+----------+---------+-------+------+

Mean Latency per Pass: 21.4 ms (GPU local)
```

---

## 6. `stop-gui`: Der visuelle Driving-Adapter

Die GUI dient als interaktives Schaufenster für die Präsentation. Sie rendert einen schematischen 2D-Operationssaal mit `vello` (GPU-beschleunigtes 2D-Rendering via WGPU) oder vektorisiertem `tiny-skia`/`egui`.

### 6.1 Threading & Event-Architektur

Die Anwendung trennt strikt zwischen CLI-Input, async Core-Execution und Rendering-Loop:

```
[CLI Stdin Reader Thread]
           |
           | String (Prompt)
           v
[Tokio Runtime: stop-core Executor]
           |
           |-- (1) Event: PromptReceived(text) ----> [Unbounded Channel]
           |-- (2) Loop Pass 1 Executed ---------->         |
           |-- (3) StateDeltaApplied(delta) ------>         v
           +-- (4) ExecutionFinished -------------> [Winit Event Loop / GUI]
                                                            |
                                                            v
                                                   Redraw Frame (Vello)

```

### 6.2 Visuelle Elemente des OP-Saals

1. **Der OP-Tisch (Mitte):** Schematische Liege, die sich bei `TiltTable` visuell in Grad-Schritten neigt.
2. **Die Decken-OP-Leuchte (Oben):** Ein Lichtkegel auf den Tisch. Helligkeit und Transparenz ändern sich reaktiv mit `primary_intensity_pct`. Bei Modus `AmbientRed` schaltet der Raum auf rote Laparoskopie-Hintergrundbeleuchtung um.
3. **Endoskop-Monitor (Rechts oben):** Zeigt ein simuliertes laparoskopisches Ziel (Kreise/Gewebe-Vektorform). Bei `ZoomIn`/`ZoomOut` skaliert der Bildausschnitt. Bei `irrigation_active` werden Wassertropfen animiert.
4. **Insufflator-Druckanzeige (Links):** Digitaler Bar-Graph für $CO_2$-Druck mit visuellem Warnbereich ab > 15 mmHg.
5. **HUD & Telemetrie-Leiste (Unten):**
* **Prompt-Display:** Zeigt den eingetippten Befehl sofort an.
* **Inferenz-HUD:** Zeigt den Multi-Pass-Verlauf mit Latenz an:
`Pass 1: Light -> Dim (-2) [19ms] | Pass 2: Camera -> Zoom (+1) [21ms] | Done.`



---

## 7. Erweiterungsschnittstelle für Audio & STT (Future Seam)

Damit das System später ohne Code-Umbau um Mikrofon-Input und Whisper / Speech-to-Text erweitert werden kann, definiert `stop-core` ein klares Eingabe-Trait:

```rust
use async_trait::async_trait;

#[async_trait]
pub trait CommandStreamSource: Send + Sync {
    async fn next_command(&mut self) -> Result<Option<String>, SourceError>;
}

/// Aktuelle Standard-Implementierung
pub struct StdinCliSource;

#[async_trait]
impl CommandStreamSource for StdinCliSource {
    async fn next_command(&mut self) -> Result<Option<String>, SourceError> {
        let mut buffer = String::new();
        std::io::stdin().read_line(&mut buffer)?;
        let trimmed = buffer.trim().to_string();
        if trimmed.is_empty() { Ok(None) } else { Ok(Some(trimmed)) }
    }
}

/// Spätere Audio-STT Implementierung (Platzhalter im Crate-Design)
pub struct MicrophoneWhisperSource {
    // Audio device handle (cpal / rodio)
    // Whisper-rs inference context
}

```

---

## 8. Agenten-Implementierungs-Roadmap

Der ausführende Agent soll die Implementierung in 5 Phasen abarbeiten:

### Phase 1: Workspace & Core-Fundament

* `mise.toml` anlegen und Workspace-`Cargo.toml` aufsetzen.
* `stop-core`: `RoomState`, `ActionDecision`, `TargetDevice`, `ActionKind` und Serialisierungs-Tests implementieren.
* State-Delta-Logik unit-testen (z. B. `apply_action_to_state` verhindert Drücke > 25 mmHg); Safety-Caps als Setter in `state.rs` verankern.

### Phase 2: Mock-Engine & Single-Pass Loop

* Trait `DecisionEngineProvider` definieren.
* `MockDecisionEngine` schreiben, die konfigurierbare Sequenzen zurückgibt.
* `SinglePassExecutor` implementieren und testen: exakt ein Inferenz-Call pro Utterance, alle Objekt-Entscheidungen in diesem einen Pass.
* HTTP-Client für JevK5 / System-One API implementieren.

### Phase 3: Datensatz-Generator (`stop-dataset`)

* OpenRouter API-Client mit `reqwest` bauen.
* Structured Prompting für synthetische OP-Dialoge aufsetzen.
* Rauschtranskripte als Zwischenschritt generieren (`--include-noise`): handlungsneutrale Zwischen-dialoge, Smalltalk, Füllwörter, Selbstkorrekturen, Team-Kommentare ohne Gerätebezug, die vor dem eigentlichen Befehl im Transkript landen (siehe Abschnitt 4.3).
* Binary `generate-data` erstellen und ersten 100-Zeilen Test-Datensatz in `data/test_suite.jsonl` erzeugen.

### Phase 4: Benchmarking Crate (`stop-benchmark`)

* Parser für `data/test_suite.jsonl` implementieren (inkl. `kind: "noise"`-Einträge als Null-Erwartung, siehe Abschnitt 4.3).
* Metrik-Berechnung für Accuracy, Sequence Exact Match und Latenz schreiben.
* CLI-Reporting mit formatierter Ausgabe fertigstellen.

### Phase 5: GUI & Interaktive Demo (`stop-gui`)

* Winit-Fenster und Vello/2D-Rendering-Pipeline einrichten.
* Raumobjekte zeichnen: OP-Tisch, Lichtkegel, Monitor, Insufflator-Bar.
* Inter-Thread-Channels verbinden: CLI-Eingabe $\to$ Core-Multi-Pass-Engine $\to$ State-Update $\to$ GUI-Repaint.

Nach jeder Phase soll ein Testabschnitt, der /diffx-start-review skill ausgeführt und bei Freigabe der JJ-Commit erstellt werden.

---
