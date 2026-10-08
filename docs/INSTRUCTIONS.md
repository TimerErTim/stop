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
4. Falls der Slot `further_action_needed == true` signalisiert, re-evaluiert die Engine den **aktualisierten Raumzustand** mit demselben Prompt (Multi-Pass Loop), bis `further_action_needed == false` oder ein Sicherheitslimit erreicht ist.

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
        | State Delta Event           |  - Loop if needed       |
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

```toml
[tools]
rust = "nightly"

[env]
RUST_LOG = "info,stop_core=debug,stop_gui=debug"
JEV_API_BASE_URL = "http://localhost:8080" # Lokale JevK5 GPU Instanz
OPENROUTER_API_KEY = ""                     # Für stop-dataset Generator

```

### 2.2 Cargo Workspace Layout

```
stop/
├── mise.toml
├── Cargo.toml
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

Die JevK5-Modell-Köpfe geben parallele Klassifikationen aus:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JevDecisionSlots {
    /// Signalisiert dem Orchestrator, ob ein weiterer Durchlauf nötig ist
    pub further_action_needed: bool,

    /// Zielgerät der aktuellen Teilaktion
    pub target_device: TargetDevice,

    /// Konkrete Operation auf dem Zielgerät
    pub action_kind: ActionKind,

    /// Relativer Schrittwert / Diskrete Anpassung
    pub step_value: StepValue,

    /// Sicherheitsrelevante Verifikation nötig (z. B. Überdruck oder OP-Tischneigung)
    pub requires_sterile_confirm: bool,
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
    AdjustPressure,
    ToggleInsufflation,
    TiltTable,
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

### 3.3 Der Multi-Pass Orchestrator

Die Engine führt den zyklischen Inferenz-Loop aus:

```rust
pub struct MultiPassExecutor<E: DecisionEngineProvider> {
    engine: E,
    max_passes: usize, // Standard: 4 (Safety Guard gegen Endlosschleifen)
}

impl<E: DecisionEngineProvider> MultiPassExecutor<E> {
    pub async fn process_utterance(
        &self,
        current_state: &mut RoomState,
        utterance: &str,
    ) -> Result<Vec<AppliedActionReport>, ExecutionError> {
        let mut executed_actions = Vec::new();
        let mut pass_count = 0;

        loop {
            pass_count += 1;
            if pass_count > self.max_passes {
                tracing::warn!("Max passes reached, breaking cycle");
                break;
            }

            // 1. Inferenz-Payload bauen (State + Prompt + History)
            let prediction = self.engine.infer_slots(current_state, utterance).await?;

            // 2. State-Delta deterministisch anwenden
            if prediction.target_device != TargetDevice::None {
                let delta = apply_action_to_state(current_state, &prediction)?;
                executed_actions.push(delta);
            }

            // 3. Abbruchbedingung prüfen
            if !prediction.further_action_needed || prediction.target_device == TargetDevice::None {
                break;
            }
        }

        Ok(executed_actions)
    }
}

```

---

## 4. `stop-dataset`: Synthetische Datenbeschaffung via OpenRouter

Um eine Ground Truth ohne manuelle Annotation zu erhalten, generiert eine eigenständige Binary (`generate-dataset`) strukturierte chirurgische Szenarien über OpenRouter (z. B. via `anthropic/claude-3.5-sonnet` oder `openai/gpt-4o`).

### 4.1 CLI-Aufruf

```bash
cargo run -p stop-dataset --bin generate-data -- \
  --count 250 \
  --output data/test_suite.jsonl \
  --scenarios "cholecystectomy,hernia_repair,appendectomy" \
  --include-noise

```

### 4.2 Datenformat (`data/test_suite.jsonl`)

Jede Zeile beschreibt ein komplettes Operations-Szenario und enthält:
- den initialen Zustand des Raumes (`initial_state`),
- eine Serie von realistischen, englischsprachigen STT-Eingaben (`raw_utterance`), wie sie von Chirurg:innen, OP-Personal oder Assistenz in Alltagssprache mit typischen Unterbrechungen und Füllwörtern gesprochen werden,
- sowie für jede Eingabe den erwarteten Zielzustand nach Anwendung aller inferierten Aktionen (`expected_output_state`).

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
      ...
    }
  ]
}

```

---

## 5. `stop-benchmark`: Evaluierung & Statistische Analyse

Um wiederholte und unnötige API-Latenz zu vermeiden, trennt die Benchmark-Crate strikt zwischen Datengenerierung (SystemOne Requests) und Auswertung:
- **Ein zentrales Main-Binary** (`run-benchmark`) führt einmalig alle Fälle aus dem Dataset vollständig gegen die lokale JevK5-Instanz aus und speichert die Roh-Ausgaben (inkl. pro Pass/Slot Rückgaben, Latenzen, Konfidenzen) in einer Output-Datei (z.B. `benchmark_results.jsonl`).
- **Alle spezialisierten Analyse-Binaries** (`eval-accuracy`, `eval-roc`, `eval-latency` etc.) operieren rein auf diesen gespeicherten Rohdaten und NICHT auf dem Dataset direkt. Dadurch kann beliebig viele Auswertungen fahren, ohne das Model unnötig mehrfach zu befragen.

### 5.1 Ablauf

1. **Durchlauf und Speicherung**
    - `cargo run -p stop-benchmark --bin run-benchmark -- --input data/test_suite.jsonl --output data/benchmark_results.jsonl`
    - Führt alle Fälle mit SystemOne aus, persistiert sämtliche Model-Raw-Outputs, Latenzen und Metadaten.

2. **Analyse-Binaries:**
    - `cargo run -p stop-benchmark --bin eval-accuracy -- --input data/benchmark_results.jsonl`
        - Slot-spezifische Genauigkeit, **Sequence Exact Match** (Durchgänge vollständig korrekt).
    - `cargo run -p stop-benchmark --bin eval-roc -- --input data/benchmark_results.jsonl`
        - ROC-/AUC-Kennzahlen für Slots wie `requires_sterile_confirm` und `further_action_needed`, Kalibrierung.
    - `cargo run -p stop-benchmark --bin eval-latency -- --input data/benchmark_results.jsonl`
        - Latenz-Verteilung (P50, P95, P99) auf Basis der zuvor gemessenen Pass-Latenzen.

### 5.2 Konsolen-Reporting

Die Auswertung erzeugt für jede Analyse eine gut lesbare Markdown-Tabelle im Terminal:

```
+---------------------------+------------+----------+----------+
| Slot / Metrik             | Precision  | Recall   | F1-Score |
+---------------------------+------------+----------+----------+
| target_device             | 0.982      | 0.979    | 0.980    |
| action_kind               | 0.941      | 0.938    | 0.939    |
| step_value                | 0.895      | 0.891    | 0.893    |
| requires_sterile_confirm  | 0.991      | 0.965    | 0.978    |
| further_action_needed     | 0.934      | 0.950    | 0.942    |
+---------------------------+------------+----------+----------+
Sequence Exact Match (SEM): 88.4%
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
* `stop-core`: `RoomState`, `JevDecisionSlots`, `TargetDevice`, `ActionKind` und Serialisierungs-Tests implementieren.
* State-Delta-Logik unit-testen (z. B. `apply_action_to_state` verhindert Drücke > 25 mmHg).

### Phase 2: Mock-Engine & Multi-Pass Loop

* Trait `DecisionEngineProvider` definieren.
* `MockDecisionEngine` schreiben, die konfigurierbare Sequenzen zurückgibt.
* `MultiPassExecutor` implementieren und testen, ob Loops nach `further_action_needed == false` korrekt terminieren.
* HTTP-Client für JevK5 / System-One API implementieren.

### Phase 3: Datensatz-Generator (`stop-dataset`)

* OpenRouter API-Client mit `reqwest` bauen.
* Structured Prompting für synthetische OP-Dialoge aufsetzen.
* Binary `generate-data` erstellen und ersten 100-Zeilen Test-Datensatz in `data/test_suite.jsonl` erzeugen.

### Phase 4: Benchmarking Crate (`stop-benchmark`)

* Parser für `data/test_suite.jsonl` implementieren.
* Metrik-Berechnung für Accuracy, Sequence Exact Match und Latenz schreiben.
* CLI-Reporting mit formatierter Ausgabe fertigstellen.

### Phase 5: GUI & Interaktive Demo (`stop-gui`)

* Winit-Fenster und Vello/2D-Rendering-Pipeline einrichten.
* Raumobjekte zeichnen: OP-Tisch, Lichtkegel, Monitor, Insufflator-Bar.
* Inter-Thread-Channels verbinden: CLI-Eingabe $\to$ Core-Multi-Pass-Engine $\to$ State-Update $\to$ GUI-Repaint.

Nach jeder Phase soll ein Testabschnitt, der /diffx-start-review skill ausgeführt und bei Freigabe der JJ-Commit erstellt werden.

---