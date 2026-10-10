use tokio::runtime::{Handle, Runtime};

/// Blockiert den aktuellen Thread, bis das Future fertig ist – egal wo es aufgerufen wird.
pub fn block_on_anywhere<F: Future>(future: F) -> F::Output {
    match Handle::try_current() {
        // Fall A: Wir sind BEREITS in einem Tokio-Async-Kontext.
        // Ein neues `block_on` würde hier zum Absturz führen.
        Ok(handle) => {
            // Wir nutzen die bestehende Laufzeit, um das Future zu blockieren
            tokio::task::block_in_place(move || {
                handle.block_on(future)
            })
        }
        // Fall B: Wir sind OUTSIDE (in normalem, synchronen Code).
        // Wir müssen eine eigene temporäre Laufzeit starten.
        Err(_) => {
            let rt = Runtime::new().unwrap();
            rt.block_on(future)
        }
    }
}
