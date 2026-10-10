//! ASCII table rendering for the evaluation console reports (spec 5.2).

/// Renders a titled section: one title line, then an ASCII box table.
/// Used for each split (overall, per field, per scenario, ...).
pub fn render_section(title: &str, headers: &[&str], rows: &[Vec<String>]) -> String {
    format!("{title}\n{}", render_table(headers, rows))
}

/// Renders an ASCII box table: `+---+` borders, one column per header.
pub fn render_table(headers: &[&str], rows: &[Vec<String>]) -> String {
    let columns = headers.len();
    let mut widths: Vec<usize> = headers.iter().map(|h| h.chars().count()).collect();
    for row in rows {
        for (i, cell) in row.iter().enumerate().take(columns) {
            widths[i] = widths[i].max(cell.chars().count());
        }
    }

    let border = |output: &mut String| {
        output.push('+');
        for width in &widths {
            output.push_str(&"-".repeat(width + 2));
            output.push('+');
        }
        output.push('\n');
    };

    let mut output = String::new();
    border(&mut output);
    output.push('|');
    for (header, width) in headers.iter().zip(&widths) {
        output.push_str(&format!(" {header:<width$} |", width = width));
    }
    output.push('\n');
    border(&mut output);
    for row in rows {
        output.push('|');
        for (i, width) in widths.iter().enumerate() {
            let cell = row.get(i).map(String::as_str).unwrap_or("");
            output.push_str(&format!(" {cell:<width$} |", width = width));
        }
        output.push('\n');
    }
    border(&mut output);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_section_with_title() {
        let section = render_section(
            "Per field",
            &["Field", "Accuracy"],
            &[vec!["light_brightness".to_string(), "0.900".to_string()]],
        );
        assert!(section.starts_with("Per field\n+"), "{section}");
        assert!(section.contains("light_brightness"), "{section}");
    }

    #[test]
    fn renders_boxed_table() {
        let table = render_table(
            &["Metric", "Value"],
            &[vec!["accuracy".to_string(), "0.500".to_string()]],
        );
        let expected = "\
+----------+-------+
| Metric   | Value |
+----------+-------+
| accuracy | 0.500 |
+----------+-------+
";
        assert_eq!(table, expected);
    }
}
