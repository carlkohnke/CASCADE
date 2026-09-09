# Tracked validation results

Store one directory per validation campaign. Each directory should contain:

- `report.md`: human-readable objective, method, results, failures, and conclusion.
- `result.json`: machine-readable metadata and measurements.
- `commands.txt`: exact invocations in execution order.
- Compact comparison CSV files where useful.
- SHA-256 hashes for inputs and retained raw evidence.

Never replace a failed result with a passing one. Keep both campaign records and link the superseding run.
