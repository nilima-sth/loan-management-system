import { useState } from "react";

const PROMPT = `You are a data generation engine simulating customers of a commercial bank in Nepal. Generate exactly 50 highly diverse, realistic customer service messages for the intent category: loan_apply.

Language Distribution:
- 40% Pure Nepali (Devanagari script) — exactly 20 messages
- 40% Romanized Nepali / Nepenglish (e.g., 'Mero account ma loan paincha?') — exactly 20 messages
- 20% English — exactly 10 messages

Rules for Realism:
- Vary the length: some should be 2-word urgent phrases, others full polite sentences with context.
- Include common typos and SMS shorthand (e.g., 'lon', 'aplly', 'plz', 'k garnuparne', 'kti din').
- Use Nepali filler words occasionally (e.g., 'hazur', 'khoi', 'ni', 'ta').
- Cover diverse loan types: home loan, business loan, personal loan, vehicle loan, education loan, agriculture loan, gold loan.
- Vary the persona: farmer, student, small business owner, salaried employee, housewife, trader.
- Some messages should ask about eligibility, some about process/documents, some just stating they want a loan, some asking about interest rates, some following up on existing applications.

Return ONLY a valid JSON array of exactly 50 strings. No markdown, no explanation, no preamble, no backticks.`;

export default function LoanApplyGenerator() {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [rawJson, setRawJson] = useState("");
  const [copied, setCopied] = useState(false);

  const generate = async () => {
    setLoading(true);
    setError("");
    setMessages([]);
    setRawJson("");

    try {
      const response = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "claude-sonnet-4-6",
          max_tokens: 4000,
          messages: [{ role: "user", content: PROMPT }],
        }),
      });

      const data = await response.json();
      const text = data.content
        .filter((b) => b.type === "text")
        .map((b) => b.text)
        .join("");

      const clean = text.replace(/```json|```/g, "").trim();
      const parsed = JSON.parse(clean);
      setMessages(parsed);
      setRawJson(JSON.stringify(parsed, null, 2));
    } catch (err) {
      setError("Generation failed: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  const copyJson = () => {
    navigator.clipboard.writeText(JSON.stringify(messages));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const langLabel = (i) => {
    if (i < 20) return { label: "NP", color: "#C62828", title: "Pure Nepali" };
    if (i < 40) return { label: "RN", color: "#1565C0", title: "Romanized Nepali" };
    return { label: "EN", color: "#2E7D32", title: "English" };
  };

  return (
    <div style={{ fontFamily: "'Segoe UI', sans-serif", maxWidth: 860, margin: "0 auto", padding: 24, background: "#f8f9fa", minHeight: "100vh" }}>
      {/* Header */}
      <div style={{ background: "#8B0000", borderRadius: 12, padding: "20px 28px", marginBottom: 24, color: "#fff" }}>
        <div style={{ fontSize: 11, letterSpacing: 2, textTransform: "uppercase", opacity: 0.75, marginBottom: 4 }}>Data Generation Engine</div>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700 }}>🏦 Nepal Bank · loan_apply</h1>
        <div style={{ marginTop: 6, fontSize: 13, opacity: 0.85 }}>
          50 messages · 20 Pure Nepali · 20 Romanized · 10 English
        </div>
      </div>

      {/* Controls */}
      <div style={{ display: "flex", gap: 12, marginBottom: 24 }}>
        <button
          onClick={generate}
          disabled={loading}
          style={{
            background: loading ? "#999" : "#8B0000",
            color: "#fff",
            border: "none",
            borderRadius: 8,
            padding: "12px 28px",
            fontSize: 15,
            fontWeight: 600,
            cursor: loading ? "not-allowed" : "pointer",
            transition: "background 0.2s",
          }}
        >
          {loading ? "⏳ Generating..." : "⚡ Generate 50 Messages"}
        </button>
        {messages.length > 0 && (
          <button
            onClick={copyJson}
            style={{
              background: copied ? "#2E7D32" : "#fff",
              color: copied ? "#fff" : "#333",
              border: "1.5px solid #ccc",
              borderRadius: 8,
              padding: "12px 22px",
              fontSize: 15,
              fontWeight: 500,
              cursor: "pointer",
            }}
          >
            {copied ? "✅ Copied!" : "📋 Copy JSON"}
          </button>
        )}
      </div>

      {error && (
        <div style={{ background: "#ffebee", border: "1px solid #ef9a9a", borderRadius: 8, padding: 16, color: "#c62828", marginBottom: 20 }}>
          {error}
        </div>
      )}

      {/* Legend */}
      {messages.length > 0 && (
        <div style={{ display: "flex", gap: 16, marginBottom: 16, flexWrap: "wrap" }}>
          {[
            { label: "NP", color: "#C62828", title: "Pure Nepali (Devanagari)", count: 20 },
            { label: "RN", color: "#1565C0", title: "Romanized Nepali", count: 20 },
            { label: "EN", color: "#2E7D32", title: "English", count: 10 },
          ].map((l) => (
            <div key={l.label} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 13, color: "#555" }}>
              <span style={{ background: l.color, color: "#fff", borderRadius: 4, padding: "2px 7px", fontWeight: 700, fontSize: 11 }}>{l.label}</span>
              {l.title} ({l.count})
            </div>
          ))}
        </div>
      )}

      {/* Message List */}
      {messages.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {messages.map((msg, i) => {
            const lang = langLabel(i);
            return (
              <div
                key={i}
                style={{
                  background: "#fff",
                  border: "1px solid #e0e0e0",
                  borderLeft: `4px solid ${lang.color}`,
                  borderRadius: 8,
                  padding: "10px 14px",
                  display: "flex",
                  gap: 12,
                  alignItems: "flex-start",
                }}
              >
                <span style={{ minWidth: 26, color: "#bbb", fontSize: 12, paddingTop: 2 }}>#{i + 1}</span>
                <span
                  style={{
                    background: lang.color,
                    color: "#fff",
                    borderRadius: 4,
                    padding: "2px 6px",
                    fontWeight: 700,
                    fontSize: 10,
                    minWidth: 24,
                    textAlign: "center",
                    marginTop: 2,
                  }}
                  title={lang.title}
                >
                  {lang.label}
                </span>
                <span style={{ fontSize: 14, color: "#222", lineHeight: 1.5, flex: 1 }}>{msg}</span>
              </div>
            );
          })}
        </div>
      )}

      {/* Raw JSON */}
      {rawJson && (
        <details style={{ marginTop: 28 }}>
          <summary style={{ cursor: "pointer", fontWeight: 600, fontSize: 14, color: "#555", padding: "8px 0" }}>
            🗂 Raw JSON Output
          </summary>
          <pre style={{ background: "#1e1e1e", color: "#d4d4d4", borderRadius: 8, padding: 16, fontSize: 12, overflow: "auto", maxHeight: 400, marginTop: 8 }}>
            {rawJson}
          </pre>
        </details>
      )}

      {!messages.length && !loading && (
        <div style={{ textAlign: "center", padding: "60px 20px", color: "#aaa" }}>
          <div style={{ fontSize: 48, marginBottom: 12 }}>🏔</div>
          <div style={{ fontSize: 15 }}>Click "Generate" to produce 50 realistic loan_apply messages from Nepal bank customers.</div>
        </div>
      )}
    </div>
  );
}