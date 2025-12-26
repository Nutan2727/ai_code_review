import re
from dataclasses import dataclass
from typing import List, Dict, Any
from flask import Flask, request, render_template_string
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------- CONFIG ----------------
MODEL_NAME = "t5-small"

# ---------------- SIMPLE ANALYZER ----------------
@dataclass
class Issue:
    file: str
    line: int
    category: str
    message: str
    snippet: str

def detect_issues(filename: str, text: str) -> List[Issue]:
    issues: List[Issue] = []
    lines = text.splitlines()
    for i, line in enumerate(lines, start=1):
      # Style: long lines
        if len(line) > 120:
            issues.append(Issue(filename, i, "Style", f"Line length {len(line)} exceeds 120 chars", line))
      # Maintainability: TODO comments
        if "TODO" in line:
            issues.append(Issue(filename, i, "Maintainability", "TODO left in code", line))
      # Best practice: print statements
        if "print(" in line:
            issues.append(Issue(filename, i, "BestPractice", "Avoid print in library code; prefer logging", line))
      # Error handling: bare except
        if re.match(r"^\s*except\s*:\s*$", line):
            issues.append(Issue(filename, i, "ErrorHandling", "Bare except detected; catch specific exceptions", line))
    return issues

# ---------------- LLM SUGGESTER ----------------
class LLMSuggester:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    def suggest_for_issue(self, issue: Issue) -> str:
        prompt = (
            f"File: {issue.file}\nLine: {issue.line}\nCategory: {issue.category}\n"
            f"Issue: {issue.message}\nSnippet:\n{issue.snippet}\n\n"
            f"Suggest a fix with explanation."
        )
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model.generate(inputs, max_length=200, num_beams=4)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------------- FLASK APP ----------------
app = Flask(__name__)

TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>AI Code Review Assistant</title>
  <meta charset="utf-8"/>
  <style>
    :root {
      --bg-grad-light: linear-gradient(135deg, #74ebd5 0%, #ACB6E5 100%);
      --primary: #3498db;
      --accent: #16a085;
      --text: #333;
      --card-bg: #fff;
      --code-bg: #f4f4f4;
      --hover: #eaf2f8;
    }
    body {
      font-family: 'Segoe UI', system-ui, -apple-system, Arial, sans-serif;
      margin: 0;
      padding: 0;
      background: var(--bg-grad-light);
      min-height: 100vh;
      color: var(--text);
      transition: background 0.4s ease, color 0.4s ease;
    }
    body.dark {
      --bg-grad-light: linear-gradient(135deg, #1f2937 0%, #0f172a 100%);
      --primary: #60a5fa;
      --accent: #10b981;
      --text: #f4f4f4;
      --card-bg: #1f2937;
      --code-bg: #111827;
      --hover: #334155;
      background: var(--bg-grad-light);
      color: var(--text);
    }
    .container {
      max-width: 1200px;
      margin: auto;
      padding: 40px 24px;
    }
    h2 {
      text-align: center;
      margin: 0 0 24px;
      letter-spacing: 0.5px;
    }
    .subheader {
      text-align: center;
      margin: -8px 0 24px;
      opacity: 0.85;
    }
    .card {
      background: var(--card-bg);
      border-radius: 14px;
      box-shadow: 0 8px 30px rgba(0,0,0,0.12);
      padding: 20px;
      margin-bottom: 28px;
      transition: background 0.4s ease, color 0.4s ease, transform 0.2s ease;
      animation: fadeIn 0.6s ease;
    }
    .card:hover { transform: translateY(-2px); }
    @keyframes fadeIn {
      from {opacity: 0; transform: translateY(8px);}
      to {opacity: 1; transform: translateY(0);}
    }
    label { font-weight: 600; }
    textarea, input[type=file], button {
      width: 100%;
      padding: 12px 12px;
      margin-top: 10px;
      border-radius: 10px;
      border: 1px solid #d1d5db;
      font-size: 14px;
      background: #fff;
      transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    body.dark textarea, body.dark input[type=file] {
      background: #0b1220;
      color: var(--text);
      border: 1px solid #334155;
    }
    textarea:focus, input[type=file]:focus {
      outline: none;
      border-color: var(--primary);
      box-shadow: 0 0 0 3px rgba(52,152,219,0.2);
    }
    button {
      background: linear-gradient(135deg, #6a11cb 0%, var(--primary) 100%);
      color: #fff;
      border: none;
      cursor: pointer;
      transition: opacity 0.2s ease, transform 0.1s ease;
      font-weight: 600;
    }
    button:hover { opacity: 0.92; }
    button:active { transform: scale(0.99); }
    .toggle-btn {
      position: fixed;
      top: -4px;
      right: 18px;
      background: var(--accent);
      color: #fff;
      border: none;
      padding: 10px 14px;
      border-radius: 10px;
      cursor: pointer;
      box-shadow: 0 6px 18px rgba(0,0,0,0.2);
      z-index: 50;
      font-weight: 600;
    }
    .toggle-btn:hover { filter: brightness(1.05); }
    table {
      width: 90%;
      border-collapse: collapse;
      margin-top: 8px;
      border-radius: 10px;
      overflow: hidden;
    }
    th, td {
      padding: 12px;
      border-bottom: 1px solid #e5e7eb;
      text-align: left;
    }
    th {
      background: var(--primary);
      color: #fff;
      font-weight: 700;
      letter-spacing: 0.3px;
    }
    body.dark th { background: #334155; }
    tr:nth-child(even) { background: #f9fafb; }
    body.dark tr:nth-child(even) { background: #0b1220; }
    tr:hover { background: var(--hover); }
    .code {
      background: var(--code-bg);
      padding: 10px;
      border-radius: 8px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 13px;
      line-height: 1.5;
      overflow-x: auto;
    }
    .charts-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
      align-items: stretch;
      justify-items: center;
    }
    .chart-card {
      padding: 12px;
      border-radius: 12px;
      background: var(--card-bg);
      box-shadow: 0 6px 18px rgba(0,0,0,0.08);
      width: 100%;
      max-width: 380px;
    }
    .chart-title {
      font-weight: 700;
      margin: 4px 8px 8px;
      text-align: center;
    }
    .chart-wrap {
      position: relative;
      width: 100%;
      height: 280px; /* medium height */
    }
  </style>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
  <button class="toggle-btn" onclick="toggleDarkMode()">🌙 Toggle Dark Mode</button>
  <div class="container">
    <h2>🌟 AI POWERED CODE REVIEW ASSISSTANCE 🌟</h2>
    <p class="subheader">Analyze code, visualize issues, and get AI-powered suggestions</p>
    
    <div class="card">
      <form method="POST" enctype="multipart/form-data">
        <label>Paste your code:</label>
        <textarea name="code" rows="10" placeholder="Paste any code here...">{{ code or "" }}</textarea>
        <label style="margin-top:14px;">Or upload a file:</label>
        <input type="file" name="file">
        <button type="submit" style="margin-top:14px;">Analyze Code</button>
      </form>
    </div>

    {% if metrics %}
    <div class="card">
      <h3 style="text-align:center; margin: 8px 0 18px;">📊 ISSUES OVERVIEW </h3>
      <div class="charts-grid">
        <div class="chart-card">
          <div class="chart-title">Bar</div>
          <div class="chart-wrap"><canvas id="barChart"></canvas></div>
        </div>
        <div class="chart-card">
          <div class="chart-title">Doughnut</div>
          <div class="chart-wrap"><canvas id="doughnutChart"></canvas></div>
        </div>
        <div class="chart-card">
          <div class="chart-title">Line</div>
          <div class="chart-wrap"><canvas id="lineChart"></canvas></div>
        </div>
        <div class="chart-card">
          <div class="chart-title">Polar Area</div>
          <div class="chart-wrap"><canvas id="polarChart"></canvas></div>
        </div>
      </div>
    </div>

    <div class="card">
      <h3 style="text-align:center; margin: 8px 0 18px;">🔎 DETAILED FINDINGS </h3>
      <table>
        <thead>
          <tr>
            <th>File</th><th>Line</th><th>Category</th><th>Message</th><th>Snippet</th><th>AI Suggestion</th>
          </tr>
        </thead>
        <tbody>
          {% for m in metrics %}
          <tr>
            <td>{{ m.file }}</td>
            <td>{{ m.line }}</td>
            <td>{{ m.category }}</td>
            <td>{{ m.message }}</td>
            <td><div class="code">{{ m.snippet }}</div></td>
            <td><div class="code">{{ m.suggestion }}</div></td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>

    <script>
      const metrics = {{ metrics|tojson }};
      const counts = {};
      metrics.forEach(m => { counts[m.category] = (counts[m.category] || 0) + 1; });

      const labels = Object.keys(counts);
      const values = Object.values(counts);
      const colors = ['#e74c3c','#3498db','#2ecc71','#f1c40f','#9b59b6','#1abc9c','#fd79a8','#55efc4'];

      // Chart.js global defaults responsive medium sizing
      Chart.defaults.responsive = true;
      Chart.defaults.maintainAspectRatio = false;

      const commonOpts = {
        plugins: {
          legend: { position: 'bottom', labels: { boxWidth: 12, padding: 10 } },
          tooltip: { enabled: true }
        },
        scales: {
          x: { ticks: { color: getTextColor() }, grid: { display: false } },
          y: { ticks: { color: getTextColor() }, grid: { color: getGridColor() } }
        }
      };

      function getTextColor() {
        return document.body.classList.contains('dark') ? '#e5e7eb' : '#333';
      }
      function getGridColor() {
        return document.body.classList.contains('dark') ? '#334155' : '#e5e7eb';
      }

      const barCtx = document.getElementById('barChart').getContext('2d');
      const doughnutCtx = document.getElementById('doughnutChart').getContext('2d');
      const lineCtx = document.getElementById('lineChart').getContext('2d');
      const polarCtx = document.getElementById('polarChart').getContext('2d');

      let barChart = new Chart(barCtx, {
        type: 'bar',
        data: { labels, datasets: [{ label: 'Count', data: values, backgroundColor: colors }] },
        options: commonOpts
      });

      let doughnutChart = new Chart(doughnutCtx, {
        type: 'doughnut',
        data: { labels, datasets: [{ data: values, backgroundColor: colors }] },
        options: { plugins: { legend: { position: 'bottom' } } }
      });

      let lineChart = new Chart(lineCtx, {
        type: 'line',
        data: { labels, datasets: [{ label: 'Count', data: values, fill: false, borderColor: '#6a11cb', tension: 0.3, pointBackgroundColor: '#6a11cb' }] },
        options: { plugins: { legend: { display: false } }, scales: { x: { ticks: { color: getTextColor() } }, y: { ticks: { color: getTextColor() }, grid: { color: getGridColor() } } } }
      });

      let polarChart = new Chart(polarCtx, {
        type: 'polarArea',
        data: { labels, datasets: [{ data: values, backgroundColor: colors }] },
        options: { plugins: { legend: { position: 'bottom' } } }
      });

      // Update charts when theme toggles so tick/grid colors adapt
      function refreshChartColors() {
        const isDark = document.body.classList.contains('dark');
        const tickColor = isDark ? '#e5e7eb' : '#333';
        const gridColor = isDark ? '#334155' : '#e5e7eb';

        [barChart, lineChart].forEach(chart => {
          if (chart.options.scales) {
            chart.options.scales.x.ticks.color = tickColor;
            chart.options.scales.y.ticks.color = tickColor;
            if (chart.options.scales.y.grid) chart.options.scales.y.grid.color = gridColor;
          }
          chart.update();
        });
      }

      function toggleDarkMode() {
        document.body.classList.toggle('dark');
        refreshChartColors();
      }
      window.toggleDarkMode = toggleDarkMode; // expose to button
    </script>
    {% endif %}
  </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    code = ""
    metrics: List[Dict[str, Any]] = []
    if request.method == "POST":
        code = request.form.get("code", "")
        file = request.files.get("file")
        if file and file.filename:
            code = file.read().decode("utf-8", errors="replace")
        if code.strip():
            issues = detect_issues("uploaded_code.py", code)
            suggester = LLMSuggester()
            for issue in issues:
                suggestion = suggester.suggest_for_issue(issue)
                metrics.append({
                    "file": issue.file,
                    "line": issue.line,
                    "category": issue.category,
                    "message": issue.message,
                    "snippet": issue.snippet,
                    "suggestion": suggestion
                })
    return render_template_string(TEMPLATE, code=code, metrics=metrics)

# ---------------- MAIN ----------------
if __name__ == "__main__":
    print("Launching AI Code Review Assistant at http://localhost:5000")
    app.run(debug=True)



