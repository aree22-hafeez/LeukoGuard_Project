const form = document.getElementById("cbc-form");
const example = document.getElementById("example");
const compare = document.getElementById("compare");
const results = document.getElementById("results");
const errorBox = document.getElementById("error");
const inputs = new Map();

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

async function loadForm() {
  const response = await fetch("/api/analyzer-panel");
  if (!response.ok) throw new Error("Could not load the analyzer fields. Reload to try again.");
  const schema = await response.json();
  for (const [group, fields] of Object.entries(schema.groups)) {
    const details = document.createElement("details");
    details.className = "group";
    const summary = document.createElement("summary");
    summary.textContent = group;
    const grid = document.createElement("div");
    grid.className = "field-grid";
    for (const {name, unit} of fields) {
      const label = document.createElement("label");
      label.textContent = name + " (" + unit + ")";
      const input = document.createElement("input");
      input.type = "number";
      input.min = "0";
      input.step = "any";
      inputs.set(name, input);
      label.append(input);
      grid.append(label);
    }
    details.append(summary, grid);
    document.getElementById("groups").append(details);
  }
  example.disabled = compare.disabled = false;
}

example.onclick = async () => {
  try {
    const response = await fetch("/panel-example.json");
    if (!response.ok) throw new Error("Could not load the illustrative panel.");
    const values = await response.json();
    for (const [name, input] of inputs) input.value = values[name];
    results.hidden = errorBox.hidden = true;
    document.querySelector("details.group").open = true;
  } catch (error) { showError(error.message); }
};

form.oninput = () => { results.hidden = errorBox.hidden = true; };
form.addEventListener("invalid", event => { event.target.closest("details").open = true; }, true);
form.onsubmit = async event => {
  event.preventDefault();
  results.hidden = errorBox.hidden = true;
  const values = {};
  for (const [name, input] of inputs) {
    if (input.value !== "") values[name] = input.valueAsNumber;
  }
  const fields = document.getElementById("groups");
  fields.inert = true;
  compare.disabled = example.disabled = true;
  try {
    const response = await fetch("/api/analyzer-panel/predict", {
      method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(values)
    });
    const body = await response.json();
    if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : "Check the measurement values and try again.");
    const tbody = document.getElementById("scores");
    tbody.replaceChildren();
    for (const result of body.results) {
      const row = tbody.insertRow();
      for (const value of [result.model, result.score.toFixed(4), result.threshold.toFixed(4),
                          result.flagged ? "Flagged" : "Below threshold"]) {
        row.insertCell().textContent = value;
      }
    }
    document.getElementById("coverage").textContent =
      body.n_entered + " of " + body.n_features + " measurements entered. " +
      (body.n_entered < body.n_features ? "The fitted pipelines imputed the missing measurements. Missing inputs can affect the scores." : "");
    results.hidden = false;
    document.getElementById("results-title").focus();
  } catch (error) { showError(error.message); }
  finally { fields.inert = false; compare.disabled = example.disabled = false; }
};
loadForm().catch(error => showError(error.message));
