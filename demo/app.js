"use strict";

const form = document.querySelector("#profile-form");
const fieldsContainer = document.querySelector("#field-groups");
const exampleButton = document.querySelector("#example-button");
const predictButton = document.querySelector("#predict-button");
const formStatus = document.querySelector("#form-status");
const requestError = document.querySelector("#request-error");
const resultPanel = document.querySelector(".results-panel");
const resultsStatus = document.querySelector("#results-status");
const resultContent = document.querySelector("#result-content");
const emptyState = document.querySelector("#empty-state");
let config;
let lastResult = null;
let profileVersion = 0;

const groups = [
  {title: "Interests & goals", names: ["Fields_of_Interest", "Preferred_Work_Style", "Work_Type_Interest", "Willing_to_Relocate"]},
  {title: "Education & skills", names: ["Highest_qualification", "Stream", "Current_Academic_Level", "Grade_CGPA_Percentage", "Technical_Skills", "Soft_Skills", "Certifications"]},
  {title: "Experience & background", names: ["Age", "Gender", "Location", "Languages_Known", "Past_Jobs_Internships", "Achievements", "Skills_Gained"]},
];
const labels = {
  Fields_of_Interest: "What interests you most?", Preferred_Work_Style: "Preferred work style",
  Work_Type_Interest: "Type of work", Willing_to_Relocate: "Open to relocating?",
  Highest_qualification: "Highest qualification", Stream: "Field of study",
  Current_Academic_Level: "Current stage", Grade_CGPA_Percentage: "Grade (percentage)",
  Technical_Skills: "Technical skills", Soft_Skills: "People & thinking skills",
  Certifications: "Certifications", Age: "Age", Gender: "Gender", Location: "Location",
  Languages_Known: "Languages", Past_Jobs_Internships: "Jobs or internships",
  Achievements: "Achievements", Skills_Gained: "Skills gained from experience",
};
const placeholders = {
  Technical_Skills: "e.g. Python, SQL, Data Analysis", Soft_Skills: "e.g. Communication, Teamwork",
  Certifications: "e.g. Python, AWS", Location: "e.g. Singapore", Languages_Known: "e.g. English",
  Past_Jobs_Internships: "e.g. Research Assistant", Achievements: "e.g. Hackathon Winner",
  Skills_Gained: "e.g. Programming, Leadership", Age: "e.g. 24", Grade_CGPA_Percentage: "e.g. 85",
};

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function createField(field) {
  const wide = ["Fields_of_Interest", "Technical_Skills", "Soft_Skills", "Certifications", "Past_Jobs_Internships", "Achievements", "Skills_Gained"].includes(field.name);
  const wrapper = element("div", `field${wide ? " wide" : ""}`);
  const label = element("label", "", labels[field.name] || field.label);
  label.htmlFor = field.name;
  if (field.required) label.append(element("span", "required-label", "Required"));
  const input = element(field.type === "select" ? "select" : "input");
  input.name = field.name;
  input.id = field.name;
  input.required = Boolean(field.required);
  if (field.type === "select") {
    input.add(new Option(field.required ? "Choose an interest" : "Select (optional)", ""));
    for (const value of field.options) input.add(new Option(value, value));
  } else {
    input.type = field.type === "number" ? "number" : "text";
    input.placeholder = placeholders[field.name] || field.placeholder || "Optional";
    input.maxLength = 1000;
    if (field.type === "number") {
      input.min = field.min ?? 0;
      input.max = field.max ?? 100;
      input.step = field.name === "Age" ? "1" : "any";
    }
  }
  wrapper.append(label, input);
  if (["Technical_Skills", "Soft_Skills", "Certifications"].includes(field.name)) {
    const hint = element("span", "field-hint", "Separate multiple items with commas.");
    hint.id = `${field.name}-hint`;
    input.setAttribute("aria-describedby", hint.id);
    wrapper.append(hint);
  }
  if (field.name === "Grade_CGPA_Percentage") {
    const hint = element("span", "field-hint", "Use 0–100, rather than a GPA out of 4 or 5.");
    hint.id = "grade-hint";
    input.setAttribute("aria-describedby", hint.id);
    wrapper.append(hint);
  }
  return wrapper;
}

function buildForm() {
  fieldsContainer.replaceChildren();
  const byName = new Map(config.fields.map(field => [field.name, field]));
  groups.forEach((group, index) => {
    const section = element("details", "field-group");
    section.open = index === 0;
    const summary = element("summary");
    summary.append(element("span", "section-number", `0${index + 1}`), document.createTextNode(group.title));
    const grid = element("div", "field-grid");
    group.names.forEach(name => grid.append(createField(byName.get(name))));
    section.append(summary, grid);
    fieldsContainer.append(section);
  });
}

function clearResults(message) {
  profileVersion += 1;
  lastResult = null;
  resultContent.hidden = true;
  emptyState.hidden = false;
  requestError.hidden = true;
  formStatus.textContent = message;
  document.querySelector("#results-subtitle").textContent = "Three suggestions, shaped by your profile.";
}

function setBusy(busy) {
  predictButton.disabled = busy;
  resultPanel.setAttribute("aria-busy", String(busy));
  document.querySelector("#predict-label").textContent = busy ? "Finding your careers…" : "Explore my careers";
}

function renderResults(result) {
  const list = document.querySelector("#recommendation-list");
  list.replaceChildren();
  result.recommendations.forEach(item => {
    const row = element("li", "recommendation");
    const title = element("div", "recommendation-top");
    title.append(element("span", "rank", `0${item.rank}`), element("h3", "", item.career));
    const score = element("div", "score-row");
    score.append(element("span", "", "MODEL SCORE"), element("span", "score-value", `${(item.score * 100).toFixed(1)}%`));
    const track = element("div", "score-track");
    track.setAttribute("aria-hidden", "true");
    const fill = element("div", "score-fill");
    fill.style.width = `${Math.max(0, Math.min(100, item.score * 100))}%`;
    track.append(fill);
    row.append(title, score, track);
    list.append(row);
  });
  document.querySelector("#results-subtitle").textContent = `Exploring possibilities in ${result.profile.Fields_of_Interest}.`;
  document.querySelector("#model-label").textContent = `Powered by ${result.model}`;
  emptyState.hidden = true;
  resultContent.hidden = false;
  resultsStatus.textContent = `Your three suggestions are ${result.recommendations.map(item => item.career).join(", ")}.`;
}

exampleButton.addEventListener("click", () => {
  config.fields.forEach(field => { form.elements.namedItem(field.name).value = config.example[field.name] ?? ""; });
  clearResults("Example profile loaded. You can edit any field before exploring.");
  document.querySelector(".field-group").open = true;
  resultsStatus.textContent = "Example loaded. Select Explore my careers to generate suggestions.";
});

form.addEventListener("input", () => clearResults("Profile updated. Explore again to refresh your suggestions."));
form.addEventListener("invalid", event => {
  const group = event.target.closest("details");
  if (group) group.open = true;
}, true);

form.addEventListener("submit", async event => {
  event.preventDefault();
  const profile = Object.fromEntries(config.fields.map(field => [field.name, form.elements.namedItem(field.name).value.trim()]));
  const version = profileVersion;
  requestError.hidden = true;
  setBusy(true);
  resultsStatus.textContent = "Finding your career suggestions.";
  try {
    const response = await fetch("/api/predict", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({profile})});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Unable to generate suggestions. Please try again.");
    if (version !== profileVersion) {
      resultsStatus.textContent = "Your profile changed. Select Explore my careers again.";
      return;
    }
    lastResult = result;
    renderResults(result);
    formStatus.textContent = "Your shortlist is ready. Change a field to explore another profile.";
    if (window.matchMedia("(max-width: 700px)").matches) resultPanel.scrollIntoView({behavior: "smooth", block: "start"});
  } catch (error) {
    if (version !== profileVersion) {
      resultsStatus.textContent = "Your profile changed. Select Explore my careers again.";
      return;
    }
    lastResult = null;
    resultContent.hidden = true;
    emptyState.hidden = false;
    requestError.textContent = error instanceof TypeError ? "Could not reach the demo server. Check that it is still running, then try again." : error.message;
    requestError.hidden = false;
    resultsStatus.textContent = "Suggestions could not be generated.";
  } finally {
    setBusy(false);
  }
});

function csvCell(value) {
  let text = String(value ?? "");
  // Keep free-text profile values from being interpreted as spreadsheet formulas.
  if (typeof value === "string" && /^[\s]*[=+\-@\t\r]/.test(text)) text = `'${text}`;
  return `"${text.replaceAll('"', '""')}"`;
}

document.querySelector("#download-button").addEventListener("click", () => {
  if (!lastResult) return;
  const row = {...lastResult.profile};
  lastResult.recommendations.forEach(item => {
    row[`Recommendation_${item.rank}`] = item.career;
    row[`Recommendation_${item.rank}_Score`] = item.score;
  });
  const csv = [Object.keys(row), Object.values(row)].map(values => values.map(csvCell).join(",")).join("\r\n");
  const url = URL.createObjectURL(new Blob(["\uFEFF", csv, "\r\n"], {type: "text/csv;charset=utf-8"}));
  const link = document.createElement("a");
  link.href = url;
  link.download = "my-career-shortlist.csv";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

async function initialize() {
  try {
    const response = await fetch("/api/config");
    if (!response.ok) throw new Error("Could not load the profile options. Please refresh the page.");
    config = await response.json();
    buildForm();
    document.querySelector("#profile-fields").disabled = false;
    exampleButton.disabled = false;
    predictButton.disabled = false;
    document.querySelector("#model-label").textContent = `Powered by ${config.model}`;
  } catch (error) {
    const message = document.querySelector("#startup-error");
    message.textContent = "The demo could not connect to its server. Start demo.cmd and refresh this page.";
    message.hidden = false;
    fieldsContainer.replaceChildren();
  }
}

initialize();
