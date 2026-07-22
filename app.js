const MODEL = {
  means: [2.9848066071, 599.9360505973, 10.9004450691, 2.4987116421, 10.5775601678, 12.4120783213, 0.4868529469],
  standards: [0.5954258996, 172.4102039098, 5.7085185604, 1.6957115182, 0.8480081525, 0.8912273499, 0.1455336164],
  weights: [1.4635156513, 0.665308645, 3.6631495622, -0.6983991644, -0.0078712702, -0.0806359062, 0.0245991866, -0.2726182584],
};

const form = document.querySelector("#loan-form");
const fields = Object.fromEntries(
  ["income", "loan", "fico", "years", "dependents", "assets", "cash"].map((id) => [
    id,
    document.querySelector(`#${id}`),
  ]),
);

const money = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

function calculate() {
  const income = Number(fields.income.value);
  const loan = Number(fields.loan.value);
  const fico = Number(fields.fico.value);
  const years = Number(fields.years.value);
  const dependents = Number(fields.dependents.value);
  const assets = Number(fields.assets.value);
  const cash = Number(fields.cash.value);

  if (![income, loan, fico, years, dependents, assets, cash].every(Number.isFinite) || income <= 0 || years <= 0) {
    return;
  }

  const totalAssets = assets + cash;
  const loanToIncome = loan / (income + 1);
  const loanToAssets = loan / (totalAssets + 1);
  const raw = [loanToIncome, fico, years, dependents, Math.log1p(income), Math.log1p(totalAssets), loanToAssets];
  const scaled = raw.map((value, index) => (value - MODEL.means[index]) / MODEL.standards[index]);
  const z = MODEL.weights[0] + scaled.reduce((sum, value, index) => sum + value * MODEL.weights[index + 1], 0);
  let probability = sigmoid(z);

  let risk;
  let loanPenalty;
  if (loanToIncome < 2) [risk, loanPenalty] = ["LOW", 1];
  else if (loanToIncome < 4) [risk, loanPenalty] = ["MEDIUM", 0.9];
  else if (loanToIncome < 6) [risk, loanPenalty] = ["MODERATE", 0.7];
  else if (loanToIncome < 7) [risk, loanPenalty] = ["HIGH", 0.6];
  else if (loanToIncome < 8) [risk, loanPenalty] = ["VERY HIGH", 0.5];
  else [risk, loanPenalty] = ["EXTREME", 0.3];

  probability *= loanPenalty;
  if (fico < 580) probability *= 0.3;
  else if (fico < 620) probability *= 0.5;
  else if (fico < 670) probability *= 0.75;
  else if (fico < 740) probability *= 0.9;
  else if (fico < 800) probability *= 0.95;

  const monthly = loan / (years * 12);
  const incomeShare = (monthly / (income / 12)) * 100;
  if (incomeShare > 43) probability *= 0.6;
  probability = Math.min(Math.max(probability, 0), 0.97);

  const percent = Math.round(probability * 100);
  document.querySelector("#probability").innerHTML = `${percent}<span>%</span>`;
  document.querySelector("#gauge").style.setProperty("--score", percent);
  document.querySelector("#monthly-payment").textContent = money.format(monthly);
  document.querySelector("#income-share").textContent = `${incomeShare.toFixed(1)}%`;

  const badge = document.querySelector("#risk-badge");
  badge.innerHTML = `<i></i> ${risk}`;
  badge.className = loanToIncome < 2 ? "risk-low" : loanToIncome < 6 ? "risk-medium" : "risk-high";
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  calculate();
});
form.addEventListener("reset", () => window.setTimeout(calculate));
calculate();
