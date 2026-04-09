function predictWinner() {
  fetch("/predict_winner", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      team1: team1.value,
      team2: team2.value,
      toss_winner: toss_winner.value,
      toss_decision: toss_decision.value,
      venue: venue.value,
    }),
  })
    .then((res) => res.json())
    .then((data) => {
      result.innerText = "Winner: " + data.winner;
    });
}

function predictToss() {
  fetch("/predict_toss", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      team1: t1.value,
      team2: t2.value,
    }),
  })
    .then((res) => res.json())
    .then((data) => {
      toss_result.innerText = "Toss Winner: " + data.toss_winner;
    });
}

function predictScore() {
  fetch("/predict_score", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      over: over.value,
      cum_runs: runs.value,
    }),
  })
    .then((res) => res.json())
    .then((data) => {
      score_result.innerText = "Predicted Score: " + data.score;
    });
}

function getH2H() {
  fetch(`/api/head2head?t1=${h1.value}&t2=${h2.value}`)
    .then((res) => res.json())
    .then((data) => {
      h2h_result.innerHTML = `
            Matches: ${data.total} <br>
            ${h1.value} Wins: ${data.t1_wins} <br>
            ${h2.value} Wins: ${data.t2_wins}
        `;
    });
}

// Show error
function showError(message) {
  const box = document.getElementById("errorBox");
  box.innerText = message;
  box.classList.remove("d-none");

  setTimeout(() => {
    box.classList.add("d-none");
  }, 3000);
}

// Validate teams
function validateTeams(team1, team2) {
  if (!team1 || !team2) {
    showError("Please select both teams!");
    return false;
  }

  if (team1 === team2) {
    showError("❌ Same team cannot be selected!");
    return false;
  }

  return true;
}
