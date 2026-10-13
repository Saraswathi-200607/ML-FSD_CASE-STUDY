// Charts of the user's own recorded measurements (data comes from json_script).
(function () {
  var el = document.getElementById("chart-data");
  if (!el || typeof Chart === "undefined") return;
  var d = JSON.parse(el.textContent);
  function line(id, label, data, color) {
    var canvas = document.getElementById(id);
    if (!canvas) return;
    new Chart(canvas, {
      type: "line",
      data: { labels: d.labels, datasets: [{ label: label, data: data, borderColor: color,
              backgroundColor: color, tension: 0.2, pointRadius: 4 }] },
      options: { responsive: true, plugins: { legend: { display: true } },
                 scales: { y: { beginAtZero: false } } }
    });
  }
  line("bmiChart", "BMI", d.bmi, "#0d6efd");
  line("weightChart", "Weight (kg)", d.weight, "#198754");
})();
