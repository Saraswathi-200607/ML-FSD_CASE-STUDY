// Loading state: disable the submit button and show a spinner while the server works.
document.querySelectorAll("form[data-loading-form]").forEach(function (form) {
  form.addEventListener("submit", function () {
    var btn = form.querySelector("button[type=submit]");
    if (!btn) return;
    var spinner = btn.querySelector(".spinner-border");
    var label = btn.querySelector(".btn-label");
    if (spinner) spinner.classList.remove("d-none");
    if (label) label.textContent = "Working...";
    setTimeout(function () { btn.disabled = true; }, 0);
  });
});
