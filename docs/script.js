const tabs = [...document.querySelectorAll("[role='tab']")];
const panels = [...document.querySelectorAll("[role='tabpanel']")];
const sessionNames = new Set(tabs.map((tab) => tab.dataset.session));

function activateSession(session, { updateHash = true, moveFocus = false } = {}) {
  if (!sessionNames.has(session)) return;

  tabs.forEach((tab) => {
    const selected = tab.dataset.session === session;
    tab.classList.toggle("is-active", selected);
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
    if (selected && moveFocus) tab.focus();
  });

  panels.forEach((panel) => {
    const selected = panel.dataset.panel === session;
    panel.classList.toggle("is-active", selected);
    panel.hidden = !selected;
  });

  if (updateHash && window.location.hash !== `#${session}`) {
    window.history.pushState(null, "", `#${session}`);
  }
}

tabs.forEach((tab, index) => {
  tab.addEventListener("click", () => activateSession(tab.dataset.session));

  tab.addEventListener("keydown", (event) => {
    let nextIndex = index;

    if (event.key === "ArrowRight") nextIndex = (index + 1) % tabs.length;
    if (event.key === "ArrowLeft") nextIndex = (index - 1 + tabs.length) % tabs.length;
    if (event.key === "Home") nextIndex = 0;
    if (event.key === "End") nextIndex = tabs.length - 1;
    if (nextIndex === index) return;

    event.preventDefault();
    activateSession(tabs[nextIndex].dataset.session, { moveFocus: true });
    tabs[nextIndex].scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
  });
});

document.querySelectorAll("[data-copy]").forEach((button) => {
  button.addEventListener("click", async () => {
    const target = document.getElementById(button.dataset.copy);
    if (!target) return;

    const originalLabel = button.textContent;
    try {
      await navigator.clipboard.writeText(target.innerText);
      button.textContent = "Copied!";
    } catch {
      button.textContent = "Select text";
    }

    window.setTimeout(() => {
      button.textContent = originalLabel;
    }, 1800);
  });
});

window.addEventListener("hashchange", () => {
  const session = window.location.hash.slice(1);
  if (sessionNames.has(session)) activateSession(session, { updateHash: false });
});

const requestedSession = window.location.hash.slice(1);
activateSession(sessionNames.has(requestedSession) ? requestedSession : "day1", {
  updateHash: false,
});
