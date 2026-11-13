const button = document.querySelector("#copy");
const command = document.querySelector("#install-command");
const copyStatus = document.querySelector("#status");
if (button instanceof HTMLButtonElement && command && copyStatus) {
  button.hidden = false;
  button.addEventListener("click", async () => {
    button.disabled = true;
    /** @type {number | undefined} */
    let timeout;
    try {
      await Promise.race([
        navigator.clipboard.writeText(command.textContent ?? ""),
        new Promise((_, reject) => {
          timeout = window.setTimeout(
            () => reject(new Error("Clipboard timeout")),
            3000,
          );
        }),
      ]);
      copyStatus.textContent = "Copied.";
      button.setAttribute("data-copied", "");
    } catch {
      const range = document.createRange();
      range.selectNodeContents(command);
      const selection = window.getSelection();
      selection?.removeAllRanges();
      selection?.addRange(range);
      copyStatus.textContent = "Command selected. Copy it from the selection.";
    } finally {
      window.clearTimeout(timeout);
      button.disabled = false;
    }
  });
}
