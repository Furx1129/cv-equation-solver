/**
 * Keep Gradio Sketchpad on brush after clear/eraser and on first paint.
 * Bound to #cv-equation-sketchpad (see ui/app.py).
 */
function cvSketchToolLabel(btn) {
  if (!btn) return "";
  return (
    btn.getAttribute("aria-label") ||
    btn.getAttribute("data-label") ||
    btn.title ||
    btn.textContent ||
    ""
  ).toLowerCase();
}

function cvSelectSketchBrush() {
  const root = document.getElementById("cv-equation-sketchpad");
  if (!root) return false;

  const brushKeys = ["brush", "画笔", "pen", "铅笔", "draw"];
  const buttons = root.querySelectorAll("button");
  for (const btn of buttons) {
    const label = cvSketchToolLabel(btn);
    if (brushKeys.some((key) => label.includes(key))) {
      btn.click();
      return true;
    }
  }

  const toolbar =
    root.querySelector('[class*="toolbar"]') ||
    root.querySelector('[class*="tool"]');
  if (toolbar) {
    const first = toolbar.querySelector("button");
    if (first) {
      first.click();
      return true;
    }
  }
  return false;
}

function cvShouldReturnToBrush(btn) {
  const label = cvSketchToolLabel(btn);
  const resetKeys = [
    "eraser",
    "橡皮",
    "清除",
    "clear",
    "delete",
    "删除",
    "undo",
    "撤销",
    "reset",
  ];
  return resetKeys.some((key) => label.includes(key));
}

function cvBindSketchpadToolReset() {
  const root = document.getElementById("cv-equation-sketchpad");
  if (!root || root.dataset.cvToolsBound === "1") return;
  root.dataset.cvToolsBound = "1";

  root.addEventListener(
    "click",
    (event) => {
      const btn = event.target.closest("button");
      if (!btn || !root.contains(btn)) return;
      if (!cvShouldReturnToBrush(btn)) return;
      setTimeout(cvSelectSketchBrush, 0);
      setTimeout(cvSelectSketchBrush, 150);
      setTimeout(cvSelectSketchBrush, 400);
    },
    true,
  );
}

function cvInitSketchpad() {
  cvBindSketchpadToolReset();
  cvSelectSketchBrush();
  let attempts = 0;
  const timer = setInterval(() => {
    cvSelectSketchBrush();
    attempts += 1;
    if (attempts >= 24) clearInterval(timer);
  }, 250);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", cvInitSketchpad);
} else {
  cvInitSketchpad();
}
