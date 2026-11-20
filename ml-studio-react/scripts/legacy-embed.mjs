import fs from 'node:fs';

const source = new URL('../../linear-regression-lab/', import.meta.url);
const output = new URL('../public/legacy/index.html', import.meta.url);
const read = name => fs.readFileSync(new URL(name, source), 'utf8');
const styles = new Set(['style.css', 'studio.css']);
const scripts = new Set(['vendor/plotly.min.js', 'iris-data.js', 'engine.js', 'app.js', 'studio.js']);

let html = read('legacy-index.html');
html = html.replace(/<link rel="stylesheet" href="([^"]+)">/g, (_, name) => {
  if (!styles.has(name)) return '';
  const css = read('dist/' + name).replace(/^@import[^\n]+\n/, '');
  return '<style>' + css + '</style>';
});
html = html.replace(/<script src="([^"]+)"><\/script>/g, (_, name) => (
  scripts.has(name) ? '<script>' + read('dist/' + name).replace(/<\/script/gi, '<\\/script') + '</script>' : ''
));
html = html.replace('</head>', `<style>
  :root { --bg: #f3f5fd; --ink: #202753; }
  body { font-family: system-ui, sans-serif; overflow-wrap: anywhere; }
  body > .sidebar { display: block; position: static; width: auto; padding: 10px 18px; border: 0; }
  .sidebar > :not(nav), #notebookNav, #cinemaToggle { display: none; }
  .sidebar nav { display: flex; flex-wrap: wrap; gap: 6px; }
  .sidebar .lesson { flex: 1 1 140px; width: auto; margin: 0; }
  .sidebar .lesson small { display: none; }
  body > main { margin: 0; max-width: none; padding: 0 18px; display: flow-root; }
  main > header { min-height: 42px; height: auto; }
  .intro { padding-top: 22px; }
  @media(max-width: 650px) {
    body > main, body > .sidebar { padding-inline: 9px; }
    .sidebar .lesson { flex-basis: 110px; }
  }
</style></head>`);
html = html.replace('</body>', `<script>
  const embedMain = document.querySelector('main');
  const embedNav = document.querySelector('body > .sidebar');
  function reportHeight() {
    const height = Math.ceil(embedNav.getBoundingClientRect().height + embedMain.getBoundingClientRect().height);
    if (window.parent !== window) window.parent.postMessage({type: 'ml-regression-height', height}, '*');
  }
  const embedObserver = new ResizeObserver(reportHeight);
  embedObserver.observe(embedMain);
  embedObserver.observe(embedNav);
  window.addEventListener('load', reportHeight);
  // Keep the lesson notes anchored to the control when the frame is taller than the viewport.
  document.querySelectorAll('#helpBtn, #guideBtn').forEach(button => button.addEventListener('click', () => {
    const guide = document.querySelector('#guide');
    guide.style.position = 'absolute';
    guide.style.margin = '0 auto';
    guide.style.top = Math.max(8, button.getBoundingClientRect().bottom + 8) + 'px';
    guide.style.bottom = 'auto';
  }));
</script></body>`);
fs.writeFileSync(output, html);
