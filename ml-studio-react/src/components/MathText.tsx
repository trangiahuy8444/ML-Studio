import katex from 'katex';

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function renderInlineSegment(segment: string): string {
  const tokens: string[] = [];
  const stash = (html: string) => {
    const key = `\u0000MLT${tokens.length}\u0000`;
    tokens.push(html);
    return key;
  };

  // Handle inline `code` first so $ inside backticks is rendered literally
  let s = segment.replace(/`([^`]+)`/g, (_, inner: string) => stash(`<code>${escapeHtml(inner)}</code>`));
  // Then render inline math $...$ before **bold** so bold wraps across math spans seamlessly
  s = s.replace(/\$([^$\n]+)\$/g, (_, expr: string) =>
    stash(katex.renderToString(expr.trim(), { displayMode: false, throwOnError: false })),
  );
  // Escape remaining HTML, apply **bold**, and restore tokens
  s = escapeHtml(s).replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  return s.replace(/\u0000MLT(\d+)\u0000/g, (_, idx: string) => tokens[Number(idx)] ?? '');
}

function renderMarkdownMathToHtml(source: string): string {
  const text = source.trim();
  if (!text) return '';

  // First extract display math blocks $$...$$
  const blocks = text.split(/(\$\$[\s\S]*?\$\$)/g);
  const out: string[] = [];

  for (const block of blocks) {
    const trimmed = block.trim();
    if (!trimmed) continue;

    if (trimmed.startsWith('$$') && trimmed.endsWith('$$') && trimmed.length >= 4) {
      const expr = trimmed.slice(2, -2).trim();
      out.push(
        `<div class="math-block">${katex.renderToString(expr, {
          displayMode: true,
          throwOnError: false,
        })}</div>`,
      );
      continue;
    }

    // Split text block by double newlines into paragraphs or bullet lists
    const paragraphs = trimmed.split(/\n\n+/);
    for (const para of paragraphs) {
      const lines = para
        .split('\n')
        .map((l) => l.trimEnd())
        .filter((l) => l.trim().length > 0);
      if (lines.length === 0) continue;

      const allBullets = lines.every((l) => /^[-*]\s+/.test(l.trim()));
      if (allBullets) {
        const items = lines
          .map((l) => `<li>${renderInlineSegment(l.trim().replace(/^[-*]\s+/, ''))}</li>`)
          .join('');
        out.push(`<ul>${items}</ul>`);
      } else {
        // Mixed lines or standard paragraph
        const htmlLines: string[] = [];
        let listBuffer: string[] = [];
        const flushList = () => {
          if (listBuffer.length > 0) {
            out.push(`<ul>${listBuffer.join('')}</ul>`);
            listBuffer = [];
          }
        };
        const flushPara = () => {
          if (htmlLines.length > 0) {
            out.push(`<p>${htmlLines.join('<br/>')}</p>`);
            htmlLines.length = 0;
          }
        };

        for (const line of lines) {
          const t = line.trim();
          if (/^[-*]\s+/.test(t)) {
            flushPara();
            listBuffer.push(`<li>${renderInlineSegment(t.replace(/^[-*]\s+/, ''))}</li>`);
          } else {
            flushList();
            htmlLines.push(renderInlineSegment(t));
          }
        }
        flushList();
        flushPara();
      }
    }
  }

  return out.join('');
}

export function MathText({ text, className = '' }: { text: string; className?: string }) {
  const html = renderMarkdownMathToHtml(text);
  return <div className={`math-text ${className}`.trim()} dangerouslySetInnerHTML={{ __html: html }} />;
}

export function MathBlock({ math }: { math: string }) {
  const html = katex.renderToString(math.trim(), { displayMode: true, throwOnError: false });
  return <div className="math-block" dangerouslySetInnerHTML={{ __html: html }} />;
}
