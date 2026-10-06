// Minimal DOM helpers + formatters. No framework, no build step.

export function h(tag, props = null, ...children) {
  const node = document.createElement(tag);
  if (props) {
    for (const [key, value] of Object.entries(props)) {
      if (value === null || value === undefined || value === false) continue;
      if (key === 'class') node.className = value;
      else if (key === 'text') node.textContent = String(value);
      else if (key === 'html') node.innerHTML = value;
      else if (key === 'style' && typeof value === 'object') Object.assign(node.style, value);
      else if (key === 'dataset' && typeof value === 'object') Object.assign(node.dataset, value);
      else if (key.startsWith('on') && typeof value === 'function') {
        node.addEventListener(key.slice(2).toLowerCase(), value);
      } else if (key === 'value') node.value = value;
      else if (key === 'selected' || key === 'checked' || key === 'disabled') node[key] = Boolean(value);
      else node.setAttribute(key, value === true ? '' : String(value));
    }
  }
  append(node, children);
  return node;
}

export function append(node, children) {
  for (const child of children.flat(4)) {
    if (child === null || child === undefined || child === false) continue;
    node.appendChild(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

export function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
  return node;
}

export function mount(node, ...children) {
  clear(node);
  append(node, children);
  return node;
}

export function setText(node, value) {
  if (node) node.textContent = String(value);
  return node;
}

export function toggleClass(node, className, active) {
  if (node) node.classList.toggle(className, Boolean(active));
  return node;
}

export function setOptions(select, values, selected, format = (value) => String(value)) {
  clear(select);
  for (const value of values) {
    const option = h('option', { value });
    option.textContent = format(value);
    if (String(value) === String(selected)) option.selected = true;
    select.appendChild(option);
  }
  return select;
}

// ---------------------------------------------------------------- form fields

export function field(labelText, control) {
  return h('div', { class: 'grow' }, h('label', { class: 'field', text: labelText }), control);
}

export function selectField(labelText, values, selected, onChange, format) {
  const node = h('select');
  setOptions(node, values, selected, format);
  node.addEventListener('change', () => onChange(node.value));
  return field(labelText, node);
}

export function numberField(labelText, value, onChange, extra = {}) {
  const node = h('input', { type: 'number', value, ...extra });
  node.addEventListener('change', () => onChange(Number(node.value)));
  return field(labelText, node);
}

export function rangeField(labelText, { min, max, step = 1, value }, onInput) {
  const readout = h('span', { class: 'hint', text: String(value) });
  const node = h('input', { type: 'range', min, max, step, value });
  node.addEventListener('input', () => {
    readout.textContent = String(node.value);
    onInput(Number(node.value));
  });
  return h('div', { class: 'grow' },
    h('label', { class: 'field' }, labelText, ' ', readout),
    node);
}

export function checkboxField(labelText, checked, onChange) {
  const input = h('input', { type: 'checkbox', checked });
  input.addEventListener('change', () => onChange(input.checked));
  return h('label', { class: 'check' }, input, labelText);
}

export function button(labelText, onClick, { className = '', disabled = false, title = '' } = {}) {
  const node = h('button', { class: className, disabled, title }, labelText);
  node.addEventListener('click', onClick);
  return node;
}

export function card(labelText, ...children) {
  return h('div', { class: 'card' },
    labelText ? h('span', { class: 'card-label-top', text: labelText }) : null,
    ...children);
}

export function metric(labelText, value, sub = '') {
  return h('div', { class: 'metric' },
    h('span', { class: 'm-label', text: labelText }),
    h('span', { class: 'm-value', text: value }),
    sub ? h('span', { class: 'm-sub', text: sub }) : null);
}

export function table(columns, rows, { bestIndex = -1 } = {}) {
  const head = h('tr', null, ...columns.map((column) => h('th', { text: column.label })));
  const body = rows.map((row, index) => h('tr', { class: index === bestIndex ? 'best' : '' },
    ...columns.map((column) => h('td', {
      class: column.align === 'right' ? 'num' : '',
      text: column.value(row, index),
    }))));
  return h('table', { class: 'data' }, h('thead', null, head), h('tbody', null, ...body));
}

// ------------------------------------------------------------------ feedback

export function toast(message, kind = 'info', timeout = 5200) {
  const host = document.getElementById('toast-host');
  if (!host) return;
  const node = h('div', { class: `toast ${kind === 'info' ? '' : kind}`, text: message });
  host.appendChild(node);
  if (timeout > 0) {
    setTimeout(() => node.remove(), timeout);
  }
  return node;
}

export function errorBox(message) {
  return message ? h('div', { class: 'err-box', text: message }) : null;
}

export function warnBox(message) {
  return message ? h('div', { class: 'warn-box', text: message }) : null;
}

export function okBox(message) {
  return message ? h('div', { class: 'ok-box', text: message }) : null;
}

export function feed(container, lines, { limit = 14 } = {}) {
  clear(container);
  const items = lines.slice(-limit).reverse();
  if (!items.length) {
    container.appendChild(h('div', { class: 'hint', text: 'Noch keine Ereignisse.' }));
    return container;
  }
  for (const line of items) {
    const text = typeof line === 'string' ? line : line.text;
    const isKill = typeof line === 'string'
      ? /DOWN|WINS|killed|DRAW|ROUND COMPLETE/.test(line)
      : Boolean(line.kill);
    container.appendChild(h('div', { class: isKill ? 'kill' : '', text }));
  }
  return container;
}

export function statusClass(status) {
  if (['running', 'complete'].includes(status)) return 'status-good';
  if (['starting', 'paused', 'stopping'].includes(status)) return 'status-warn';
  return 'status-bad';
}

// ---------------------------------------------------------------- formatting

export const fmt = {
  num(value, digits = 0) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed.toLocaleString('de-DE', { maximumFractionDigits: digits, minimumFractionDigits: digits }) : '—';
  },
  fixed(value, digits = 2) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed.toFixed(digits) : '—';
  },
  percent(value, digits = 1) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? `${(parsed * 100).toFixed(digits)}%` : '—';
  },
  clock(seconds) {
    const total = Math.max(0, Math.floor(Number(seconds) || 0));
    const hours = String(Math.floor(total / 3600)).padStart(2, '0');
    const minutes = String(Math.floor((total % 3600) / 60)).padStart(2, '0');
    const secs = String(total % 60).padStart(2, '0');
    return `${hours}:${minutes}:${secs}`;
  },
  mb(value) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? `${parsed.toLocaleString('de-DE', { maximumFractionDigits: 0 })} MB` : '—';
  },
};

// ------------------------------------------------------------------- polling

export function poller(callback, delay) {
  let timer = null;
  const tick = async () => {
    try {
      await callback();
    } catch (error) {
      console.warn('poll failed', error);
    }
  };
  return {
    start() {
      if (timer !== null) return;
      tick();
      timer = setInterval(tick, delay);
    },
    stop() {
      if (timer === null) return;
      clearInterval(timer);
      timer = null;
    },
    get active() {
      return timer !== null;
    },
  };
}

export function throttle(callback, delay) {
  let pending = false;
  let lastArgs = null;
  return (...args) => {
    lastArgs = args;
    if (pending) return;
    pending = true;
    setTimeout(() => {
      pending = false;
      callback(...lastArgs);
    }, delay);
  };
}
