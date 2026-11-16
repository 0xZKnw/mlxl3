// Line icons drawn to match the SF Symbols used by the Mac app (1.6 px strokes, 20 px grid).
const P = {
  compose: '<path d="M10.5 3.5H5.2A1.7 1.7 0 0 0 3.5 5.2v9.6a1.7 1.7 0 0 0 1.7 1.7h9.6a1.7 1.7 0 0 0 1.7-1.7V9.5"/><path d="m8 12 .5-2.6 6.7-6.7a1.4 1.4 0 0 1 2 2l-6.7 6.7Z"/>',
  search: '<circle cx="8.8" cy="8.8" r="5.3"/><path d="m12.7 12.7 4 4"/>',
  chat: '<path d="M4.5 4.5h11a1.5 1.5 0 0 1 1.5 1.5v6.6a1.5 1.5 0 0 1-1.5 1.5H9l-3.6 2.6v-2.6h-.9A1.5 1.5 0 0 1 3 12.6V6a1.5 1.5 0 0 1 1.5-1.5Z"/>',
  stack: '<rect x="4" y="7.5" width="12" height="9" rx="1.6"/><path d="M5.8 5.2h8.4M7.6 3h4.8"/>',
  gear: '<circle cx="10" cy="10" r="2.6"/><path d="M10 2.8v2M10 15.2v2M2.8 10h2M15.2 10h2M4.9 4.9l1.4 1.4M13.7 13.7l1.4 1.4M4.9 15.1l1.4-1.4M13.7 6.3l1.4-1.4"/>',
  sidebar: '<rect x="2.8" y="4" width="14.4" height="12" rx="2"/><path d="M7.6 4v12"/>',
  sliders: '<path d="M3.5 6h6M13.5 6h3M3.5 14h3M10.5 14h6"/><circle cx="11.5" cy="6" r="1.8"/><circle cx="8.5" cy="14" r="1.8"/>',
  chevronDown: '<path d="m6 8 4 4 4-4"/>',
  chevronRight: '<path d="m8 5.5 4.5 4.5L8 14.5"/>',
  arrowLeft: '<path d="M16 10H4.5M9 5.5 4.5 10 9 14.5"/>',
  arrowUp: '<path d="M10 16V4.5M5.5 9 10 4.5 14.5 9"/>',
  arrowDown: '<path d="M10 4v11.5M5.5 11 10 15.5 14.5 11"/>',
  arrowUpRight: '<path d="M6 14 14 6M7.5 6H14v6.5"/>',
  download: '<path d="M10 3.5v9M6 9l4 4 4-4M4 16.5h12"/>',
  update: '<circle cx="10" cy="10" r="7"/><path d="M10 6.2v7M7 10.6l3 3 3-3"/>',
  stop: '<rect x="6" y="6" width="8" height="8" rx="1.2" fill="currentColor" stroke="none"/>',
  plus: '<path d="M10 4.5v11M4.5 10h11"/>',
  copy: '<rect x="7" y="7" width="9" height="9" rx="1.6"/><path d="M13 4.8V4.6A1.6 1.6 0 0 0 11.4 3H4.6A1.6 1.6 0 0 0 3 4.6v6.8A1.6 1.6 0 0 0 4.6 13h.2"/>',
  check: '<path d="m4.5 10.5 3.5 3.5 7.5-8"/>',
  pencil: '<path d="m4 16 .7-3.3 8.6-8.6a1.6 1.6 0 0 1 2.3 2.3L7 15.3Z"/>',
  retry: '<path d="M15.5 10a5.5 5.5 0 1 1-1.6-3.9"/><path d="M15.8 3.6v3.2h-3.2"/>',
  pin: '<path d="M7.5 3.5h5l-.8 4.5 2.8 2.8H5.5L8.3 8Z"/><path d="M10 10.8V16.5"/>',
  trash: '<path d="M4 5.5h12M8 5.5V4h4v1.5M5.5 5.5l.8 10.5h7.4l.8-10.5"/>',
  folder: '<path d="M3 6a1.5 1.5 0 0 1 1.5-1.5h3.3l1.7 1.8h6A1.5 1.5 0 0 1 17 7.8v6.7a1.5 1.5 0 0 1-1.5 1.5h-11A1.5 1.5 0 0 1 3 14.5Z"/>',
  folderPlus: '<path d="M3 6a1.5 1.5 0 0 1 1.5-1.5h3.3l1.7 1.8h6A1.5 1.5 0 0 1 17 7.8v6.7a1.5 1.5 0 0 1-1.5 1.5h-11A1.5 1.5 0 0 1 3 14.5Z"/><path d="M10 8.5v5M7.5 11h5"/>',
  eject: '<path d="M10 4 15.5 11h-11Z"/><path d="M4.5 15h11"/>',
  hex: '<path d="M10 2.8 16.2 6.4v7.2L10 17.2l-6.2-3.6V6.4Z"/><path d="M10 10 16.2 6.4M10 10v7.2M10 10 3.8 6.4"/>',
  lock: '<rect x="5" y="9" width="10" height="7.5" rx="1.5"/><path d="M7 9V6.8a3 3 0 0 1 6 0V9"/>',
  cpu: '<rect x="5" y="5" width="10" height="10" rx="1.6"/><path d="M8 2.8V5M12 2.8V5M8 15v2.2M12 15v2.2M2.8 8H5M2.8 12H5M15 8h2.2M15 12h2.2"/>',
  sync: '<path d="M15.6 8A5.8 5.8 0 0 0 5 6.2M4.4 12A5.8 5.8 0 0 0 15 13.8"/><path d="M15.8 3.8V8h-4.2M4.2 16.2V12h4.2"/>',
  wave: '<path d="M2.5 10c1.5 0 1.5-4 3-4s1.5 8 3 8 1.5-8 3-8 1.5 8 3 8 1.5-4 3-4"/>',
  photo: '<rect x="3" y="4.5" width="14" height="11" rx="1.6"/><circle cx="7.2" cy="8.3" r="1.3"/><path d="m3.5 14 4-3.6 3 2.6 2.5-2 3.5 3"/>',
  sparkles: '<path d="M8.5 3.5 9.8 7.2l3.7 1.3-3.7 1.3-1.3 3.7-1.3-3.7L3.5 8.5l3.7-1.3ZM14.5 11.5l.7 1.8 1.8.7-1.8.7-.7 1.8-.7-1.8-1.8-.7 1.8-.7Z"/>',
  bolt: '<path d="M11 2.8 4.8 11.2H10l-1 6 6.2-8.4H10Z"/>',
  close: '<path d="m5.5 5.5 9 9M14.5 5.5l-9 9"/>',
  pause: '<path d="M7.5 5v10M12.5 5v10"/>',
  play: '<path d="M7 4.8v10.4L15 10Z"/>',
  textSize: '<path d="M3 15.5 6.6 6h.8l3.6 9.5M4.3 12.4h5.4M12 15.5l2.6-6.6h.6l2.6 6.6M12.9 13.3h4"/>',
  palette: '<path d="M10 3a7 7 0 0 0 0 14c1.2 0 1.6-.9 1.2-1.8-.5-1-.1-2.2 1.2-2.2H14a3 3 0 0 0 3-3A7 7 0 0 0 10 3Z"/><circle cx="6.6" cy="9.4" r=".9" fill="currentColor"/><circle cx="9" cy="6.4" r=".9" fill="currentColor"/><circle cx="12.6" cy="6.8" r=".9" fill="currentColor"/>',
  chip: '<rect x="4" y="6" width="12" height="8" rx="1.4"/><path d="M6.5 6V4M9 6V4M11.5 6V4M14 6V4M6.5 16v-2M9 16v-2M11.5 16v-2M14 16v-2"/>',
  link: '<path d="M8.6 11.4a3 3 0 0 0 4.2 0l2.6-2.6a3 3 0 0 0-4.2-4.2l-.9.9"/><path d="M11.4 8.6a3 3 0 0 0-4.2 0l-2.6 2.6a3 3 0 0 0 4.2 4.2l.9-.9"/>',
  hand: '<path d="M7 10V4.8a1.2 1.2 0 0 1 2.4 0V9M9.4 9V3.8a1.2 1.2 0 0 1 2.4 0V9M11.8 9V4.8a1.2 1.2 0 0 1 2.4 0V11c0 3.3-2 5.5-5 5.5-2 0-3.2-1-4.3-2.8L3.6 11.4a1.2 1.2 0 0 1 2-1.3L7 12"/>',
  speaker: '<path d="M4 8h2.6L10 5v10l-3.4-3H4Z"/><path d="M13 7.6a3.4 3.4 0 0 1 0 4.8M15.2 5.6a6.2 6.2 0 0 1 0 8.8"/>',
  branch: '<circle cx="6" cy="4.8" r="1.6"/><circle cx="6" cy="15.2" r="1.6"/><circle cx="14" cy="7.4" r="1.6"/><path d="M6 6.4v7.2M14 9c0 3-3.2 3.2-6.8 5"/>',
  export: '<path d="M10 12.5V3.5M6.5 7 10 3.5 13.5 7M4.5 10.5v4.3a1.7 1.7 0 0 0 1.7 1.7h7.6a1.7 1.7 0 0 0 1.7-1.7v-4.3"/>',
  terminal: '<rect x="2.8" y="4" width="14.4" height="12" rx="2"/><path d="m6 8 2.5 2L6 12M10.5 12.5H14"/>',
  server: '<rect x="3.5" y="3.5" width="13" height="5.5" rx="1.4"/><rect x="3.5" y="11" width="13" height="5.5" rx="1.4"/><path d="M6.5 6.2h.1M6.5 13.8h.1M10 6.2h3.5M10 13.8h3.5"/>',
  chart: '<path d="M3.5 3.5v13h13"/><path d="m6.5 12.5 3-3.5 2.5 2 4-5"/>',
  doc: '<path d="M5.5 3h6l3.5 3.5v9a1.5 1.5 0 0 1-1.5 1.5h-8A1.5 1.5 0 0 1 4 15.5v-11A1.5 1.5 0 0 1 5.5 3Z"/><path d="M11.5 3v3.5H15M7 10h6M7 13h4.5"/>',
  paperclip: '<path d="m15.5 9.5-5.6 5.6a3.4 3.4 0 0 1-4.8-4.8l6-6a2.2 2.2 0 0 1 3.1 3.1l-6 6a1 1 0 0 1-1.5-1.5l5.5-5.5"/>',
  tool: '<path d="M12.6 3.4a3.6 3.6 0 0 0-4 4.9l-5 5a1.5 1.5 0 0 0 2.1 2.1l5-5a3.6 3.6 0 0 0 4.9-4l-2.2 2.2-2.1-.6-.6-2.1Z"/>',
  warning: '<path d="M10 3.5 17 16H3Z"/><path d="M10 8.5v3.5M10 14.2h.01"/>',
  folderOpen: '<path d="M3 6a1.5 1.5 0 0 1 1.5-1.5h3.3l1.7 1.8h5A1.5 1.5 0 0 1 16 7.8V9"/><path d="M3 15.5 5 9.5h12.5l-2 6Z"/>',
};

export function icon(name, size = 16) {
  const body = P[name] || "";
  return `<svg class="ico" width="${size}" height="${size}" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${body}</svg>`;
}

// The MLXL3 monogram, traced from the Mac app's AppIcon.png (same 1024 grid and margins).
export function monogram(size = 20) {
  return `<svg class="monogram" width="${size}" height="${size}" viewBox="0 0 1024 1024" aria-hidden="true"><path fill="currentColor" d="M95 287h70l197 198 176-198h114L368 623 203 460v275H95ZM635 370l65-83h100c63 0 115 48 115 108 0 39-20 76-42 97 32 18 55 58 55 100 0 78-63 143-133 143H457v-78l65-79v67h273c20 0 35-20 35-53 0-32-18-49-35-49H577l71-86h145c19 0 35-17 35-45 0-27-16-42-35-42Z"/></svg>`;
}

export const ICONS = Object.keys(P);
