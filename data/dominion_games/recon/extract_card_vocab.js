const fs = require('fs');
const path = '/private/tmp/claude-501/-Users-paisho-Projects-Mastermind/486eaf4e-1fb8-40c5-9de3-6947283860d8/scratchpad/body.pretty.js';
const src = fs.readFileSync(path, 'utf8');

const marker = 'CardNames = {';
const startIdx = src.indexOf(marker);
if (startIdx === -1) { console.error('marker not found'); process.exit(1); }
let i = startIdx + marker.length - 1; // pointing at '{'
let depth = 0;
let objStart = i;
for (; i < src.length; i++) {
  const c = src[i];
  if (c === '{') depth++;
  else if (c === '}') { depth--; if (depth === 0) break; }
}
const objEnd = i;
const objText = src.slice(objStart, objEnd + 1);
fs.writeFileSync('/private/tmp/claude-501/-Users-paisho-Projects-Mastermind/486eaf4e-1fb8-40c5-9de3-6947283860d8/scratchpad/cardnames_obj.txt', objText);
console.log('extracted length', objText.length);

// Now parse top-level entries: KEY: new CardName(...)
// We'll walk and split on top-level commas (depth==1 relative to objText's outer brace)
let entries = [];
let d = 0;
let cur = '';
for (let idx = 1; idx < objText.length - 1; idx++) {
  const c = objText[idx];
  if (c === '(' || c === '[' || c === '{') d++;
  if (c === ')' || c === ']' || c === '}') d--;
  if (c === ',' && d === 0) {
    entries.push(cur.trim());
    cur = '';
  } else {
    cur += c;
  }
}
if (cur.trim()) entries.push(cur.trim());

console.log('entry count', entries.length);

const results = [];
entries.forEach((e, ord) => {
  const m = e.match(/^([A-Z0-9_]+):\s*new CardName\((.*)\)$/s);
  if (!m) {
    results.push({ord, key: null, raw: e.slice(0,200)});
    return;
  }
  const key = m[1];
  const argsStr = m[2];
  // parse args at top level split by comma
  let args = [];
  let dd = 0, ccur = '';
  for (let k = 0; k < argsStr.length; k++) {
    const c = argsStr[k];
    if (c === '(' || c === '[' || c === '{') dd++;
    if (c === ')' || c === ']' || c === '}') dd--;
    if (c === ',' && dd === 0) { args.push(ccur.trim()); ccur=''; }
    else ccur += c;
  }
  if (ccur.trim()) args.push(ccur.trim());
  const nameMatch = args[0] && args[0].match(/^"(.*)"$/);
  results.push({
    ord,
    key,
    name: nameMatch ? nameMatch[1] : args[0],
    expansion: args[1],
    cost: args[2],
    types: args[3],
    isKingdomPile: args[4],
    sortingGroup: args[5],
    isFake: args[6] || false,
  });
});

fs.writeFileSync('/private/tmp/claude-501/-Users-paisho-Projects-Mastermind/486eaf4e-1fb8-40c5-9de3-6947283860d8/scratchpad/card_vocab_raw.json', JSON.stringify(results, null, 2));
console.log('wrote card_vocab_raw.json with', results.length, 'entries');
console.log('first 5:', results.slice(0,5));
console.log('any parse failures:', results.filter(r => r.key === null).length);
