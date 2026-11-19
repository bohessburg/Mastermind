
import sys, json, os, time
st_path = sys.argv[1]
try: st = json.load(open(st_path))
except Exception: st = {"phase": {}, "lines": 0, "status": ""}
lines = []
for raw in sys.stdin:
    raw = raw.rstrip("\n")
    if raw.startswith("P "):
        _, arm, js = raw.split(" ", 2)
        try: p = json.loads(js)
        except Exception: continue
        key = "%s:%s" % (p.get("generation"), p.get("phase"))
        prev = st["phase"].get(arm)
        if prev != key:
            if prev:
                pg, pph = prev.split(":")
                dur = (p.get("phase_started_unix", time.time()) - st.get("since", {}).get(arm, p.get("phase_started_unix", time.time()))) / 60
                print(f"c35c {arm} gen {pg}: {pph} done ({dur:.0f} min) -> gen " + str(p.get("generation")) + " " + str(p.get("phase")), flush=True)
            st["phase"][arm] = key
            st.setdefault("since", {})[arm] = p.get("phase_started_unix", time.time())
    elif raw.startswith("L "):
        lines.append(raw[2:])
    elif raw.startswith("S "):
        s = raw.split()[2] if len(raw.split()) > 2 else "?"
        if s != st.get("status") and s not in ("RUNNING",):
            print(f"c35c supervisor status {s}", flush=True)
        st["status"] = s
old = st["lines"]
for l in lines[st["lines"]:]:
    if l.startswith("{"):
        d = json.loads(l)
        if d.get("wins") is None:
            continue
        print("c35c GATE gen %s: %s %.3f (%s-%s-%s)" % (d.get("generation"), d.get("gate_result"), d.get("gate_score") or 0, d.get("wins"), d.get("losses"), d.get("ties")), flush=True)
    else:
        print("c35c " + l[:200], flush=True)
st["lines"] = len(lines)
json.dump(st, open(st_path, "w"))
sys.exit(3 if any(l.startswith("{") for l in lines[old:]) else 0)
