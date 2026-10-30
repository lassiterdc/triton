#!/usr/bin/env python3
"""SOURCE-SWMM-RETURN-CODES -- no call to swmm_open, swmm_start or swmm_step in
src/triton.h or src/swmm_triton.h discards its return code.

Why it is worth keeping: SWMM reports a rejected input (a missing time-series
file, a bad [OPTIONS] keyword) ONLY through these return codes. When the code is
discarded, every later swmm_step returns at once without routing and without
writing depths, so the coupled run completes, exits 0 and prints "Simulation
ends" against a sewer that never moves. Measured: a missing .dat (SWMM ERROR 361)
and a bad option (ERROR 200) both gave RC 0 and a final surface volume 20 % below
the healthy coupled run.

What it asserts: every such call is CAPTURED -- the character before the call
name, ignoring whitespace, comments and string literals, is one of `=`, `(` or
`,`. A call that starts a statement (preceded by `;`, `{`, `}`, `)` or the
keywords `else`/`do`) discards its return and fails the test. The test also runs
its own predicate on two synthetic snippets, one discarding and one capturing,
so a predicate that can no longer go red is itself a failure.

Pure Python over src/: no build, no MPI, no SWMM. Usage:
    test_swmm_return_codes_source.py {TRITON source root}
"""

import re
import sys
from pathlib import Path

CALL_RE = re.compile(r"\bswmm_(open|start|step)\s*\(")
FILES = ("src/triton.h", "src/swmm_triton.h")
CAPTURING = set("=(,")


def blank_comments_and_strings(text):
    """Replace comments and string/char literals with spaces, keeping newlines,
    so offsets and line numbers survive."""
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif c == "/" and nxt == "*":
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(re.sub(r"[^\n]", " ", text[i:j]))
            i = j
        elif c in "\"'":
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if text[j] == "\\" else 1
            j = min(j + 1, n)
            out.append(re.sub(r"[^\n]", " ", text[i:j]))
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def discarded_calls(text):
    """Return [(line, name)] for every swmm_open/start/step call whose return is
    not captured, and the total number of calls seen."""
    code = blank_comments_and_strings(text)
    bad, seen = [], 0
    for m in CALL_RE.finditer(code):
        seen += 1
        k = m.start() - 1
        while k >= 0 and code[k].isspace():
            k -= 1
        prev = code[k] if k >= 0 else ";"
        word = re.search(r"(\w+)$", code[: k + 1])
        if prev in CAPTURING and not (word and word.group(1) in ("else", "do")):
            continue
        bad.append((code.count("\n", 0, m.start()) + 1, "swmm_" + m.group(1)))
    return bad, seen


def self_test():
    discarding = "void f() {\n  swmm_open(a, b, c);\n  if (x) swmm_step(&e, q, d, t);\n}\n"
    capturing = ("void f() {\n  int rc =\n  swmm_open(a, b, c);\n"
                 "  g(swmm_start(1), \"swmm_step(\");\n  // swmm_step(x);\n}\n")
    b1, s1 = discarded_calls(discarding)
    b2, s2 = discarded_calls(capturing)
    ok = (len(b1) == 2 and s1 == 2 and b2 == [] and s2 == 2)
    print("self-test: discarding snippet flags %d of %d calls (want 2 of 2); "
          "capturing snippet flags %d of %d (want 0 of 2) -> %s"
          % (len(b1), s1, len(b2), s2, "ok" if ok else "FAILED"))
    return ok


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    root = Path(argv[1])
    ok = self_test()
    total = 0
    for rel in FILES:
        path = root / rel
        text = path.read_text(encoding="utf-8", errors="replace")
        bad, seen = discarded_calls(text)
        total += seen
        print("%s: %d swmm_open/swmm_start/swmm_step call(s), %d discarding"
              % (rel, seen, len(bad)))
        for line, name in bad:
            print("FAIL %s:%d: the return code of %s is discarded" % (rel, line, name))
            ok = False
    # Population floor: today's tree has one swmm_open, one swmm_start and two
    # swmm_step calls. Finding fewer means the scan no longer reaches them.
    if total < 4:
        print("FAIL: found %d calls across %s, expected at least 4 -- the scan "
              "no longer reaches the call sites" % (total, ", ".join(FILES)))
        ok = False
    print("PASS SOURCE-SWMM-RETURN-CODES" if ok else "FAIL SOURCE-SWMM-RETURN-CODES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
