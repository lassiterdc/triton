"""End-to-end consistency check of the T1 SWMM-step capture files (chunk 2a.2) on the coupled
quality fixture. Needs a coupled triton.exe and mpirun, so it runs on Rivanna only.

usage: test_t1_capture_consistency.py SRC EXE NP MODE   (SRC: the source tree, EXE: triton.exe,
NP: rank count, MODE: full | quick)

Per fixture variant (base, nonconv = MAX_TRIALS 1 with a tiny head tolerance so SWMM leaves nodes
non-converged, t3 = swmm_coupling_interval 4) it runs a clean captures-on run and checks the files
against each other, not against the writer's own arithmetic:
  X1 stamps of T1a, T1c, T1d are one sequence; X2 stamp k - stamp k-1 equals dt k, first stamp = dt 1,
  last stamp = the run end; X3 the domain p3 sums of T1a equal the serial sums of T1c's per-node values;
  X4 T1b's per-node sums, per interval, equal the per-interval sums of T1c, its non-converged counts
  equal the per-interval occurrences of each ordinal in T1a's blocks; X5 sidecars equal the .inp's names;
  X6 sizes; X7 T1a record count equals the exchange side-file's record count (T3 arm: only SWMM steps);
  X8 ordinals in each block ascend and lie in [0, n); X9 T1d holds cfs (link C1 of the fixture carries
  its 0.010 cms baseline, 0.35315 cfs); X10 SWMM_SOLVE and SWMM_LOG of performance.txt are nonzero on
  rank 0 in every variant, the T3 one included, and their sum does not exceed SWMM_STEP.
full mode adds: snapshot-route and replay-route resumes from checkpoint 2 and a partial-record resume
(each T1 binary cut mid-record after checkpoint 2) must equal the clean files byte for byte; captures-off
run: no T1 file, DIAG 0, on-vs-off compare of every common file.
"""
import os, sys, shutil, subprocess, struct, filecmp, re, math

SRC, EXE, NP, MODE = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
sys.path.insert(0, SRC + "/test/unit")
import test_quality_resume as q

T1 = ["swmm_step_diag.bin", "swmm_node_path3.bin", "swmm_node_path3_step.bin", "swmm_link_flow_diag.bin",
      "swmm_node_diag.nodes", "swmm_link_flow_diag.links"]
fails = []
RES = {}


def fail(msg):
    print("  FAIL:", msg)
    fails.append(msg)


def run(cfg, cwd, proj, tag):
    launch = ["mpirun", "-n", NP, os.path.join(proj, "bin", "triton.exe"), cfg]
    r = subprocess.run(launch, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    ok = r.returncode == 0 and "Simulation ends" in r.stdout
    print("---- %s exit %d simulation-ends %s" % (tag, r.returncode, "Simulation ends" in r.stdout))
    if not ok:
        print(r.stdout[-2500:])
        fail(tag + " did not complete")
    return r.stdout


def setup(name, extra, nonconv):
    d = os.path.abspath(name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    cfg = q.write_fixture(d)
    if nonconv:
        inp = os.path.join(d, "model.inp")
        t = open(inp).read()
        t = t.replace("ROUTING_STEP         00:00:01\n", "ROUTING_STEP         00:00:01\nMAX_TRIALS           1\nHEAD_TOLERANCE       0.0000001\n")
        assert "MAX_TRIALS" in t
        open(inp, "w").write(t)
    with open(cfg, "a") as f:
        f.write(extra)
    proj = os.path.join(d, "proj")
    os.makedirs(os.path.join(proj, "bin"))
    os.symlink(EXE, os.path.join(proj, "bin", "triton.exe"))
    return d, cfg, proj


def rd(path):
    return open(path, "rb").read()


def parse(u):
    out = {}
    nodes = open(os.path.join(u, "swmm_node_diag.nodes")).read().splitlines()
    links = open(os.path.join(u, "swmm_link_flow_diag.links")).read().splitlines()
    n, L = len(nodes), len(links)
    out["nodes"], out["links"] = nodes, links
    b = rd(os.path.join(u, "swmm_step_diag.bin"))
    hdr = struct.unpack_from("<4i", b, 0)
    out["h_a"] = hdr
    recs, off = [], 16
    while off < len(b):
        t, dt, trials, nnc, pp, pn, lv = struct.unpack_from("<ddiiddd", b, off)
        ords = list(struct.unpack_from("<%di" % nnc, b, off + 48))
        recs.append((t, dt, trials, nnc, pp, pn, lv, ords))
        off += 48 + 4 * nnc
    out["walk_end_ok"] = off == len(b)
    out["a"] = recs
    c = rd(os.path.join(u, "swmm_node_path3_step.bin"))
    out["h_c"] = struct.unpack_from("<4i", c, 0)
    rs = 16 + 8 * n
    out["c"] = [(struct.unpack_from("<d", c, 16 + i * rs)[0], list(struct.unpack_from("<%dd" % n, c, 16 + i * rs + 16)))
                for i in range((len(c) - 16) // rs)]
    out["c_size_ok"] = (len(c) - 16) % rs == 0
    d = rd(os.path.join(u, "swmm_link_flow_diag.bin"))
    out["h_d"] = struct.unpack_from("<4i", d, 0)
    rs = 8 + 8 * L
    out["d"] = [(struct.unpack_from("<d", d, 16 + i * rs)[0], list(struct.unpack_from("<%dd" % L, d, 16 + i * rs + 8)))
                for i in range((len(d) - 16) // rs)]
    out["d_size_ok"] = (len(d) - 16) % rs == 0
    bb = rd(os.path.join(u, "swmm_node_path3.bin"))
    out["h_b"] = struct.unpack_from("<4i", bb, 0)
    rs = 8 + 20 * n
    ivs = []
    for i in range((len(bb) - 16) // rs):
        o = 16 + i * rs
        t = struct.unpack_from("<d", bb, o)[0]
        pos = list(struct.unpack_from("<%dd" % n, bb, o + 8))
        neg = list(struct.unpack_from("<%dd" % n, bb, o + 8 + 8 * n))
        nc = list(struct.unpack_from("<%di" % n, bb, o + 8 + 16 * n))
        ivs.append((t, pos, neg, nc))
    out["b"] = ivs
    out["b_size_ok"] = (len(bb) - 16) % rs == 0
    return out


def perf_rank0(u):
    """performance.txt of a run as {column: value} for rank 0."""
    lines = open(os.path.join(u, "performance.txt")).read().splitlines()
    hdr = [h for h in re.split(r"[,\s]+", lines[0].lstrip("%")) if h]
    for ln in lines[1:]:
        row = [x for x in re.split(r"[,\s]+", ln) if x]
        if len(row) == len(hdr) and float(row[0]) == 0.0:
            return dict(zip(hdr, [float(x) for x in row]))
    return {}


def side_file_records(u, n_coupled_guess=None):
    p = os.path.join(u, "swmm", "model_exchange_replay.bin")
    b = rd(p)
    magic, nn, w = struct.unpack_from("<3i", b, 0)
    rs = w * (1 + nn)
    return (len(b) - 12) // rs, nn, w


def analyze(u, label, t3, end_time):
    print("== analyze %s ==" % label)
    o = parse(u)
    n, L = len(o["nodes"]), len(o["links"])
    a, c, d, b = o["a"], o["c"], o["d"], o["b"]
    print("  sidecars: %d non-outfall nodes %s, %d links %s" % (n, o["nodes"], L, o["links"]))
    print("  headers a %s c %s d %s b %s" % tuple([[hex(o[k][0])] + list(o[k][1:]) for k in ("h_a", "h_c", "h_d", "h_b")]))
    if not (o["h_a"][0] == 0x44535031 and o["h_a"][3] == 0 and o["h_b"][0] == 0x44504E33 and o["h_b"][3] == n
            and o["h_c"][0] == 0x44505333 and o["h_c"][3] == n and o["h_d"][0] == 0x444C4631 and o["h_d"][3] == L):
        fail(label + ": header magic or n_items")
    for k in ("h_a", "h_b", "h_c", "h_d"):
        if o[k][1] != 1 or o[k][2] != 8:
            fail(label + ": %s version/value_width" % k)
    if not (o["walk_end_ok"] and o["c_size_ok"] and o["d_size_ok"] and o["b_size_ok"]):
        fail(label + ": a file's length is not 16 + records x record size (X6)")
    K = len(a)
    print("  T1a %d records; T1c %d; T1d %d; T1b %d intervals" % (K, len(c), len(d), len(b)))
    if not (K == len(c) == len(d)) or K == 0:
        fail(label + ": record counts of T1a/T1c/T1d differ or are zero (X1)")
        return o
    sa = [r[0] for r in a]
    if sa != [r[0] for r in c] or sa != [r[0] for r in d]:
        fail(label + ": stamps of T1a, T1c, T1d are not one sequence (X1)")
    if sa != sorted(sa):
        fail(label + ": T1a stamps decrease")
    if abs(sa[0] - a[0][1]) > 1e-9 * max(1.0, sa[0]):
        fail(label + ": first stamp %r is not the first dt %r (X2: stamp must be the step END time)" % (sa[0], a[0][1]))
    bad = [k for k in range(1, K) if abs((sa[k] - sa[k - 1]) - a[k][1]) > 1e-9 * max(1.0, sa[k])]
    print("  X2 stamp-difference vs dt: %d of %d steps disagree; first stamp %r first dt %r last stamp %r" % (len(bad), K - 1, sa[0], a[0][1], sa[-1]))
    if bad:
        fail(label + ": stamp difference differs from dt at %d steps (X2)" % len(bad))
    if abs(sa[-1] - end_time) > 1e-6:
        fail(label + ": last stamp %r is not the run end %r (X2)" % (sa[-1], end_time))
    ex = 0
    worst = 0.0
    for k in range(K):
        pp = 0.0
        pn = 0.0
        for v in c[k][1]:
            if v > 0.0:
                pp += v
            elif v < 0.0:
                pn += v
        if pp == a[k][4] and pn == a[k][5]:
            ex += 1
        worst = max(worst, abs(pp - a[k][4]), abs(pn - a[k][5]))
    print("  X3 domain sums == serial sums of per-node values: exact %d of %d, worst abs diff %g" % (ex, K, worst))
    if worst > 1e-9:
        fail(label + ": T1a domain sums disagree with T1c per-node values (X3), worst %g" % worst)
    # X4
    tb = [r[0] for r in b]
    lo = 0.0
    exact = 0
    mism = []
    for i, (t, pos, neg, nc) in enumerate(b):
        sp = [0.0] * n
        sn = [0.0] * n
        cnt = [0] * n
        for k in range(K):
            if lo + 1e-9 < sa[k] <= t + 1e-9:
                for j, v in enumerate(c[k][1]):
                    if v > 0.0:
                        sp[j] += v
                    elif v < 0.0:
                        sn[j] += v
                for ordi in a[k][7]:
                    cnt[ordi] += 1
        if sp == pos and sn == neg and cnt == nc:
            exact += 1
        else:
            mism.append((i, t))
        lo = t
    print("  X4 T1b interval sums and counts re-derived from T1c and T1a: %d of %d intervals identical; stamps %s; mismatches %s" % (exact, len(b), tb, mism))
    if mism:
        fail(label + ": T1b intervals %s disagree with the per-step files (X4)" % mism)
    if not t3 and len(b) != 3:
        fail(label + ": expected 3 T1b intervals")
    # X8
    bad8 = [k for k in range(K) if a[k][3] != len(a[k][7]) or a[k][7] != sorted(a[k][7]) or any(x < 0 or x >= n for x in a[k][7])]
    nnc_steps = sum(1 for r in a if r[3] > 0)
    print("  X8 non-converged: steps with a non-converged node %d, ordinals listed %d, malformed blocks %d" % (nnc_steps, sum(r[3] for r in a), len(bad8)))
    if bad8:
        fail(label + ": blocks not ascending/in range or count field != block length (X8)")
    # X7
    nrec, nn, w = side_file_records(u)
    print("  X7 exchange side-file records %d (num_nodes %d width %d) vs T1a records %d" % (nrec, nn, w, K))
    if nrec != K:
        fail(label + ": T1a has %d records but SWMM stepped %d times (X7)" % (K, nrec))
    # units: link flow of the first listed link at the last step vs the model's baseline
    print("  link flows at the last step (file values): %s" % dict(zip(o["links"], [round(x, 6) for x in d[-1][1]])))
    # X9: T1d is in cfs (erratum E-11)
    c1 = d[-1][1][o["links"].index("C1")]
    want = 0.010 / 0.0283168
    print("  X9 link C1 at the last step %r cfs, expected %r" % (c1, want))
    if abs(c1 - want) > 1e-3:
        fail(label + ": T1d link C1 is %r, not %r cfs (X9)" % (c1, want))
    # X10: the solve and log timers read nonzero on rank 0, also in the T3 branch
    pr = perf_rank0(u)
    print("  X10 rank 0 SWMM_STEP %r SWMM_SOLVE %r SWMM_LOG %r" % (pr.get("SWMM_STEP"), pr.get("SWMM_SOLVE"), pr.get("SWMM_LOG")))
    if not (pr.get("SWMM_SOLVE", 0.0) > 0.0 and pr.get("SWMM_LOG", 0.0) > 0.0):
        fail(label + ": SWMM_SOLVE or SWMM_LOG is zero on rank 0 (X10)")
    if pr.get("SWMM_SOLVE", 0.0) + pr.get("SWMM_LOG", 0.0) > pr.get("SWMM_STEP", 0.0) * 1.001:
        fail(label + ": SWMM_SOLVE + SWMM_LOG exceeds SWMM_STEP on rank 0, so a bracket is unbalanced (X10)")
    return o


def t1_equal(ua, ub, tag):
    for n_ in T1:
        a, bb = os.path.join(ua, n_), os.path.join(ub, n_)
        same = os.path.exists(bb) and rd(a) == rd(bb)
        print("  %s: %s byte-identical to clean: %s (%d bytes)" % (tag, n_, same, os.path.getsize(bb) if os.path.exists(bb) else -1))
        if not same:
            fail("%s %s differs from the clean run" % (tag, n_))


def resume_arms(d, proj, u, o, label, base_extra_for_resume=""):
    for arm, extra in (("snapshot", ""), ("replay", "swmm_snapshot_disable=1\n")):
        r_out = os.path.join(proj, "out_" + arm)
        shutil.rmtree(r_out, ignore_errors=True)
        shutil.copytree(u, r_out, symlinks=True)
        rcfg = os.path.join(r_out, "cfg", "config_2.cfg")
        text = open(rcfg).read()
        text = re.sub(r'(?m)^output_folder=.*$', 'output_folder="out_%s"' % arm, text)
        if not text.endswith("\n"):
            text += "\n"
        open(rcfg, "w").write(text + extra)
        run(rcfg, d, proj, "%s resume %s" % (label, arm))
        t1_equal(u, r_out, "%s %s" % (label, arm))
    # partial record: cut every T1 binary mid-record after checkpoint 2 (t = 1200 s), resume snapshot route
    r_out = os.path.join(proj, "out_partial")
    shutil.rmtree(r_out, ignore_errors=True)
    shutil.copytree(u, r_out, symlinks=True)
    n = len(o["nodes"]); L = len(o["links"])
    a = o["a"]
    cuts = {}
    off = 16
    for r in a:
        if r[0] > 1200.0 + 1e-6:
            cuts["swmm_step_diag.bin"] = off + 25
            break
        off += 48 + 4 * r[3]
    ks = [k for k, r in enumerate(a) if r[0] > 1200.0 + 1e-6]
    if ks:
        cuts["swmm_node_path3_step.bin"] = 16 + ks[0] * (16 + 8 * n) + 17
        cuts["swmm_link_flow_diag.bin"] = 16 + ks[0] * (8 + 8 * L) + 11
    cuts["swmm_node_path3.bin"] = 16 + 2 * (8 + 20 * n) + 13
    for name, cut in cuts.items():
        p = os.path.join(r_out, name)
        full = os.path.getsize(p)
        with open(p, "r+b") as f:
            f.truncate(cut)
        print("  partial arm: %s cut %d -> %d bytes" % (name, full, cut))
    rcfg = os.path.join(r_out, "cfg", "config_2.cfg")
    text = open(rcfg).read()
    text = re.sub(r'(?m)^output_folder=.*$', 'output_folder="out_partial"', text)
    open(rcfg, "w").write(text if text.endswith("\n") else text + "\n")
    run(rcfg, d, proj, "%s resume from partial records" % label)
    t1_equal(u, r_out, "%s partial" % label)


def variant(label, extra, nonconv, t3, full):
    on_extra = "swmm_diag_steps=1\nswmm_diag_node_path3_per_step=1\nswmm_diag_links=ALL\n" + extra
    d, cfg, proj = setup("on_" + label, on_extra, nonconv)
    run(cfg, d, proj, "%s clean run, captures on (np %s)" % (label, NP))
    u = os.path.join(proj, "out")
    if any(not os.path.exists(os.path.join(u, n_)) for n_ in T1):
        fail(label + ": a T1 file is missing")
        return
    o = analyze(u, label, t3, 1800.0)
    RES[label] = o
    if full:
        resume_arms(d, proj, u, o, label)
        d2, cfg2, proj2 = setup("off_" + label, extra, nonconv)
        run(cfg2, d2, proj2, "%s clean run, captures off" % label)
        u2 = os.path.join(proj2, "out")
        present = [n_ for n_ in T1 if os.path.exists(os.path.join(u2, n_))]
        print("  captures off: T1 files present: %s" % present)
        if present:
            fail(label + ": a T1 file exists with every toggle off")
        po = open(os.path.join(u2, "performance.txt")).read().splitlines()
        pn = open(os.path.join(u, "performance.txt")).read().splitlines()
        print("  captures-off performance.txt: %s\n  %s" % (po[0], po[1]))
        print("  captures-on  performance.txt: %s\n  %s" % (pn[0], pn[1]))
        try:
            hdr = [h for h in re.split(r"[,\s]+", po[0].lstrip("%")) if h]
            row = [x for x in re.split(r"[,\s]+", po[1]) if x]
            if "DIAG" in hdr and len(hdr) == len(row):
                dv = float(row[hdr.index("DIAG")])
                print("  DIAG captures-off = %r, captures-on = %r" % (dv, float([x for x in re.split(r"[,\s]+", pn[1]) if x][hdr.index("DIAG")])))
                if dv != 0.0:
                    fail(label + ": DIAG is not exactly 0 with captures off")
            else:
                print("  performance.txt header/row length differ (%d vs %d); DIAG not parsed" % (len(hdr), len(row)))
        except Exception as ex:
            print("  performance parse error", ex)
        def walk(root):
            r = {}
            for dp, dn, fn in os.walk(root):
                for f in fn:
                    r[os.path.relpath(os.path.join(dp, f), root)] = os.path.join(dp, f)
            return r
        wa, wb = walk(u), walk(u2)
        common = sorted(k for k in wa if k in wb and k not in set(T1))
        diff = [k for k in common if not filecmp.cmp(wa[k], wb[k], shallow=False)]
        print("  on vs off: %d common files, %d differ: %s" % (len(common), len(diff), diff))


def linkfile_arm():
    """swmm_diag_links as a path: the engine-spelled names in file order, case-insensitive match, columns equal the ALL run's."""
    print("== linkfile arm ==")
    base = RES.get("base")
    if base is None:
        fail("linkfile arm: no base ALL run")
        return
    d, cfg, proj = setup("on_linkfile", "", False)
    lf = os.path.join(d, "links.txt")
    open(lf, "w").write("st2\nc1\n")
    with open(cfg, "a") as f:
        f.write("swmm_diag_links=%s\n" % lf)
    run(cfg, d, proj, "linkfile clean run (np %s)" % NP)
    u = os.path.join(proj, "out")
    names = open(os.path.join(u, "swmm_link_flow_diag.links")).read().splitlines()
    print("  sidecar:", names, "(ALL order %s)" % base["links"])
    if names != ["ST2", "C1"]:
        fail("linkfile arm: sidecar %s is not the engine spelling of the file's names in file order" % names)
    b = rd(os.path.join(u, "swmm_link_flow_diag.bin"))
    h = struct.unpack_from("<4i", b, 0)
    rs = 8 + 16
    recs = [(struct.unpack_from("<d", b, 16 + i * rs)[0], list(struct.unpack_from("<2d", b, 16 + i * rs + 8))) for i in range((len(b) - 16) // rs)]
    print("  header %s, %d records" % ([hex(h[0])] + list(h[1:]), len(recs)))
    if h[3] != 2 or (len(b) - 16) % rs:
        fail("linkfile arm: n_items or length")
    ia, ib = base["links"].index("ST2"), base["links"].index("C1")
    same = len(recs) == len(base["d"]) and all(recs[k][0] == base["d"][k][0] and recs[k][1] == [base["d"][k][1][ia], base["d"][k][1][ib]] for k in range(len(recs)))
    print("  columns equal the ALL run's ST2 and C1 columns at every step: %s" % same)
    if not same:
        fail("linkfile arm: columns differ from the ALL run")
    for n_ in ("swmm_step_diag.bin", "swmm_node_path3.bin", "swmm_node_path3_step.bin", "swmm_node_diag.nodes"):
        if os.path.exists(os.path.join(u, n_)):
            fail("linkfile arm: %s exists with only swmm_diag_links set" % n_)


full = MODE == "full"
variant("base", "", False, False, full)
variant("nonconv", "", True, False, full)
variant("t3", "swmm_coupling_interval=4\n", False, True, full)
if full:
    linkfile_arm()
print("RESULT", "PASS" if not fails else "FAIL %s" % fails)
sys.exit(1 if fails else 0)
