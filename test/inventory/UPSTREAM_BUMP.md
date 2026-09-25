# Upstream-bump procedure — the SWMM state-snapshot inventory step

`external/swmm` is a **tracked tree**, not a submodule, so a bump to a newer EPA
SWMM release lands as an ordinary commit on this branch rather than as a pointer
move. Nothing about that bump tells you whether it changed a struct the coupled
resume snapshot serializes — and a change that goes unnoticed does not fail
loudly. It restores a field into the wrong slot, or drops one, and the run
completes with a report that looks right.

This file states the step that closes that gap. It is one of the **two declared
firing points** of the inventory-regeneration check; the other is the slow-tier
CTest node `SLOW-SNAPSHOT-INVENTORY-REGENERATION`, registered in
`test/inventory/CMakeLists.txt` and excluded from the fast suite by the
`slow` label.

## The step

Run this **after** updating `external/swmm` and **before** committing the bump:

```bash
python3 test/inventory/regenerate_snapshot_inventory.py \
    --solver-dir external/swmm/src/solver --check
```

- **Exit 0** — the new EPA source produces the identical inventory. The bump
  does not touch the snapshot's surface. Nothing further is owed.
- **Exit 1** — the inventory changed. The script prints a unified diff of the
  committed inventory against the regenerated one. **Do not commit the bump
  until every line of that diff is accounted for.**

## Resolving a non-empty diff

The diff names the object and the field. Work through it in this order, because
the cases are distinguishable and the remedies differ:

1. **A field was ADDED to a serialized struct.** This is the case the `sizeof`
   guard is blind to when the addition lands in existing padding — measured on
   this platform, `TNodeStats`, `TOutfallStats`, `TLinkStats` and
   `TTimeStepStats` each carry a 4-byte hole an added `int` occupies without
   changing `sizeof`. **This regeneration check is the only thing that catches
   it.** Add the field to the serializer and the deserializer, in the same
   position in both.

2. **A field was REMOVED or RENAMED.** The field-by-field serializer names it,
   so the build already fails. Regenerating the inventory records the change;
   fixing the serializer resolves the build.

3. **A field's `read_by_report_path` flipped to `yes`.** EPA started reporting a
   quantity it previously only accumulated. The field was already serialized
   (every field of a serialized struct is written regardless of the column), so
   no code change is owed — commit the regenerated inventory.

4. **A field's column flipped to `no`, or a new entry appeared in the closure
   boundary.** EPA moved a report path into a translation unit the walk does not
   enter. Check `REPORT_BEARING_UNITS` in the script: if the report writer moved
   to a new file, that file belongs in the walked set.

Then re-run without `--check` to write the new inventory, and commit it in the
**same commit as the bump** so the two never disagree on disk.

## The Criterion-P half of the same step

The command above regenerates **both** halves of the inventory, so the single
`--check` invocation already covers routing state as well as report state. What
changes is how you read a non-empty diff, because Criterion P's failure modes
are different ones:

1. **A row reads `UNTRIAGED`.** The routing closure reads a quantity that
   appears in neither `ADMITTED_ROUTING_STATE` nor `EXCLUDED_ROUTING_STATE` nor
   a whole-object rule in `EXCLUDED_ROUTING_OBJECTS`. Decide which it is and say
   why. **Under Criterion P the two directions are not symmetric**: under-capture
   is the wrong numbers, over-capture is a maintenance cost — *except* for `.inp`
   configuration, where restoring it lets a stale snapshot silently override the
   model the operator is running. So admit state freely and exclude
   configuration by triage, never the reverse.

2. **A row moved between `LIVE` and `KILLED`.** EPA changed where the step
   prologue writes a field relative to where it reads it. The kill pass is
   ORDER-SENSITIVE and a flat reading of it is unsound, so resolve this by
   reading the prologue in execution order rather than by grepping for the
   write.

3. **`PROLOGUE-ORDER-VIOLATION`.** The six prologue functions no longer execute
   in the order `ROUTING_PROLOGUE` declares. Mode 2 is not well-posed until this
   is resolved: update the declared order to the source's, and re-examine every
   `KILLED` row, because the order is what decided them.

4. **`PREFILTER-VIOLATION`.** A translation unit declared in
   `PREFILTER_EXCLUDED_UNITS` now declares mutable state, so its own
   declarations belong back in the candidate pool. Remove it from the list and
   triage whatever it contributes. **This is the mode a translation-unit bound
   could not fail on** — it is the reason the bound sits on the field axis.

5. **`PREFILTER-UNDECLARED`.** A unit the walk reaches has stopped declaring
   mutable state. Add it to `PREFILTER_EXCLUDED_UNITS` so the exclusion stays a
   committed declaration a reader can check, rather than a recomputation that
   verifies itself.

6. **A `D-R6` handle reads `UNTRIAGED`.** Upstream added a `TFile`. The
   enumeration is keyed on the TYPE and not on a declaration site, so it found
   the new handle automatically; supply its row. An EMPTY admitted set is a
   passing result.

**Two defects this check cannot catch by itself, and the node that does.** The
regeneration compares two outputs of the SAME parser, so a parser that stops
seeing a construct makes both sides agree and the diff empty. Two such defects
have already been repaired here — `struct_fields` dropping every declarator but
the last of a multi-declarator line, and `split_functions` refusing a definition
whose header carries a trailing `//` comment — and the second silently removed
thirty rows while the run exited 0. `SLOW-SNAPSHOT-INVENTORY-CLOSURE-MODES`
reintroduces each defect on a throwaway copy and asserts the row disappears. If
you touch either helper, run that node.

## Why the script lives here and is never hand-ported

It sits in TRITON's own test tree, on the precedent of
`test/reference/compare_runs_simple.py`. The vendored tree carries EPA's source
and nothing of ours; the machinery that checks our assumptions about that source
is ours and stays on our side of the line. That placement is what keeps the cost
of the vendored surface bounded to the serializer itself — the machinery is the
same machinery either way, and only its location is in question.
