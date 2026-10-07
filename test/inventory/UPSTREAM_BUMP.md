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

## Why the script lives here and is never hand-ported

It sits in TRITON's own test tree, on the precedent of
`test/reference/compare_runs_simple.py`. The vendored tree carries EPA's source
and nothing of ours; the machinery that checks our assumptions about that source
is ours and stays on our side of the line. That placement is what keeps the cost
of the vendored surface bounded to the serializer itself — the machinery is the
same machinery either way, and only its location is in question.
