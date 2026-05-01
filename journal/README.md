# M11 Journal

Manual filesystem journal for milestone 11 (scene reconstruction). Layout is the OLAI 4-resource design (`journal` / `thread` / `entry` / `note`) — written by hand, ahead of OLAI's server-side implementation, so the eventual conversion is a mechanical read-and-store rather than a translation exercise.

## Layout

```
journal/
└── journals/
    └── m11-scene-reconstruction/
        ├── journal.md
        └── threads/
            ├── inbox/                  ← capture-fast, working surface
            │   ├── thread.md
            │   └── notes/
            ├── matcha-quality/         ← inquiry into MAtCha mesh quality
            │   ├── thread.md
            │   ├── entries/
            │   └── notes/
            └── post-processing/        ← Phase B tooling (orient / cull / cameras / color)
                ├── thread.md
                ├── entries/
                └── notes/
```

## Conventions

- **Hierarchy is strict containment.** Every entry and note lives inside exactly one thread, which lives inside exactly one journal.
- **Cross-thread surfacing** is done by the *referencing* thread (in its `references:` list), not by the referenced entry. Single-writer rule.
- **Notes are raw, entries are synthesized.** An entry's `consolidates_notes:` list points to the notes it drew from. Flip the cited notes' `consolidated:` to true.
- **Folder per entry/note.** Even when there's only the markdown file. Keeps attachments simple later.
- **Slugs are kebab-case ASCII lowercase.**
- **Inbox is a working surface, not a permanent home.** Sweep it during entry-writing sessions.

## What goes in the journal vs the rest of the milestone tree

| Source | Role |
|--------|------|
| `PLAN.md` | Forward-looking plan |
| `experiments/<id>/README.md` | Per-run reference |
| `experiments/DECISION-MATRIX.md` | Current pipeline beliefs (snapshot) |
| `experiments/<id>/CAPTURE-LESSONS.md` | Capture-side findings |
| **journal/ (this tree)** | **Chronology — what we tried, what we learned, what we changed our mind about, what's still open** |

Journal entries reference the canonical artifacts above; they should not duplicate their content. The journal's job is the *narrative*.

## When the OLAI server lands

Conversion is mechanical:

- `journal.md` / `thread.md` / `entry.md` / `note.md` each become a resource at the matching ORN path.
- Frontmatter fields map 1:1 to model fields.
- Markdown body becomes the resource's content variant.
- `references:` and `consolidates_notes:` paths become typed handles.

Don't write tooling. Don't invent fields. Don't optimize the folder-per-entry away. Keep it manual until OLAI lands; that's the whole point.
