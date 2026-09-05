---
tags:
  - agents
  - inventory
  - factory
  - library
  - claude-code
created: 2026-09-05
source: Open Brain memory search, 2026-09-05 (not verified against disk; see Evidence note)
status: draft-from-memory
---

# Persistent Agents Inventory (Library + Factory)

> [!warning] Evidence note
> [Fact] This inventory was assembled from Open Brain captures dated 2026-05-08 to 2026-09-05. The Factory repo (Roman-TSA/icm-workspaces) and the Library vault (D:\Creative Core) could not be read from this session, so nothing below was checked against disk.
> [Fact] The Factory Handbook (2026-08-01) counts 23 workspaces. Only 19 are named in memory, so at least 4 are missing from this list.
> [Uncertain] Anything marked ⚪ or 🟡 may have been renamed, merged, or retired since its capture date.

## Legend (persistence tiers)

| Icon | Tier | Meaning |
|---|---|---|
| 🟢 | Scheduled or hooked | Fires without Ted starting it (Windows Task Scheduler, Claude Code scheduler, or a hook) |
| 🔵 | Standing command or subagent | Always installed, invoked by name (slash command or user-level subagent) |
| 🟡 | Pipeline workspace | Persistent folder in the Factory with numbered stages, run one stage at a time |
| ⚪ | Prompt library | Durable, paste-able prompts; persistent as text, not as a running actor |
| 🔴 | Retired or superseded | Kept for history only |

## System map

```
                    ┌──────────────────────────────┐
                    │  ORCHESTRATION (who does work) │
                    │  /swarm + 5 subagents          │
                    └──────────────┬─────────────────┘
                                   │ delegates
    ┌──────────────────────────────┼──────────────────────────────┐
    │                              │                              │
┌───▼──────────────┐   ┌───────────▼───────────┐   ┌──────────────▼──────┐
│ FACTORY          │   │ LIBRARY                │   │ RHYTHM + MEMORY      │
│ ICM-Workspaces   │   │ D:\Creative Core       │   │ scheduled tasks,     │
│ 23 pipelines     │──►│ Wiki, Layer-0, CPIP    │◄──│ /recap, Open Brain   │
│ (wiki-capture    │   │ (knowledge deposits)   │   │ (cross-surface)      │
│  final stage)    │   │                        │   │                      │
└──────────────────┘   └────────────────────────┘   └──────────────────────┘
```

## 1. Orchestration and model-seat control

Use case: keep Fable 5 spend low by pushing reading, drafting, and bulk work to cheaper seats.

| Tier | Agent | Where | What it does | Captured |
|---|---|---|---|---|
| 🔵 | /swarm | ~\.claude\commands | Fan-out orchestrator; Fable or Opus seat designs, delegates, QCs | 2026-07-08, upgraded 2026-07-18 |
| 🔵 | opus-planner | ~\.claude\agents | Planning subagent (Opus) | 2026-07-08 |
| 🔵 | opus-specialist | ~\.claude\agents | Deep single-topic work (Opus) | 2026-07-08 |
| 🔵 | sonnet-worker | ~\.claude\agents | Research, drafting, bulk reads (Sonnet 5) | 2026-07-08 |
| 🔵 | haiku-runner | ~\.claude\agents | Cheap mechanical runs (Haiku 4.5) | 2026-07-08 |
| 🔵 | opus48-verifier | ~\.claude\agents | Independent refutation pass on a different model (Opus 4.8) | 2026-08-04 |

[Fact] Protocol doc: [[Agent Swarm Protocol - Fable 5 Orchestration]] and [[model-downgrade-playbook]]. Operating rule since 2026-08-19: at most 3 concurrent agents when an external API (Granola) is involved.

## 2. Session memory, handoff, and cross-surface continuity

Use case: never lose where a session left off; make facts reachable from claude.ai, Cowork, and Code.

| Tier | Agent | Where | What it does | Captured |
|---|---|---|---|---|
| 🔵 | /recap v2.1 | ~\.claude\commands\recap.md | One handoff report per session into ICM-Workspaces\_handoffs, master index SESSION-HANDOFF.md | live 2026-07-10, v2.1 2026-07-19 |
| ⚪ | chat-recap, cowork-recap | _config\surface-recaps | Same 4-section report for Chat and Cowork; one Open Brain capture each | 2026-07-06 |
| 🔵 | Memory Spine (session-close capture) | MEMORY-SPINE.md, PROCESS-MAPS.md | Claude proposes 2 to 5 Open Brain captures at session close; SHIP LIST in TASKS.md read aloud at open | 2026-07-04 |
| 🟢 | Open Brain Keep-Alive | _config\scripts\open-brain-keepalive.ps1 | Mon and Thu 12:15 PM ping so the Supabase free tier never pauses | 2026-07-22 |
| 🔵 | /triage | Claude Code command | Session cleanup review (fed by session-explorer) | 2026-07-25 |
| 🟡 | dashboard-forge | ICM-Workspaces\dashboard-forge | Builds local dashboards over session data; run #1 produced session-explorer | 2026-07-25, 2026-07-28 |

## 3. Daily rhythm and self-measurement

Use case: a deliberate cadence with a one-step manual override, not a background loop.

| Tier | Agent | Where | What it does | Captured |
|---|---|---|---|---|
| 🟢 | Daily Note Linker | Windows task, every 15 min | Lists every markdown file created today (Library and Factory) inside the daily note | 2026-07-22 |
| 🟢 | ladder-map-noon | Claude Code scheduler, 12:00 | Rebuilds the AI Leverage Ladder chart in the daily note | 2026-07-23 |
| 🟢 | ladder-map-night | Claude Code scheduler, 21:30 | Closing ladder run for the day | 2026-07-23 |
| 🔵 | /ladder-map | ~\.claude\commands\ladder-map.md | Manual rebuild, source of truth for the two tasks above | 2026-07-23 |
| 🔴 | hourly-ladder-map | deleted | Replaced by noon and night runs | retired 2026-07-23 |
| 🔵 | /self-weekly | Library session only | Weekly self review; privacy floor keeps it local | 2026-07-04 |
| ⚪ | Daily journaling workflow | Daily-Notes\_template.md | Morning intent, capture, end-of-day distill; weekly and monthly wiki upkeep | 2026-05-08 |

## 4. Document rendering and decision records

Use case: every deliverable lands in the format Ted actually reviews (paper, Word, Excel).

| Tier | Agent | Where | What it does | Captured |
|---|---|---|---|---|
| 🟢 | print-plan hook | PreToolUse on ExitPlanMode, _config\print-plan\print-plan.ps1 | Every plan approval auto-opens a printable Word review sheet plus its Excel workbook | 2026-07-23, xlsx opened since 2026-09-04 |
| 🔵 | /word | Claude Code command | Markdown to ADHD-formatted Word plus Excel companion, both opened | 2026-09-04 |
| 🔵 | /excel | Claude Code command | Excel workbook only, via _config\xlsx\build-adhd-xlsx.py | 2026-09-04 |
| 🔵 | build-boxed-docx.py | _config | Word renderer (section-per-box) | 2026-08-19 |
| 🔵 | /decide | Claude Code command | Runs the decision-options-map workflow without pasting | 2026-07-20 |
| ⚪ | Decision Triple | vault master contract | md plus docx plus xlsx on every settled decision | 2026-07-29 |

## 5. Knowledge capture into the Library

Use case: get raw material through the .raw firewall and into the Wiki as linked, curated entries.

| Tier | Agent | Where | What it does | Captured |
|---|---|---|---|---|
| 🟡 | wiki-capture (final stage, every pipeline) | each workspace | Deposits keepers into D:\Creative Core\Wiki | 2026-07-03 |
| ⚪ | library-capture-ai.json, library-capture-manual.json | _config\clipper | Obsidian Web Clipper templates; clips land in Wiki\_raw with routing frontmatter | 2026-08-07 |
| 🔵 | /wiki-ingest, /wiki-query, /wiki-lint, /save, /autoresearch | Library session commands | Ingest articles, check duplicates, catch orphans, save sessions, research a topic into Wiki | 2026-05-08 |
| 🟡 | transcript-capture | ICM-Workspaces | YouTube videos that already have captions | 2026-07-29 |
| 🟡 | vault-scribe | ICM-Workspaces | Images into vault notes | 2026-07-29 |
| 🟡 | audacity-transcription | ICM-Workspaces | Ted's own audio and video with no transcript | 2026-07-29 |
| 🟡 | notebook-forge | ICM-Workspaces | Gemini Notebook artifacts, keepers only cross into the Library | 2026-07-28 |
| 🟡 | content-analysis | ICM-Workspaces | U-M-CAP analysis of content | 2026-07-28 |

## 6. Meetings, CRM, and the CentSight seat

Use case: every Granola recording gets a processing home; the internship gets daily and weekly truth.

| Tier | Agent | Where | What it does | Captured |
|---|---|---|---|---|
| 🟢 | Granola to Obsidian nightly sync | scheduled task, D:\Creative Core\Granola\_sync-spec.md | Lands transcripts in the vault; Rule 7 holds sensitive recordings | 2026-08-19 |
| 🟡 | granola-triage | ICM-Workspaces\granola-triage | Classifies each meeting into 5 types and routes it | 2026-08-19 |
| 🟡 | granola-hubspot-sync | ICM-Workspaces | Client and CRM meetings into HubSpot | 2026-08-19 |
| 🟡 | victory-lap-internship | ICM-Workspaces\victory-lap-internship | Daily worklog, weekly review, SELL-200 check-ins | 2026-08-21 to 2026-08-25 |
| 🔵 | Roman's Day Log, CentSight Daily Pulse | Claude artifacts (stable URLs) | Daily dashboards; triggers "update my day log", "confirm my day log" | 2026-08-21 |
| 🔵 | CentSight Weekly Review | Claude artifact | Friday report to Gerald; triggers "build my week", "confirm my week" | 2026-08-23 |
| ⚪ | Granola Claude Code Prompt Library v1.0 | Library vault | 31 prompts, 8 categories | 2026-09-04 |

## 7. Career, LinkedIn, and outreach

Use case: job search, personal brand, and sales enablement with honesty rails.

| Tier | Agent | Where | What it does | Captured |
|---|---|---|---|---|
| 🟡 | job-search | ICM-Workspaces | Application pack per posting; recommended first end-to-end pilot | 2026-05-08 |
| 🟡 | linkedin-content | ICM-Workspaces | Ideate, draft, publish; post strategy v1 in _config | 2026-08-22, 2026-09-02 |
| 🟡 | ai-portfolio | ICM-Workspaces | Honest-fit portfolio site (Supabase plus Next.js); blocked at 01-content on 3 answers from Ted | 2026-07-18 |
| ⚪ | CPIP Domain-Prompts (6) | D:\creative core\CPIP\Domain-Prompts | job-search-application-pack, wiki-deposit-qc, self-weekly-quality-rubric, car-204-deliverable-builder, linkedin-frame-essay-builder, decision-options-map | 2026-07-19, 2026-07-20 |
| ⚪ | Sales enablement prompt library | _drafts\2026-08-21-... | 53 prompts, 16 categories, universal system block | 2026-08-21 |
| ⚪ | Stevie Case Prompt Library | D:\Creative Core\CentSight | 24 prompts, 7 categories | 2026-08-23 |

## 8. Coursework (NLU)

| Tier | Agent | Where | What it does | Captured |
|---|---|---|---|---|
| 🟡 | nlu-coursework | ICM-Workspaces\nlu-coursework | Graded deliverables per course and week (SELL-110, CAR-204, now SELL-200) | 2026-07-25 |
| ⚪ | course-tracker.md | Factory root | Thin mirror of vault assignment status | 2026-07-13 |

## 9. Liberation Labs and product sites

| Tier | Agent | Where | What it does | Captured |
|---|---|---|---|---|
| 🟡 | project-emet | ICM-Workspaces | Emet product work | 2026-07-03 |
| 🟡 | liblabs-website | ICM-Workspaces | Liberation Labs site | 2026-07-03 |
| 🟡 | thc-website | ICM-Workspaces | THC site | 2026-07-03 |
| 🟡 | website-storyboard | ICM-Workspaces (loose files also at Library root) | Site storyboarding | 2026-07-22 |
| 🔵 | FOXFIRE | external, ll.glitchlit.com | 3-track lead, visibility, and partnership scanner built by a collaborator | 2026-05-08 |

[Uncertain] The five older pipelines (job-search, linkedin-content, project-emet, liblabs-website, thc-website) had wiki-capture stages pointing at a dead path as of 2026-07-03. No capture records the fix.

## 10. Machine maintenance and asset forging

| Tier | Agent | Where | What it does | Captured |
|---|---|---|---|---|
| ⚪ | Free-D-Space prompt plus Free-Up-Space.ps1 | C:\Users\tedal | Propose-only Recycle-Bin cleanup; Ted runs the script | 2026-06-20 |
| ⚪ | sysaudit scripts plus 17-prompt system ladder | _config\scripts\sysaudit, _drafts\2026-09-05-... | Read-only laptop audit | 2026-08-21, 2026-09-05 |
| 🟡 | icon-forge | ICM-Workspaces | Icon generation (named as a sibling of dashboard-forge; no direct capture) | [Uncertain] |

## Most persistent, ranked

Ranked by how often they fire without Ted and how many other agents depend on them.

1. 🟢 Daily Note Linker (every 15 minutes, both vaults)
2. 🟢 print-plan hook (every plan approval)
3. 🟢 ladder-map-noon and ladder-map-night (twice daily)
4. 🟢 Granola nightly sync (nightly, feeds granola-triage and the internship journal)
5. 🟢 Open Brain Keep-Alive (twice weekly, keeps cross-surface memory alive)
6. 🔵 /recap plus the Memory Spine (every Code session)
7. 🔵 /swarm plus 5 subagents (every delegated session)
8. 🟡 wiki-capture final stage (every pipeline run)

## Known gaps in this inventory

- [Fact] 4 or more Factory workspaces unnamed in memory (Handbook says 23).
- [Uncertain] Which of the older pipelines have run end to end.
- [Uncertain] Whether the /self-weekly, /wiki-* and /save commands still exist under those names.
- [Fact] No user-level agent list was read; the five subagents are known only from captures.

## Related

- [[Factory Pipeline Handbook]]
- [[Agent Swarm Protocol - Fable 5 Orchestration]]
- [[Factory Playbook]], [[Library Playbook]], [[Cross-Surface Spine]]
- [[Ladder Mapping (Runbook)]]
