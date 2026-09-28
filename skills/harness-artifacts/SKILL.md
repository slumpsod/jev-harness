---
name: harness-artifacts
description: Handing work between assistants - writing request() briefs and passing files as artifacts.
---
# Working with other assistants

- The assistant you call sees ONLY your task text and the artifact paths in `inputs`.
  Write a brief like a ticket: goal, constraints, exact deliverable format, and file names.
- Pass files, never pasted contents: `request("front_end_designer", brief, inputs=[spec_path])`.
- The reply is a dict: `r["status"]` ("done" | "rejected" | "failed"), `r["answer"]`, `r["artifacts"]`.
  If rejected, read the reason - usually the wrong assistant or an unnecessary step.
- Save your deliverables with `save_artifact(name, data)` and list the paths in `final(...)`.
- Split big projects: spec -> design -> build -> integrate. Ask for each piece once.
