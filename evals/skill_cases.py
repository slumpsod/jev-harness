"""Live eval of Jev's skill picking (jev.pick_skills) plus an offline threshold sweep
over the same decide_skills() rule the harness uses.

Cases: the 52 labelled requests from openjev/router_tune_semif.py (40 targeted + 12
no-skill controls), plus harness-local skills. Each case runs against the persona whose
skills_allowed contains the expected skill, i.e. after correct routing, so this measures
skill picking alone.

    python evals/skill_cases.py            # live pass (~80s on semif), then sweep
    python evals/skill_cases.py --offline  # re-sweep the cached raw answers

History (2026-09-27, semif): per-skill nouls alone peaked at 69-79% clean picks - three
phrasings tried; absolute nouls rate sibling skills ~0.99 alike and cost ~0.18s each.
Choice over all skills (+ none option) then a fits noul on the top 3: 94% clean,
100% correct silence, 2 needless loads, ~1.3s per turn.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jevh.config import load_config  # noqa: E402
from jevh.jev import Jev, decide_skills  # noqa: E402
from jevh.personas import load_personas  # noqa: E402
from jevh.skills import load_roster  # noqa: E402
import _out  # noqa: E402  (evals/_out.py: where a run's results go)

# (id, expected skill or None, request, persona override for None cases, acceptable extras)
CASES = [
    ("insulation", "eco-insulation-quotes", "Put together a quote for 2,400 sqft of R-30 insulation in a Spanish Fork home, including two exhaust fans and an air-sealing package, with the Enbridge rebate applied."),
    ("payroll-field", "payroll-tracker", "Log the job I did today: 6 hours of labor at 75 an hour plus 300 in materials, pay it to Talon, and add it to this week's payroll batch."),
    ("payroll-ins", "insulation-specialists-payroll", "Pull the Insulation Specialists payroll ledger for last month and check the totals before I send it to Kristopher."),
    ("calendar-add", "appointment-scheduling", "Add an appointment: Thursday 2pm, customer Dave Miller, 430 Cleveland St, phone 435-555-0199, bring the radon kit."),
    ("calendar-multi", "google-workspace-multi-calendar", "What's on my calendar for the rest of the week? Check the inspections and installs calendars too, not just the main one."),
    ("gmail-search", "email-search", "Search my Gmail for the 1099 Justin Pacheco sent me last year and tell me what the total was."),
    ("gmail-attach", "gmail-attachments", "Download the contract attachment from the email Kris sent this morning."),
    ("gmail-send", "google-workspace", "Send a follow-up email to the customer from the meeting this morning, signed from the Salt Lake office."),
    ("gws-cli-trap", "gws-cli", "The gws CLI is throwing a weird error when I send an attachment - walk through the known traps."),
    ("comfyui", "comfyui", "Run the image-to-video workflow on this reference image, 5 seconds, MiniMax H3."),
    ("blender-mcp", "blender-mcp", "Drive the live Blender GUI and set up a three-point lighting rig over the car model in the scene."),
    ("blender-proc", "blender-3d-modeling", "Procedurally build a low-poly rocket ship model in Blender from primitives."),
    ("blender-render", "blender-render-automation", "Render the current scene from the command line in Cycles, 200 samples, frames 1 through 48."),
    ("hue", "openhue", "Turn off all the living room lights and set the bedroom to 40 percent warm."),
    ("roku", "roku-ecp", "Switch the living room Roku to HDMI input 2 using the ECP script."),
    ("cyberpunk", "cyberpunk-2077-modding", "Set up the redscript mod so the input loader XML gets picked up by the game."),
    ("dlss", "dlss5-game-injection", "Inject the DLSS 5 NR preset into the game using RenoDX and confirm it's active."),
    ("minecraft", "minecraft-modpack-server", "Stand up a Minecraft modpack server from a CurseForge pack with a whitelist of three players."),
    ("pokemon", "pokemon-player", "Send my Pikachu to attack with Thunderbolt in the Pokemon battle."),
    ("radon", "radon-mitigation-study", "Quiz me on radon: what's the difference between a can test and a stack test and when do you use each?"),
    # seen live in the dashboard 2026-09-28: Choice 0.70, fits 0.09 -> not loaded
    ("radon-short", "radon-mitigation-study", "Quiz me on radon: what's the difference between a can test and a stack test?"),
    ("fitness", "fitness-tracker", "Update my training log: bench press went up to 225 for five reps this week."),
    ("claude-code", "claude-code", "Delegate the failing test on the Kaisen branch to Claude Code and have it open a PR when it passes."),
    ("hermes-agent", "hermes-agent", "Configure this Hermes agent: add a new MCP server to the config and make sure the tools register."),
    ("hermes-webapp", "hermes-local-web-apps", "Build a local dashboard web app that pulls my payroll data and serves it on the tailnet."),
    ("prime-agent", "prime-agent", "Delegate the refactor to Prime Agent in WSL and watch it make progress."),
    ("github-pr", "github-pr-workflow", "Open a PR on the texas-insulation-calculator repo, wait for CI, and merge it once green."),
    ("native-mcp", "native-mcp", "Add the blender MCP server to my OpenCode config and verify the tools show up."),
    ("web-research", "web-research-workflow", "Research the history of radon mitigation standards and build a sourced timeline."),
    ("journalist", "journalist", "Deep-dive investigation: fact-check the claims in this article about Douglas Arfman and build a sourced timeline."),
    ("local-ai", "local-ai-models", "Pick a local LLM that fits the 4080's 16GB and set it up for serving."),
    ("video-analysis", "video-analysis", "Extract frames from this security camera clip and tell me what happened between 2 and 3am."),
    ("maps", "maps", "Geocode 430 Cleveland St Orem and give me the driving route from Spanish Fork."),
    ("serpapi", "serpapi-image-search", "Search Google Images for photos of attics with R-30 insulation and return the image URLs."),
    ("doc-tools", "document-tools", "Extract text out of this scanned PDF, OCR it, and fill in the signature field."),
    ("desktop-gui", "desktop-gui-automation", "Automate the desktop app: take a screenshot, find the button visually, and click it."),
    ("win-admin", "windows-system-admin", "Add a registry key and an autostart entry for the new service, from git-bash."),
    ("openjev-judge", "openjev-judgment-tools", "Build a judgment tool on the local openjev model that scores whether a claim is supported by a document."),
    ("openjev-up", "openjev", "Check that the local openjev NLI server is up and which model it's serving."),
    ("koe-hash", "koe-no-hashi-voice-dashboard", "Wire Koe-no-Hashi TTS into the voice dashboard and test a phrase."),
    ("hermes-tui", "debugging-hermes-tui-commands", "Debug why a TUI slash command crashes on the gateway - check the Python side and the Ink UI."),
    # connecting the user's own accounts (nango-setup), added 2026-10-02
    ("nango-gmail", "nango-setup", "I want the assistant to be able to read my Gmail. Walk me through connecting it."),
    ("nango-calendar", "nango-setup", "Connect my Google Calendar so you can see my schedule."),
    ("nango-github", "nango-setup", "Help me link my GitHub account to the assistant."),
    ("nango-reconnect", "nango-setup", "My Gmail connection stopped working, I think the login expired. Reconnect it."),
    ("nango-start", "nango-setup", "Set up Nango on this PC so you can use my accounts."),
    ("nango-slack", "nango-setup", "I'd like you to be able to post in my Slack. How do I connect it?"),
    ("nango-todoist", "nango-setup", "Hook up my Todoist account to the assistant."),
    ("none-oauth", None, "What is OAuth and why do apps ask for a redirect URL?"),
    # harness-local skills
    ("lift-split", "workout-programming", "Make me a 4 day upper/lower split for strength."),
    ("deload", "workout-programming", "Plan a deload week, I've stalled on bench for three sessions."),
    ("landing", "frontend-design", "Design a landing page for my insulation business with a clean modern look."),
    ("palette", "frontend-design", "Pick a color palette and type scale for the company website."),
    ("pixel", "pixel-art", "Make a pixel-art NES-style sprite of a hard hat."),
    ("comic", "baoyu-comic", "Make a short educational comic explaining how attic insulation works."),
    ("comfy-img", "comfyui", "Generate a hero image of mountains at sunset with Qwen Image."),
    ("webapp-simple", "simple-web-app-build", "Build a simple single-page calculator tool for R-value estimates my customers can use."),
    # controls: no skill should load
    ("none-math", None, "What's 17 percent of 2,340?"),
    ("none-fact", None, "What year did the US get the moon landing?"),
    ("none-chat", None, "What's a good way to describe the weather in a short story without using the word 'cold'?"),
    ("none-opinion", None, "Is a 27B local model worth the GPU hassle over just using an API key?", "high_end_dev"),
    ("none-general", None, "Explain how a blower door test works for a customer who has never heard of one."),
    ("none-translate", None, "Translate 'we'll be there by nine' into Spanish."),
    ("none-writing", None, "Write a polite email subject line for a request to reschedule a meeting."),
    ("none-trivia2", None, "What's the capital of Utah?"),
    ("none-py", None, "What's the difference between a list and a tuple in Python?", "high_end_dev"),
    ("none-pack", None, "What should I pack for a three-day trip to Moab in September?"),
    ("none-fact2", None, "How tall is Mount Rushmore?"),
    ("none-joke", None, "Tell me a dad joke about insulation."),
    ("none-protein", None, "How much protein should I eat a day at 85kg?", "fitness_coach"),
    ("none-accent", None, "Quick gut check: is navy or charcoal the better accent for a roofing company?", "front_end_designer"),
]

# Siblings that are fine to load alongside the expected skill (not counted as needless).
ACCEPTABLE = {
    "fitness": {"workout-programming"},
    "hermes-webapp": {"simple-web-app-build"},
    "webapp-simple": {"hermes-local-web-apps", "frontend-design"},
    "landing": {"simple-web-app-build"},
    "openjev-judge": {"openjev", "openjev-systemone"},
    "openjev-up": {"openjev-systemone"},
    "calendar-add": {"google-workspace", "google-workspace-multi-calendar"},
    "calendar-multi": {"google-workspace", "gws-cli"},
    "gmail-search": {"google-workspace", "gmail-attachments"},
    "gmail-attach": {"google-workspace", "email-search"},
    "gmail-send": {"gws-cli"},
    "gws-cli-trap": {"google-workspace"},
    "web-research": {"radon-mitigation-study"},
    "journalist": {"web-research-workflow"},
    "comic": {"baoyu-article-illustrator"},
    "blender-proc": {"blender-scripting"},
    "blender-render": {"blender-scripting"},
    "blender-mcp": {"blender-scripting"},
    "comfy-img": {"comfyui-3d-generation"},
    "nango-gmail": {"google-workspace"}, "nango-calendar": {"google-workspace"}, "nango-reconnect": {"google-workspace"},
    # "none" controls where, after routing to this persona, one skill is a fair load
    "none-opinion": {"local-ai-models"},
    "none-accent": {"frontend-design"},
}

# Siblings that are fine to load alongside the expected skill (not counted as needless).
ACCEPTABLE = {
    "fitness": {"workout-programming"},
    "hermes-webapp": {"simple-web-app-build"},
    "webapp-simple": {"hermes-local-web-apps", "frontend-design"},
    "landing": {"simple-web-app-build"},
    "openjev-judge": {"openjev", "openjev-systemone"},
    "openjev-up": {"openjev-systemone"},
    "calendar-add": {"google-workspace", "google-workspace-multi-calendar"},
    "calendar-multi": {"google-workspace", "gws-cli"},
    "gmail-search": {"google-workspace", "gmail-attachments"},
    "gmail-attach": {"google-workspace", "email-search"},
    "gmail-send": {"gws-cli"},
    "gws-cli-trap": {"google-workspace"},
    "web-research": {"radon-mitigation-study"},
    "journalist": {"web-research-workflow"},
    "comic": {"baoyu-article-illustrator"},
    "blender-proc": {"blender-scripting"},
    "blender-render": {"blender-scripting"},
    "blender-mcp": {"blender-scripting"},
    "comfy-img": {"comfyui-3d-generation"},
    "nango-gmail": {"google-workspace"}, "nango-calendar": {"google-workspace"}, "nango-reconnect": {"google-workspace"},
    # "none" controls where, after routing to this persona, one skill is a fair load
    "none-opinion": {"local-ai-models"},
    "none-accent": {"frontend-design"},
}


def persona_for(case, personas) -> str:
    cid, want, text, *rest = case
    if rest:
        return rest[0]
    if want is None:
        return "daily_assistant"
    for name in ("comfyui_manager", "fitness_coach", "front_end_designer", "daily_assistant", "high_end_dev"):
        if personas[name].allows_skill(want):
            return name
    raise ValueError(f"{cid}: no persona allows {want}")


def run(cfg, personas, roster) -> list[dict]:
    cfg.thresholds.update(skill_none_veto=1.01)  # never veto live, so fits are always captured
    jev = Jev(cfg)
    rows = []
    for case in CASES:
        cid, want, text = case[:3]
        pname = persona_for(case, personas)
        t = time.time()
        d = jev.pick_skills(text, personas[pname], roster)
        rows.append({"id": cid, "want": want, "persona": pname, "probs": d.probs, "fits": d.fits,
                     "secs": round(time.time() - t, 2)})
    return rows


def score(rows, fits_t, override, veto, k=1) -> dict:
    clean = none_ok = extras = 0
    n_none = sum(r["want"] is None for r in rows)
    fails = []
    for r in rows:
        picked = decide_skills(r["probs"], r["fits"], fits_t, override, veto, k)
        ok = ACCEPTABLE.get(r["id"], set()) | ({r["want"]} if r["want"] else set())
        n_extra = sum(p not in ok for p in picked)
        extras += n_extra
        good = all(p in ok for p in picked) if r["want"] is None else (r["want"] in picked and not n_extra)
        none_ok += good and r["want"] is None
        clean += good and r["want"] is not None
        if not good:
            fails.append((r["id"], r["want"], picked))
    n = len(rows)
    return {"clean": clean / (n - n_none), "none_ok": none_ok / n_none, "extras": extras,
            "objective": (clean + none_ok - 0.5 * extras) / n, "fails": fails}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    _out.add_args(ap)
    a = ap.parse_args()
    dest = _out.Out.of(a)
    cache = Path(__file__).with_name("skill_raw.json")
    cfg = load_config()
    cur = (cfg.t("skill_fits"), cfg.t("skill_override"), cfg.t("skill_none_veto"), cfg.lim("max_skills"))
    if a.offline:
        rows = json.loads(dest.read(cache).read_text(encoding="utf-8"))
    else:
        t = time.time()
        rows = run(cfg, load_personas(cfg.personas_dir), load_roster(cfg.skills_dirs))
        dest.write(cache).write_text(json.dumps(rows, indent=1), encoding="utf-8")
        print(f"raw answers saved to {dest.where(dest.write(cache))}")
        secs = sorted(r["secs"] for r in rows)
        print(f"{len(rows)} cases in {time.time() - t:.0f}s (median {secs[len(secs) // 2]:.2f}s per pick)")
    s = score(rows, *cur)
    print(f"config fits={cur[0]} override={cur[1]} veto={cur[2]} k={cur[3]}: clean {s['clean']:.0%} | "
          f"none_ok {s['none_ok']:.0%} | needless {s['extras']}")
    for cid, want, picked in s["fails"]:
        print(f"  x {cid:15s} want {want}  got {picked}")
    print("\nsweep (top 8)\n  fits  override veto  k | clean none_ok needless")
    grid = itertools.product((0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3), (0.8, 0.9, 1.01), (0.8, 0.9, 1.01), (1, 2))
    res = sorted(((score(rows, *g), g) for g in grid), key=lambda t: -t[0]["objective"])
    for r, (f, o, v, k) in res[:8]:
        print(f"  {f:.2f}  {o:.2f}     {v:.2f}  {k} | {r['clean']:.0%}  {r['none_ok']:.0%}    {r['extras']}")


if __name__ == "__main__":
    main()
