"""
Sprint 1 — Task 1.1: Generate synthetic learning data.

Outputs:
  data/seed_courses.json    — 200 courses
  data/seed_learners.json   — 500 learners
  data/seed_xapi_events.json — 50 000+ xAPI statements

Intentional patterns baked in:
  • Module 3–4 drop-off in ~30% of courses (visible in completion funnels)
  • Engineering dept has Python / ML skill gaps (low proficiency)
  • Blended courses complete at 25% higher rate than self-paced
  • Peak engagement at 10 am and 2 pm (UTC)
  • Some courses have very low quiz scores on specific modules
"""
from __future__ import annotations

import json
import math
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── Seed for reproducibility ──────────────────────────────────────────────────
random.seed(42)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── Domain constants ──────────────────────────────────────────────────────────
DEPARTMENTS = ["Engineering", "Sales", "Marketing", "HR", "Finance", "Operations"]

DEPT_SKILLS: dict[str, list[str]] = {
    "Engineering": [
        "Python", "Machine Learning", "Data Engineering", "Cloud Architecture",
        "DevOps", "SQL", "System Design", "Kubernetes", "Go", "Security",
    ],
    "Sales": [
        "CRM Tools", "Negotiation", "Presentation Skills", "Product Knowledge",
        "Lead Generation", "Account Management", "Salesforce", "Cold Outreach",
    ],
    "Marketing": [
        "SEO/SEM", "Content Strategy", "Data Analytics", "Social Media",
        "Email Marketing", "Brand Management", "A/B Testing", "Copywriting",
    ],
    "HR": [
        "Talent Acquisition", "Performance Management", "Employment Law",
        "HRIS Systems", "Compensation & Benefits", "DEI Practices", "Coaching",
    ],
    "Finance": [
        "Financial Modeling", "Excel/Sheets", "GAAP", "Budgeting",
        "Risk Management", "Tax Compliance", "ERP Systems", "Forecasting",
    ],
    "Operations": [
        "Process Improvement", "Supply Chain", "Lean/Six Sigma", "Project Management",
        "Vendor Management", "Quality Assurance", "Logistics", "ERP Systems",
    ],
}

# Engineering Python/ML skills have intentionally low base proficiency
DEPT_SKILL_PROFICIENCY_BASE: dict[str, dict[str, float]] = {
    "Engineering": {
        "Python": 1.8,           # gap: target 3.5
        "Machine Learning": 1.5, # gap: target 3.5
        "Data Engineering": 2.0,
        "Cloud Architecture": 2.5,
        "DevOps": 2.8,
        "SQL": 3.0,
        "System Design": 2.6,
        "Kubernetes": 2.2,
        "Go": 2.4,
        "Security": 2.7,
    },
}

COURSE_FORMATS = ["self-paced", "blended", "instructor-led", "cohort-based"]

SKILL_LEVELS = ["beginner", "intermediate", "advanced"]

COURSE_TOPICS: dict[str, list[tuple[str, list[str]]]] = {
    "Engineering": [
        ("Python for Data Science", ["Python", "Data Engineering", "SQL"]),
        ("Machine Learning Fundamentals", ["Machine Learning", "Python"]),
        ("Advanced ML with PyTorch", ["Machine Learning", "Python"]),
        ("MLOps in Production", ["Machine Learning", "DevOps", "Kubernetes"]),
        ("Cloud Architecture on Azure", ["Cloud Architecture", "Security"]),
        ("Kubernetes Deep Dive", ["Kubernetes", "DevOps"]),
        ("Data Engineering Pipelines", ["Data Engineering", "SQL", "Python"]),
        ("SQL for Engineers", ["SQL", "Data Engineering"]),
        ("System Design Patterns", ["System Design", "Cloud Architecture"]),
        ("Go Programming Essentials", ["Go", "System Design"]),
        ("DevSecOps Practices", ["DevOps", "Security"]),
        ("Python Best Practices", ["Python", "System Design"]),
    ],
    "Sales": [
        ("Salesforce CRM Mastery", ["CRM Tools", "Salesforce", "Account Management"]),
        ("Consultative Selling", ["Negotiation", "Presentation Skills"]),
        ("Cold Outreach Playbook", ["Cold Outreach", "Lead Generation"]),
        ("Enterprise Account Management", ["Account Management", "Negotiation"]),
        ("Product Demo Excellence", ["Presentation Skills", "Product Knowledge"]),
        ("Lead Generation at Scale", ["Lead Generation", "CRM Tools"]),
    ],
    "Marketing": [
        ("SEO & SEM Strategy", ["SEO/SEM", "Data Analytics"]),
        ("Content Marketing Mastery", ["Content Strategy", "Copywriting"]),
        ("Marketing Analytics", ["Data Analytics", "A/B Testing"]),
        ("Social Media Management", ["Social Media", "Brand Management"]),
        ("Email Marketing Automation", ["Email Marketing", "A/B Testing"]),
        ("Brand Building 101", ["Brand Management", "Content Strategy"]),
    ],
    "HR": [
        ("Modern Talent Acquisition", ["Talent Acquisition", "HRIS Systems"]),
        ("Performance Management Cycle", ["Performance Management", "Coaching"]),
        ("Employment Law Essentials", ["Employment Law", "Compliance"]),
        ("DEI in the Workplace", ["DEI Practices", "Coaching"]),
        ("Compensation Strategy", ["Compensation & Benefits", "HRIS Systems"]),
        ("Leadership Coaching Skills", ["Coaching", "Performance Management"]),
    ],
    "Finance": [
        ("Financial Modeling in Excel", ["Financial Modeling", "Excel/Sheets"]),
        ("GAAP Accounting Standards", ["GAAP", "Compliance"]),
        ("Budgeting & Forecasting", ["Budgeting", "Forecasting", "Financial Modeling"]),
        ("Enterprise Risk Management", ["Risk Management", "GAAP"]),
        ("ERP for Finance Teams", ["ERP Systems", "Financial Modeling"]),
        ("Tax Planning Strategies", ["Tax Compliance", "GAAP"]),
    ],
    "Operations": [
        ("Lean Six Sigma Green Belt", ["Lean/Six Sigma", "Process Improvement"]),
        ("Supply Chain Optimization", ["Supply Chain", "Vendor Management"]),
        ("Project Management Professional", ["Project Management", "Process Improvement"]),
        ("Quality Assurance Frameworks", ["Quality Assurance", "Lean/Six Sigma"]),
        ("Vendor Management & SLAs", ["Vendor Management", "Contract Management"]),
        ("Logistics & Distribution", ["Logistics", "Supply Chain"]),
    ],
}

XAPI_VERBS = {
    "launched":    "http://adlnet.gov/expapi/verbs/launched",
    "progressed":  "http://adlnet.gov/expapi/verbs/progressed",
    "completed":   "http://adlnet.gov/expapi/verbs/completed",
    "scored":      "http://adlnet.gov/expapi/verbs/scored",
    "interacted":  "http://adlnet.gov/expapi/verbs/interacted",
    "attempted":   "http://adlnet.gov/expapi/verbs/attempted",
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. COURSES
# ─────────────────────────────────────────────────────────────────────────────

def _module_titles(topic: str, n: int) -> list[str]:
    prefixes = [
        "Introduction to", "Core Concepts of", "Deep Dive into",
        "Applying", "Advanced", "Hands-on", "Capstone:"
    ]
    return [f"{prefixes[i % len(prefixes)]} {topic}" for i in range(n)]


def generate_courses(n: int = 200) -> list[dict]:
    courses: list[dict] = []
    course_idx = 0

    for dept, topic_list in COURSE_TOPICS.items():
        for title, skills in topic_list:
            for level in SKILL_LEVELS:
                if course_idx >= n:
                    break
                course_id = f"COURSE-{course_idx + 1:04d}"
                fmt = random.choice(COURSE_FORMATS)
                duration = random.choice([4, 8, 12, 16, 20, 24, 32, 40])
                num_modules = random.randint(4, 7)

                # Flag ~30% of courses as "drop-off prone" (modules 3–4 will lose learners)
                dropoff_prone = random.random() < 0.30

                # Flag ~15% of courses as having hard quiz in module 2
                hard_quiz_module = random.choice([2, 3]) if random.random() < 0.15 else None

                courses.append({
                    "course_id": course_id,
                    "title": f"{title} - {level.title()}",
                    "department": dept,
                    "skill_level": level,
                    "format": fmt,
                    "duration_hours": duration,
                    "skills_taught": skills,
                    "description": (
                        f"A {level} {fmt} course covering {', '.join(skills[:2])} "
                        f"for {dept} professionals. Duration: {duration} hours across "
                        f"{num_modules} modules."
                    ),
                    "num_modules": num_modules,
                    "module_titles": _module_titles(title, num_modules),
                    "prerequisites": (
                        [f"COURSE-{max(1, course_idx - 1):04d}"]
                        if level != "beginner" and course_idx > 0
                        else []
                    ),
                    # Hidden metadata used by the event generator
                    "_dropoff_prone": dropoff_prone,
                    "_hard_quiz_module": hard_quiz_module,
                })
                course_idx += 1
            if course_idx >= n:
                break
        if course_idx >= n:
            break

    # Fill remaining slots if we have fewer topic combinations
    while len(courses) < n:
        dept = random.choice(DEPARTMENTS)
        course_idx = len(courses)
        course_id = f"COURSE-{course_idx + 1:04d}"
        topic, skills = random.choice(COURSE_TOPICS[dept])
        level = random.choice(SKILL_LEVELS)
        fmt = random.choice(COURSE_FORMATS)
        duration = random.choice([4, 8, 12, 16, 20, 24, 32, 40])
        num_modules = random.randint(4, 7)
        courses.append({
            "course_id": course_id,
            "title": f"{topic} - {level.title()} (Extended)",
            "department": dept,
            "skill_level": level,
            "format": fmt,
            "duration_hours": duration,
            "skills_taught": skills,
            "description": (
                f"Extended {level} course on {', '.join(skills[:2])} for "
                f"{dept} teams. {duration}h, {num_modules} modules."
            ),
            "num_modules": num_modules,
            "module_titles": _module_titles(topic, num_modules),
            "prerequisites": [],
            "_dropoff_prone": random.random() < 0.30,
            "_hard_quiz_module": random.choice([2, 3]) if random.random() < 0.15 else None,
        })

    return courses


# ─────────────────────────────────────────────────────────────────────────────
# 2. LEARNERS
# ─────────────────────────────────────────────────────────────────────────────

FIRST_NAMES = [
    "Alex", "Jordan", "Taylor", "Morgan", "Riley", "Casey", "Jamie", "Avery",
    "Blake", "Cameron", "Drew", "Emery", "Finley", "Harper", "Hayden", "Jessie",
    "Kai", "Lee", "Logan", "Mackenzie", "Madison", "Parker", "Peyton", "Quinn",
    "Reese", "Rowan", "Sage", "Sam", "Skylar", "Spencer", "Sydney", "Terry",
    "Tyler", "Uma", "Val", "Whitney", "Zion", "Nour", "Priya", "Aiden",
    "Lena", "Omar", "Fatima", "Yuki", "Chen", "Sofia", "Lars", "Amara",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Wilson", "Moore", "Taylor", "Anderson", "Thomas", "Jackson", "White", "Harris",
    "Martin", "Thompson", "Young", "Walker", "Hall", "Allen", "Wright", "King",
    "Scott", "Green", "Baker", "Adams", "Nelson", "Hill", "Ramirez", "Campbell",
    "Mitchell", "Roberts", "Carter", "Phillips", "Evans", "Turner", "Torres", "Parker",
    "Collins", "Edwards", "Stewart", "Morris", "Murphy", "Cook", "Rogers", "Morgan",
    "Peterson", "Cooper",
]


def _skill_set(dept: str) -> list[dict]:
    """Return a list of skill objects with proficiency levels."""
    base = DEPT_SKILL_PROFICIENCY_BASE.get(dept, {})
    skills = DEPT_SKILLS.get(dept, [])
    result = []
    for skill in skills:
        base_prof = base.get(skill, random.uniform(2.5, 4.2))
        # Add per-learner noise
        prof = round(max(1.0, min(5.0, base_prof + random.gauss(0, 0.4))), 2)
        result.append({"skill_name": skill, "proficiency_level": prof})
    return result


def generate_learners(n: int = 500) -> list[dict]:
    learners: list[dict] = []
    used_emails: set[str] = set()

    for i in range(n):
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        dept = random.choice(DEPARTMENTS)
        email_base = f"{first.lower()}.{last.lower()}{i}"
        email = f"{email_base}@novatech.example.com"
        while email in used_emails:
            email = f"{email_base}{random.randint(1, 99)}@novatech.example.com"
        used_emails.add(email)

        hire_date = datetime.now(timezone.utc) - timedelta(days=random.randint(90, 1825))
        learners.append({
            "learner_id": f"LRN-{i + 1:04d}",
            "name": f"{first} {last}",
            "email": email,
            "department": dept,
            "job_title": f"{dept} Specialist" if i % 3 != 0 else f"Senior {dept} Lead",
            "manager_email": f"manager.{dept.lower()}@novatech.example.com",
            "hire_date": hire_date.strftime("%Y-%m-%d"),
            "skills": _skill_set(dept),
            "completed_courses": [],  # filled in after event generation
        })

    return learners


# ─────────────────────────────────────────────────────────────────────────────
# 3. xAPI EVENTS
# ─────────────────────────────────────────────────────────────────────────────

# Simulation window: last 12 months
SIM_END = datetime.now(timezone.utc)
SIM_START = SIM_END - timedelta(days=365)


def _biased_timestamp() -> datetime:
    """Return a timestamp biased toward 10 am and 2 pm UTC on weekdays."""
    day_offset = random.randint(0, 364)
    ts = SIM_START + timedelta(days=day_offset)

    # 70% chance of a weekday
    if random.random() < 0.70:
        # Shift to nearest weekday (Mon–Fri)
        while ts.weekday() >= 5:
            ts += timedelta(days=1)

    # Hour distribution: peaks at 10 and 14
    r = random.random()
    if r < 0.30:
        hour = 10
    elif r < 0.55:
        hour = 14
    elif r < 0.70:
        hour = random.randint(9, 11)
    elif r < 0.85:
        hour = random.randint(13, 15)
    else:
        hour = random.randint(8, 17)

    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return ts.replace(hour=hour, minute=minute, second=second, tzinfo=timezone.utc)


def _duration_iso(minutes: float) -> str:
    h = int(minutes // 60)
    m = int(minutes % 60)
    s = int((minutes * 60) % 60)
    return f"PT{h}H{m}M{s}S" if h else f"PT{m}M{s}S"


def _make_event(
    verb_key: str,
    learner: dict,
    course: dict,
    module_idx: int,
    ts: datetime,
    score: float | None = None,
    duration_min: float | None = None,
    completed: bool = False,
) -> dict:
    module_id = f"MOD-{module_idx + 1:02d}"
    object_id = f"https://lms.novatech.example.com/courses/{course['course_id']}/modules/{module_id}"

    event: dict = {
        "id": str(uuid.uuid4()),
        "actor": {
            "objectType": "Agent",
            "name": learner["name"],
            "mbox": f"mailto:{learner['email']}",
        },
        "verb": {
            "id": XAPI_VERBS[verb_key],
            "display": {"en-US": verb_key},
        },
        "object": {
            "objectType": "Activity",
            "id": object_id,
            "definition": {
                "name": {"en-US": course["module_titles"][module_idx]},
                "type": "http://adlnet.gov/expapi/activities/module",
            },
        },
        "context": {
            "extensions": {
                "https://lms.novatech.example.com/ext/course_id": course["course_id"],
                "https://lms.novatech.example.com/ext/module_id": module_id,
                "https://lms.novatech.example.com/ext/department": learner["department"],
                "https://lms.novatech.example.com/ext/format": course["format"],
            }
        },
        "timestamp": ts.isoformat(),
        "stored": ts.isoformat(),
        "authority": {
            "objectType": "Agent",
            "name": "NovaTech LRS",
            "mbox": "mailto:lrs@novatech.example.com",
        },
    }

    if score is not None:
        event["result"] = {
            "score": {"scaled": round(score, 3), "raw": round(score * 100, 1), "min": 0, "max": 100},
            "success": score >= 0.70,
            "completion": completed,
        }
    if duration_min is not None:
        result = event.setdefault("result", {})
        result["duration"] = _duration_iso(duration_min)

    return event


def _simulate_enrollment(
    learner: dict,
    course: dict,
    enroll_ts: datetime,
    events: list[dict],
    completed_courses: set[str],
) -> None:
    """
    Simulate a learner's journey through a course, appending xAPI events.

    Patterns:
      - blended courses have 25% higher completion than self-paced
      - drop-off prone courses: 30% of learners stop at module 3 or 4
      - hard quiz modules: score distribution shifted downward
    """
    num_modules = course["num_modules"]
    fmt = course["format"]
    dropoff_prone = course.get("_dropoff_prone", False)
    hard_quiz_mod = course.get("_hard_quiz_module")  # 0-indexed

    # Base completion probability by format
    base_completion_prob = {
        "blended": 0.68,
        "instructor-led": 0.72,
        "cohort-based": 0.75,
        "self-paced": 0.43,  # blended is ~25% higher than self-paced
    }[fmt]

    # Drop-off prone adjustment (fewer learners reach module 3+)
    dropoff_at_module = None
    if dropoff_prone and random.random() < 0.30:
        dropoff_at_module = random.choice([2, 3])  # 0-indexed: module 3 or 4

    cur_ts = enroll_ts
    module_duration_base = (course["duration_hours"] * 60) / num_modules  # minutes

    for mod_idx in range(num_modules):
        # Stop if learner dropped off at this module
        if dropoff_at_module is not None and mod_idx >= dropoff_at_module:
            break

        # Gap between modules (0.5–5 days)
        if mod_idx > 0:
            cur_ts = cur_ts + timedelta(hours=random.uniform(12, 120))

        # 1. Launch module
        launch_ts = _biased_timestamp()
        # Keep date but use biased hour
        launch_ts = cur_ts.replace(hour=launch_ts.hour, minute=launch_ts.minute, second=launch_ts.second)
        events.append(_make_event("launched", learner, course, mod_idx, launch_ts))

        # 2. Interaction events (2–6 per module)
        for _ in range(random.randint(2, 6)):
            interact_ts = launch_ts + timedelta(minutes=random.uniform(5, module_duration_base * 0.6))
            events.append(_make_event("interacted", learner, course, mod_idx, interact_ts,
                                      duration_min=random.uniform(3, 15)))

        # 3. Progress events
        progress_ts = launch_ts + timedelta(minutes=module_duration_base * 0.5)
        events.append(_make_event("progressed", learner, course, mod_idx, progress_ts,
                                  duration_min=module_duration_base * 0.5))

        # 4. Quiz / scored event
        is_last_mod = mod_idx == num_modules - 1
        if hard_quiz_mod is not None and mod_idx == hard_quiz_mod:
            # Hard quiz: mean 0.52, sd 0.12
            score = max(0.0, min(1.0, random.gauss(0.52, 0.12)))
        else:
            score = max(0.0, min(1.0, random.gauss(0.78, 0.10)))

        quiz_ts = launch_ts + timedelta(minutes=module_duration_base * 0.85)
        events.append(_make_event("scored", learner, course, mod_idx, quiz_ts,
                                  score=score, duration_min=random.uniform(10, 25)))

        # 5. Module completion (probability-gated on score)
        module_done = score >= 0.60 or random.random() < 0.20
        if module_done:
            complete_ts = launch_ts + timedelta(minutes=module_duration_base)
            is_course_complete = is_last_mod and random.random() < base_completion_prob
            events.append(_make_event("completed", learner, course, mod_idx, complete_ts,
                                      score=score, duration_min=module_duration_base,
                                      completed=is_course_complete))
            if is_course_complete:
                completed_courses.add(course["course_id"])

        cur_ts = launch_ts + timedelta(minutes=module_duration_base + random.uniform(0, 60))


def generate_xapi_events(
    courses: list[dict],
    learners: list[dict],
    target_events: int = 50_000,
) -> list[dict]:
    events: list[dict] = []
    course_map = {c["course_id"]: c for c in courses}

    # Each learner enrolls in 2–8 courses
    enrollments: list[tuple[dict, dict, datetime]] = []
    for learner in learners:
        num_enrolled = random.randint(2, 8)
        chosen = random.sample(courses, min(num_enrolled, len(courses)))
        for course in chosen:
            enroll_ts = SIM_START + timedelta(days=random.randint(0, 300))
            enrollments.append((learner, course, enroll_ts))

    # Sort enrollments chronologically
    enrollments.sort(key=lambda x: x[2])

    completed_per_learner: dict[str, set[str]] = {l["email"]: set() for l in learners}

    for learner, course, enroll_ts in enrollments:
        _simulate_enrollment(
            learner, course, enroll_ts, events,
            completed_per_learner[learner["email"]]
        )

    # If we're short of target, add more enrollments from heavy users
    extra_attempts = 0
    while len(events) < target_events and extra_attempts < 200:
        learner = random.choice(learners)
        course = random.choice(courses)
        enroll_ts = SIM_START + timedelta(days=random.randint(0, 300))
        _simulate_enrollment(
            learner, course, enroll_ts, events,
            completed_per_learner[learner["email"]]
        )
        extra_attempts += 1

    # Update learner completed_courses list
    for learner in learners:
        learner["completed_courses"] = list(completed_per_learner[learner["email"]])

    # Sort events by timestamp
    events.sort(key=lambda e: e["timestamp"])
    return events


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _strip_private_keys(courses: list[dict]) -> list[dict]:
    """Remove keys starting with _ before serialising (internal generator metadata)."""
    cleaned = []
    for c in courses:
        cleaned.append({k: v for k, v in c.items() if not k.startswith("_")})
    return cleaned


def main() -> None:
    print("Generating 200 courses ...")
    courses = generate_courses(200)
    print(f"  OK {len(courses)} courses")

    print("Generating 500 learners ...")
    learners = generate_learners(500)
    print(f"  OK {len(learners)} learners")

    print("Simulating xAPI events (target >= 50 000) ...")
    events = generate_xapi_events(courses, learners, target_events=50_000)
    print(f"  OK {len(events):,} xAPI events")

    # Strip internal metadata before saving
    courses_clean = _strip_private_keys(courses)

    print("Writing data files ...")
    (DATA_DIR / "seed_courses.json").write_text(
        json.dumps(courses_clean, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (DATA_DIR / "seed_learners.json").write_text(
        json.dumps(learners, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (DATA_DIR / "seed_xapi_events.json").write_text(
        json.dumps(events, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nDone. Files written to data/")
    print(f"  seed_courses.json      {len(courses_clean):>6} records")
    print(f"  seed_learners.json     {len(learners):>6} records")
    print(f"  seed_xapi_events.json  {len(events):>6} records")

    # Quick pattern validation
    print("\n-- Pattern validation --")
    blended = [c for c in courses_clean if c["format"] == "blended"]
    self_paced = [c for c in courses_clean if c["format"] == "self-paced"]
    print(f"  Blended courses:    {len(blended)}")
    print(f"  Self-paced courses: {len(self_paced)}")

    eng_learners = [l for l in learners if l["department"] == "Engineering"]
    eng_skills = {}
    for l in eng_learners:
        for s in l["skills"]:
            eng_skills.setdefault(s["skill_name"], []).append(s["proficiency_level"])
    print("  Engineering avg proficiency:")
    for skill in ["Python", "Machine Learning", "SQL"]:
        vals = eng_skills.get(skill, [])
        if vals:
            print(f"    {skill:25s} {sum(vals)/len(vals):.2f} / 5.0")

    completed_events = [e for e in events if e["verb"]["display"]["en-US"] == "completed"]
    print(f"  Completion events:  {len(completed_events):,}")
    scored_events = [e for e in events if e["verb"]["display"]["en-US"] == "scored"]
    low_score = [e for e in scored_events if e.get("result", {}).get("score", {}).get("scaled", 1) < 0.60]
    print(f"  Low-score events (<60%): {len(low_score):,} / {len(scored_events):,}")


if __name__ == "__main__":
    main()
