# LaunchDarkly Features: Behind the Scenes
Behind the scenes, LaunchDarkly is managing four distinct feature flags, each with its own set of rules to control feature rollout and user targeting. Make sure the client-sdk is enabled for all flags listed.

### Feature Flag Type 1: Segment-Based Targeting

- **Key:** `the-voice-feature`
- **Rule:** If the user context is part of the "Comic Characters" segment (see segment description), the flag is evaluated as true.

### Feature Flag Type 2: Segment-Based Targeting

- **Key:** `lightsaber-feature`
- **Rule:** If the user context is part of the "Comic Characters" segment (see segment description), the flag is evaluated as true.

### Feature Flag Type 3: Segment-Based Targeting

- **Key:** `kill-or-save-feature`
- **Rule:** If the user context is part of the "Comic Characters" segment (see segment description), the flag is evaluated as true.

### Feature Flag Type 4: Segment & User Targeting

- **Key:** `power-level-feature`
- **Rules:**
  1. If the user context is part of the "Comic Characters" segment (see segment description), the flag is evaluated as true.
  2. If the user key is `starlord@marvel.com`, the flag is evaluated as true (regardless of segment).

---

## Segment Definitions

### Comic Characters Segment

- **Description:** Includes all users whose `type` attribute is set to `comic_character`.
- **Example User Contexts:**
  - `hulk@marvel.com` (type: comic_character, universe: Marvel)
  - `joker@dc.com` (type: comic_character, universe: DC)
  - `starlord@marvel.com` (type: comic_character, universe: Marvel)

### Superheroes Segment

- **Description:** Includes all users whose `role` attribute contains `superhero`.
- **Example Users:**
  - `hulk@marvel.com` (role: superhero)
  - `superman@dc.com` (role: superhero)

### Villains Segment

- **Description:** Includes all users whose `role` attribute contains `villain`.
- **Example Users:**
  - `joker@dc.com` (role: villain)
  - `thanos@marvel.com` (role: villain)

---

## Metrics

### Kill Event Metric (`kill-event`)
- **Description:** This metric is recorded whenever a user clicks the "kill" button in the application UI.

### Save Event Metric (`save-event`)
- **Description:** This metric is recorded whenever a user clicks the "save" button in the application UI.

---

## Experimentation

### Background Experiment: "An Interesting Gift"
- **Description:**
  - The experiment "An Interesting Gift" is designed to measure the impact of the `lightsaber-feature` flag on user actions.
  - For 100% of user contexts, 80% are randomly assigned the `lightsaber-feature` variation of `false`, while the remaining 20% receive the variation `true`.
  - The presence of the lightsaber feature is hypothesized to influence user behavior.
- **Experiment Question:**
  - Given that the lightsaber is present, is the kill or save button more likely to be clicked?
- **How it works:**
  - Users are bucketed into the two variations using LaunchDarkly's experimentation platform.
  - The `kill-event` and `save-event` metrics are tracked for all users.
  - Results are analyzed to determine if the presence of the lightsaber feature increases the likelihood of either button being clicked.
- **Purpose:**
  - To understand the behavioral impact of the lightsaber feature and inform future product decisions.

Experiment results and analysis are available in the LaunchDarkly dashboard, allowing for data-driven feature rollouts and improvements.

---

## AI Config: "The Voice"

- **Config Name:** The Voice
- **Config Key:** `the-voice`
- **Model:** `chatgpt-4o-latest`
- **Variations:**
  1. **Jarvis Variation (Superhero Segment):**
     - System prompt instructs the AI to act as Jarvis, providing the user with thoughtful advice on whether to kill or save.
     - Targeted to users whose context is part of the "Superheroes" segment.
  2. **Doctor Doom Variation (Villain Segment):**
     - System prompt instructs the AI to act as Doctor Doom, prompting the user on whether to kill or save, likely with a more villainous perspective.
     - Targeted to users whose context is part of the "Villains" segment.

- **How it works:**
  - When a user interacts with the system, their context is evaluated against the segments.
  - The appropriate AI variation (Jarvis or Doctor Doom) is selected based on their segment membership.
  - The AI model responds according to the system prompt for that variation, enhancing the user experience with context-aware advice or persuasion.

This setup allows for dynamic, personalized AI interactions, leveraging LaunchDarkly's targeting and experimentation capabilities to deliver tailored experiences for different user groups.

