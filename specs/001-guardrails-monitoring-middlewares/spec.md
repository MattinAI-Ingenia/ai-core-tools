# Feature Specification: Guardrails Middleware & Configurable Monitoring Metrics

**Feature Branch**: `001-guardrails-monitoring-middlewares`

**Created**: 2026-06-02

**Status**: Draft

**Input**: User description: "Quiero implementar dos nuevos 'middlewares', implementado dentro de la pestaña de middlewares e integrado como los midlewares que están constituidos actualmente. - Guardrails. Un middleware que integre la funcion de 'guardrail' para crear: Input guardrails (mecanismos de control con malicious prompts, jailbreak the AI). Output guardrails: Preventing leakage of PII, block toxic and biased language, and ensuring the AI answers stick to predefined business facts/logic. Quiero que la interfaz tenga la siguiente forma: una parte de input y otra de output guardrails en dos apartados separados con tickboxes de las cosas que se deberían incluir en cada guardrail (todas ellas tickeadas como aplicadas por default). Quiero también que se cree un 'custom prompt' apartado, que tenga un mensaje pre-definido que funcione, pero en el cual el usuario pueda detallar más input/output guardrails. - Monitorization. Ahora mismo ya está implementado el Monitorization pero quiero que haya tickboxes que definan qué métricas son las que se imprimen/guardan. Eso en la interfaz de la customización y edición del middleware."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Create a Guardrails middleware with default protections (Priority: P1)

As an app editor, I open the Middlewares tab, choose to create a new middleware, and select the new "Guardrails" type. The configuration form presents two clearly separated sections — **Input Guardrails** and **Output Guardrails** — each containing a list of protections shown as checkboxes that are **all enabled by default**. I save the middleware with the defaults and attach it to an agent. From then on, the agent rejects or sanitizes malicious/jailbreak inputs and prevents unsafe outputs (PII leakage, toxic/biased language, off-topic answers that contradict defined business facts).

**Why this priority**: This is the core of the request and the minimum viable slice — a working Guardrails middleware with safe defaults delivers immediate protection value without any further configuration.

**Independent Test**: Create a Guardrails middleware accepting all defaults, attach it to an agent, then send (a) a known jailbreak/prompt-injection message and (b) a prompt designed to elicit PII or toxic content. Verify the agent blocks/sanitizes both while normal questions still get normal answers.

**Acceptance Scenarios**:

1. **Given** I am an editor on the Middlewares tab, **When** I create a new middleware and select the "Guardrails" type, **Then** the form shows separate "Input Guardrails" and "Output Guardrails" sections, each with its protections listed as checkboxes that are all checked by default.
2. **Given** a Guardrails middleware saved with default protections is attached to an agent, **When** a user sends a prompt-injection/jailbreak message, **Then** the input guardrail prevents the agent from following the malicious instruction and the user receives a safe refusal/handling response.
3. **Given** a Guardrails middleware with the "prevent PII leakage" output protection enabled is attached to an agent, **When** the agent would otherwise emit PII in its answer, **Then** the output is sanitized or blocked before the user sees it.
4. **Given** I am editing an existing Guardrails middleware, **When** I uncheck one protection (e.g. "block toxic/biased language") and save, **Then** that protection is no longer applied while the remaining checked protections still are.

---

### User Story 2 - Extend guardrails with a custom prompt (Priority: P2)

As an app editor configuring a Guardrails middleware, I find a **Custom Prompt** section that already contains a sensible, working pre-defined instruction. I can edit or append to this text to describe additional input/output rules in natural language (e.g. "never discuss competitor pricing", "always answer only about our 2026 product catalog"). After saving, these custom rules are applied in addition to the checkbox-based protections.

**Why this priority**: Adds flexibility beyond the fixed checkbox protections, letting each app encode its own business-specific guardrails. It builds on US1 but is not required for the MVP to deliver value.

**Independent Test**: Open a Guardrails middleware, confirm the Custom Prompt field is pre-populated with working default text, add a custom rule, save, attach to an agent, and verify the agent honors the custom rule while still applying the default protections.

**Acceptance Scenarios**:

1. **Given** I create a new Guardrails middleware, **When** the form loads, **Then** the Custom Prompt section is pre-filled with a default instruction that is itself a valid, functional guardrail prompt.
2. **Given** I add a custom business rule to the Custom Prompt and save, **When** a user asks something that violates that rule, **Then** the agent declines or redirects according to the custom rule.
3. **Given** I clear or modify the Custom Prompt, **When** I save, **Then** the change persists and is reflected the next time I edit the middleware.

---

### User Story 3 - Choose which monitoring metrics are recorded (Priority: P2)

As an app editor, I edit an existing Monitoring middleware (or create a new one) and see a set of **checkboxes that select which metrics are printed/stored** (e.g. input tokens, output tokens, total tokens, model names, number of LLM calls). I enable only the metrics I care about, save, and from then on only those metrics are emitted for agents using that middleware.

**Why this priority**: Monitoring already exists, so this is an enhancement rather than new capability. It improves signal/noise and storage control, but the system remains usable without it.

**Independent Test**: Edit a Monitoring middleware, deselect some metrics, save, run an agent turn, and verify that only the selected metrics appear in the recorded/printed monitoring output.

**Acceptance Scenarios**:

1. **Given** I open the configuration of a Monitoring middleware, **When** the form loads, **Then** I see a checkbox per available metric with all metrics enabled by default (preserving current behavior).
2. **Given** I uncheck a metric (e.g. "output tokens") and save, **When** an agent using this middleware completes a turn, **Then** the recorded/printed monitoring data omits the unchecked metric and includes the checked ones.
3. **Given** an existing Monitoring middleware created before this feature, **When** I open it, **Then** it behaves as if all metrics are selected (no loss of existing behavior).

---

### Edge Cases

- **All protections unchecked**: If an editor unchecks every input and output protection and leaves the custom prompt empty, the Guardrails middleware should save but effectively apply no restrictions; the UI should make this "no protection" state evident.
- **All monitoring metrics unchecked**: Saving a Monitoring middleware with no metrics selected results in nothing being recorded/printed; the UI should make clear that monitoring will produce no output in this state.
- **Guardrail blocks a legitimate request (false positive)**: A safe user prompt is incorrectly blocked. The user should receive an understandable response rather than a silent failure.
- **Conflicting custom prompt vs. checkbox protections**: A custom rule that loosens a protection that is also enabled via checkbox — the safer (more restrictive) behavior should win.
- **Overlap with the existing PII middleware**: An agent has both the standalone PII middleware and a Guardrails middleware with PII protection enabled — both applying must not corrupt the response or cause errors.
- **Existing middlewares unaffected**: Adding the new Guardrails type and monitoring metric selection must not change behavior of already-configured middlewares of other types.

## Requirements *(mandatory)*

### Functional Requirements

#### Guardrails middleware

- **FR-001**: The system MUST offer a new selectable middleware type, "Guardrails", within the existing Middlewares tab, created and managed through the same flows as current middleware types (create, edit, delete, attach to agents).
- **FR-002**: The Guardrails configuration interface MUST present two visually separated sections: "Input Guardrails" and "Output Guardrails".
- **FR-003**: The Input Guardrails section MUST include protections against malicious prompts and jailbreak/prompt-injection attempts, presented as checkboxes.
- **FR-004**: The Output Guardrails section MUST include protections for: preventing leakage of personally identifiable information (PII), blocking toxic and biased language, and ensuring answers stay aligned with predefined business facts/logic, presented as checkboxes.
- **FR-005**: Every guardrail protection checkbox MUST be enabled (checked) by default when creating a new Guardrails middleware.
- **FR-006**: Editors MUST be able to individually enable/disable each protection, and only enabled protections are applied to agents using the middleware.
- **FR-007**: The Guardrails configuration MUST include a "Custom Prompt" section pre-populated with a default instruction that functions as a valid guardrail on its own.
- **FR-008**: Editors MUST be able to edit the Custom Prompt to add or refine additional input/output guardrail rules in natural language, and saved custom rules MUST be applied in addition to the checkbox-selected protections.
- **FR-009**: When attached to an agent, enabled input guardrails MUST be applied to user input before the agent acts on it, and enabled output guardrails MUST be applied to the agent's response before it reaches the user.
- **FR-010**: The Guardrails middleware configuration (selected protections and custom prompt) MUST persist and be correctly restored when the middleware is reopened for editing.

#### Configurable monitoring metrics

- **FR-011**: The configuration/edit interface of the Monitoring middleware MUST present checkboxes that let editors select which metrics are printed/stored.
- **FR-012**: The selectable metrics MUST cover the monitoring data the system currently produces (at minimum: input tokens, output tokens, total tokens, model names, and number of LLM calls).
- **FR-013**: Only metrics selected by the editor MUST be printed/stored for agents using that Monitoring middleware.
- **FR-014**: For new Monitoring middlewares, all metrics MUST be selected by default, and existing Monitoring middlewares MUST behave as if all metrics are selected so current behavior is preserved.
- **FR-015**: The monitoring metric selection MUST persist and be correctly restored when the Monitoring middleware is reopened for editing.

#### Integration & consistency

- **FR-016**: Both new/changed middleware behaviors MUST integrate with the existing middleware-to-agent attachment mechanism so editors manage them exactly like current middlewares.
- **FR-017**: Introducing the Guardrails type and the monitoring metric selection MUST NOT alter the behavior of existing middlewares of other types.

### Key Entities *(include if feature involves data)*

- **Middleware (Guardrails type)**: A new variant of the existing Middleware entity scoped to an App. Its configuration captures, per protection, whether it is enabled for input and output guardrails, plus a custom guardrail prompt string. Attached to agents via the existing agent–middleware association.
- **Guardrail protection**: A named, individually toggleable safeguard belonging to either the input or output group (e.g. "block jailbreak/prompt injection", "prevent PII leakage", "block toxic/biased language", "stay within business facts").
- **Monitoring metric selection**: Part of the existing Monitoring middleware's configuration; a set of flags indicating which metrics (input tokens, output tokens, total tokens, models, LLM call count) are printed/stored.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: An editor can create a fully-protecting Guardrails middleware (all defaults) and attach it to an agent in under 2 minutes without any free-text configuration.
- **SC-002**: With default Guardrails enabled, at least 90% of a representative set of known jailbreak/prompt-injection test prompts fail to make the agent break its intended behavior.
- **SC-003**: With default Guardrails enabled, responses that would contain PII or toxic/biased language are sanitized or blocked in at least 95% of a representative test set, while legitimate prompts continue to receive normal answers (false-positive block rate under 5%).
- **SC-004**: Custom Prompt rules added by an editor are honored by the agent in 100% of direct test cases targeting those rules.
- **SC-005**: After deselecting specific monitoring metrics, the printed/stored monitoring output contains exactly the selected metrics and none of the deselected ones across 100% of test turns.
- **SC-006**: Existing middlewares (all current types and pre-existing Monitoring middlewares) exhibit no behavioral change after this feature is deployed.

## Assumptions

- The Guardrails middleware is implemented as a new middleware type within the existing middleware framework, reusing the current create/edit/attach UI patterns and persistence model rather than introducing a separate subsystem.
- The fixed set of guardrail protections for v1 is: Input — malicious prompt detection and jailbreak/prompt-injection prevention; Output — PII leakage prevention, toxic/biased language blocking, and adherence to predefined business facts/logic. Additional protections beyond these are expressed through the Custom Prompt.
- Guardrail enforcement is primarily prompt/instruction-based (a working default custom prompt plus per-protection rules), consistent with how the platform currently steers agent behavior; specialized external moderation services are out of scope for v1.
- The "predefined business facts/logic" that output answers must adhere to are supplied by the editor through the Custom Prompt and/or the agent's existing configuration (system prompt, knowledge base), not by a new separate data store.
- The metrics offered for the Monitoring middleware are those the platform already produces today (input/output/total tokens, model names, LLM call count); no new metric types are introduced by this feature.
- "Printed/stored" monitoring refers to the platform's existing monitoring output channel; this feature controls which metrics are emitted, not where they are emitted.
- When a guardrail blocks a request, the user receives a safe, understandable response; the exact wording is a default the editor can influence via the Custom Prompt.
- Behavior is reused across all clients of the platform since middleware configuration is backend-driven and surfaced through the shared frontend.
