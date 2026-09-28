---
name: frontend-design
description: Designing and building polished web UIs - layout, color, type, components, responsive HTML/CSS.
---
# Front-end design checklist

1. **Structure first**: semantic HTML (header/main/section/nav/button), one h1, logical heading order.
2. **Tokens**: define CSS custom properties on `:root` - 1 accent, 1-2 neutrals ramps, surface/background,
   text/text-muted, radius, spacing scale (4/8/12/16/24/32/48), and a type scale (e.g. 14/16/20/28/40).
   Add a `@media (prefers-color-scheme: dark)` block that redefines the same tokens.
3. **Layout**: mobile-first; CSS grid for page structure, flex for rows; max content width ~1100px;
   16px side gutters on phones; no horizontal scroll.
4. **Type**: one family (system-ui stack or one Google Font), line-height 1.5 body / 1.2 headings,
   60-75ch measure.
5. **Components**: consistent radius and shadows; buttons have hover, active and visible
   `:focus-visible` states; inputs have labels; touch targets >= 44px.
6. **Contrast**: body text >= 4.5:1, large text >= 3:1.
7. **Data UI** (dashboards/trackers): cards with a clear title + one number, charts with labelled
   axes, empty states that say what to do next.
8. **Images**: request generated imagery with subject, style, palette (use your hex tokens),
   aspect ratio and pixel size; reference the returned artifact path in the markup.

Deliver a single self-contained `index.html` (inline CSS/JS) unless asked otherwise.
