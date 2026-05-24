import { css } from "lit";

/** Form-control styling shared by every demo component (inputs, labels, code). */
export const controlStyles = css`
  label {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
    font-size: var(--font-size-xs);
    font-weight: 600;
    color: var(--color-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }
  input,
  textarea,
  select {
    font-family: var(--font-mono);
    font-size: var(--font-size-sm);
    color: var(--color-text);
    background: var(--color-bg);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    padding: var(--space-2) var(--space-3);
    width: 100%;
    box-sizing: border-box;
  }
  input:focus,
  textarea:focus,
  select:focus {
    outline: 2px solid var(--color-accent);
    outline-offset: -1px;
  }
  textarea {
    resize: vertical;
    line-height: 1.5;
  }
  .row {
    display: flex;
    gap: var(--space-3);
    flex-wrap: wrap;
  }
  .row > * {
    flex: 1;
    min-width: 90px;
  }
  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
  }
  .chip {
    appearance: none;
    cursor: pointer;
    font-family: inherit;
    font-size: var(--font-size-xs);
    color: var(--color-text);
    background: var(--color-code-bg);
    border: 1px solid var(--color-border);
    border-radius: 999px;
    padding: var(--space-1) var(--space-3);
  }
  .chip:hover {
    border-color: var(--color-accent);
  }
  code {
    font-family: var(--font-mono);
    font-size: var(--font-size-xs);
    background: var(--color-code-bg);
    padding: 1px 5px;
    border-radius: var(--radius-sm);
  }
`;
