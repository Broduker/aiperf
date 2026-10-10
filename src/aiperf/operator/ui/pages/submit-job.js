// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { html } from 'htm/preact';
import { load } from 'js-yaml';
import { useRef, useState } from 'preact/hooks';
import {
  api,
  isTokenRequiredError,
  setSessionToken,
} from '../lib/api.js';
import { buildJobPath, navigate } from '../lib/router.js';
import { TokenModal } from '../components/token-modal.js';
import { YamlEditor } from '../components/yaml-editor.js';

const DEFAULT_JOB_YAML = `apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfJob
metadata:
  name: my-benchmark
  namespace: default
spec:
  schemaVersion: "2.0"
  image: aiperf:latest
  benchmark:
    model: meta-llama/Llama-3.1-8B-Instruct
    endpoint:
      url: http://localhost:8000
    dataset:
      type: synthetic
      entries: 1
      prompts:
        isl: 16
        osl: 8
    phases:
      type: concurrency
      concurrency: 1
      requests: 1
`;

function parseManifest(text) {
  let manifest;
  try {
    manifest = load(text);
  } catch (error) {
    return { error: `YAML parse error: ${error.message}` };
  }
  if (!manifest || typeof manifest !== 'object' || Array.isArray(manifest)) {
    return { error: 'Manifest must be a YAML object, not a scalar or list.' };
  }
  return { manifest };
}

function validationMessage(validation) {
  if (!validation) return null;
  if (validation.error) return validation.error;
  if (validation.passed) return 'Manifest validation passed.';
  return 'Manifest validation failed.';
}

export function SubmitJob() {
  const [yaml, setYaml] = useState(DEFAULT_JOB_YAML);
  const [validation, setValidation] = useState(null);
  const [validating, setValidating] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState(null);
  const [showTokenModal, setShowTokenModal] = useState(false);
  const pendingManifest = useRef(null);

  async function validateObject(manifest) {
    setValidation(null);
    setSubmitError(null);
    setValidating(true);
    try {
      const result = await api.validateManifest(manifest);
      setValidation(result);
      return result;
    } catch (error) {
      const result = { error: `Validation request failed: ${error.message}` };
      setValidation(result);
      return result;
    } finally {
      setValidating(false);
    }
  }

  async function handleValidate() {
    const parsed = parseManifest(yaml);
    if (parsed.error) {
      setValidation(parsed);
      return;
    }
    await validateObject(parsed.manifest);
  }

  async function submitManifest(manifest) {
    setSubmitError(null);
    setSubmitting(true);
    try {
      const created = await api.createJob(manifest);
      navigate(buildJobPath(created));
    } catch (error) {
      if (isTokenRequiredError(error)) {
        pendingManifest.current = manifest;
        setShowTokenModal(true);
        return;
      }
      setSubmitError(error.message ?? String(error));
    } finally {
      setSubmitting(false);
    }
  }

  async function handleSubmit(event) {
    event.preventDefault();
    const parsed = parseManifest(yaml);
    if (parsed.error) {
      setValidation(parsed);
      return;
    }
    const result = await validateObject(parsed.manifest);
    if (!result.passed) return;
    await submitManifest(parsed.manifest);
  }

  async function handleTokenConfirm(token) {
    setSessionToken(token);
    setShowTokenModal(false);
    const manifest = pendingManifest.current;
    pendingManifest.current = null;
    if (manifest) await submitManifest(manifest);
  }

  function handleEditorKeydown(event) {
    if (event.key !== 'Tab') return;
    event.preventDefault();
    const editor = event.currentTarget;
    const start = editor.selectionStart;
    const end = editor.selectionEnd;
    const next = `${yaml.slice(0, start)}  ${yaml.slice(end)}`;
    setYaml(next);
    requestAnimationFrame(() => {
      editor.selectionStart = start + 2;
      editor.selectionEnd = start + 2;
    });
  }

  const message = validationMessage(validation);
  const canSubmit = !validating && !submitting;

  return html`
    <div class="submit-job-page" data-testid="page-submit-job">
      <div class="page-heading">
        <div>
          <div class="eyebrow">OPERATE</div>
          <h1>Submit benchmark</h1>
          <p class="text-dim">Create an AIPerfJob from a Kubernetes manifest.</p>
        </div>
        <button type="button" class="btn btn--secondary" onclick=${() => navigate('/jobs')}>
          View jobs
        </button>
      </div>

      <div class="submit-job-notice">
        Edit the complete manifest below. Validate before submitting. Do not put API keys or other credentials in the manifest.
      </div>

      <form onsubmit=${handleSubmit}>
        <div class="submit-job-layout">
          <section class="submit-job-editor-panel">
            <div class="submit-job-panel-heading">
              <div>
                <h2>Job manifest</h2>
                <p class="text-dim">The same <code>AIPerfJob</code> shape accepted by <code>kubectl apply</code>.</p>
              </div>
              <span class="submit-job-format">YAML</span>
            </div>
            <${YamlEditor}
              value=${yaml}
              onInput=${(event) => { setYaml(event.currentTarget.value); setValidation(null); setSubmitError(null); }}
              onKeydown=${handleEditorKeydown}
              testid="submit-job-yaml"
            />
          </section>

          <aside class="submit-job-side-panel">
            <div class="submit-job-actions">
              <button type="button" class="btn btn--secondary" onclick=${handleValidate} disabled=${!canSubmit}>
                ${validating ? 'Validating…' : 'Validate manifest'}
              </button>
              <button type="submit" class="btn btn--primary" disabled=${!canSubmit}>
                ${submitting ? 'Submitting…' : 'Submit benchmark'}
              </button>
            </div>

            ${message && html`
              <div class=${'submit-job-result ' + (validation?.passed ? 'submit-job-result--success' : 'submit-job-result--error')} role="status">
                <strong>${message}</strong>
                ${validation?.errors?.length > 0 && html`
                  <ul>${validation.errors.map((error) => html`<li>${error}</li>`)}</ul>
                `}
                ${validation?.warnings?.length > 0 && html`
                  <div class="submit-job-warnings">
                    <strong>Warnings</strong>
                    <ul>${validation.warnings.map((warning) => html`<li>${warning}</li>`)}</ul>
                  </div>
                `}
              </div>
            `}

            ${submitError && html`
              <div class="submit-job-result submit-job-result--error" role="alert">
                <strong>Submission failed</strong>
                <div>${submitError}</div>
              </div>
            `}

            <div class="submit-job-help">
              <h3>Required pieces</h3>
              <ul>
                <li><code>metadata.name</code> and an optional target namespace</li>
                <li><code>spec.image</code> for the benchmark container</li>
                <li><code>spec.benchmark</code> with model, endpoint, dataset, and phases</li>
              </ul>
              <p class="text-dim">Submission requires the operator mutating-route token and Kubernetes RBAC permission in the target namespace.</p>
            </div>
          </aside>
        </div>
      </form>

      ${showTokenModal && html`
        <${TokenModal}
          onConfirm=${handleTokenConfirm}
          onCancel=${() => { pendingManifest.current = null; setShowTokenModal(false); }}
        />
      `}
    </div>
  `;
}
