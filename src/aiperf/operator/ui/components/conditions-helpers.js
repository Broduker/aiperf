// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const CONDITION_LABELS = {
  ConfigValid: 'Config',
  EndpointReachable: 'Endpoint',
  PreflightPassed: 'Preflight',
  ResourcesCreated: 'Resources',
  WorkersReady: 'Workers',
  BenchmarkRunning: 'Running',
  ResultsAvailable: 'Results',
};

export function conditionClass(condition) {
  const status = (condition.status ?? '').toLowerCase();
  const reason = (condition.reason ?? '').toLowerCase();
  const type = (condition.type ?? '').toLowerCase();

  if (type.includes('warning') || reason.includes('warning')) {
    return 'condition-badge--progress';
  }
  if (status === 'true') return 'condition-badge--true';
  if (reason.includes('progress') || reason.includes('waiting')) {
    return 'condition-badge--progress';
  }
  if (status === 'false') return 'condition-badge--false';
  return 'condition-badge--unknown';
}

export function conditionLabel(type) {
  if (CONDITION_LABELS[type]) return CONDITION_LABELS[type];
  if (type === 'PreflightHasWarnings') return 'Preflight warnings';
  return type
    .replace(/Has/g, ' ')
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .trim();
}

function conditionBadge(condition) {
  return {
    type: condition.type,
    message: condition.message,
    label: conditionLabel(condition.type),
    className: conditionClass(condition),
  };
}

function terminalConditionStatuses(conditions) {
  const statuses = new Map();

  for (const condition of conditions ?? []) {
    if (condition.type !== 'Complete' && condition.type !== 'Failed') continue;
    statuses.set(condition.type, (condition.status ?? '').toLowerCase());
  }

  return statuses;
}

function shouldHideCondition(condition, terminalStatuses) {
  const status = (condition.status ?? '').toLowerCase();
  if (status === 'true') return true;

  if (status !== 'false' || (condition.type !== 'Complete' && condition.type !== 'Failed')) {
    return false;
  }

  const counterpart = condition.type === 'Complete' ? 'Failed' : 'Complete';
  const counterpartStatus = terminalStatuses.get(counterpart);
  return counterpartStatus === 'true' || counterpartStatus === 'false';
}

export function visibleConditionBadges(conditions) {
  const terminalStatuses = terminalConditionStatuses(conditions);
  return (conditions ?? [])
    .filter(condition => !shouldHideCondition(condition, terminalStatuses))
    .map(conditionBadge);
}

export function visibleConditionBadgeSummary(conditions, limit) {
  const badges = [];
  let total = 0;
  const terminalStatuses = terminalConditionStatuses(conditions);

  for (const condition of conditions ?? []) {
    if (shouldHideCondition(condition, terminalStatuses)) continue;
    total += 1;
    if (badges.length < limit) badges.push(conditionBadge(condition));
  }

  return { badges, overflow: Math.max(0, total - badges.length) };
}
