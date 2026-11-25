'use strict';
// Exploratory flags from the displayed expression scores, grouped by study.
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.defenderPcaOutliers = api;
})(typeof window === 'object' ? window : globalThis, function () {
  const cutoff = -2 * Math.log(0.05); // 95th percentile of chi-square with two degrees of freedom
  const cache = new WeakMap();

  function studyGuide(indices, scores, distances) {
    if (indices.length < 3) return null;
    let sumX = 0, sumY = 0;
    for (const i of indices) {
      const row = scores[i];
      if (!row || !Number.isFinite(row[0]) || !Number.isFinite(row[1])) return null;
      sumX += row[0];
      sumY += row[1];
    }
    const cx = sumX / indices.length, cy = sumY / indices.length;
    let xx = 0, xy = 0, yy = 0;
    for (const i of indices) {
      const dx = scores[i][0] - cx, dy = scores[i][1] - cy;
      xx += dx * dx;
      xy += dx * dy;
      yy += dy * dy;
    }
    const df = indices.length - 1;
    const sx = Math.sqrt(xx / df), varianceY = yy / df;
    if (!(sx > 0 && varianceY > 0)) return null;
    const slope = xy / df / sx;
    const residualVariance = varianceY - slope * slope;
    if (!(residualVariance > varianceY * 1e-12)) return null;
    const residualSD = Math.sqrt(residualVariance);

    // Use the same Cholesky factor to score points and draw their exact boundary.
    for (const i of indices) {
      const u = (scores[i][0] - cx) / sx;
      const v = (scores[i][1] - cy - slope * u) / residualSD;
      distances[i] = u * u + v * v;
    }
    return Array.from({length: 96}, (_, i) => {
      const angle = 2 * Math.PI * i / 96;
      const u = Math.sqrt(cutoff) * Math.cos(angle), v = Math.sqrt(cutoff) * Math.sin(angle);
      return [cx + sx * u, cy + slope * u + residualSD * v];
    });
  }

  function forResult(result) {
    if (cache.has(result)) return cache.get(result);
    const byStudy = new Map(), outlines = new Map();
    const distances = Array(result.samples.length).fill(null);
    result.samples.forEach(({study_id}, i) => {
      if (!study_id) return;
      if (!byStudy.has(study_id)) byStudy.set(study_id, []);
      byStudy.get(study_id).push(i);
    });
    for (const [study, indices] of byStudy) {
      const outline = studyGuide(indices, result.pca.scores, distances);
      if (outline) outlines.set(study, outline);
    }
    const flagged = [];
    distances.forEach((distance, i) => { if (distance > cutoff) flagged.push(i); });
    flagged.sort((a, b) => distances[b] - distances[a] || a - b);
    const guide = {outlines, distances, flagged};
    cache.set(result, guide);
    return guide;
  }

  function status(distance) {
    if (!Number.isFinite(distance)) return 'Unscored';
    return distance > cutoff ? 'Flagged' : 'Within guide';
  }
  return {forResult, cutoff, status};
});
