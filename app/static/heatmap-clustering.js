'use strict';
// Display-only hierarchical clustering. Original matrix values are never changed.
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.defenderHeatmap = api;
})(typeof window === 'object' ? window : globalThis, function () {
  function averageLinkage(vectors) {
    const n = vectors.length;
    if (!n) return null;
    const width = vectors[0].length;
    if (!vectors.every(v => v.length === width && v.every(Number.isFinite))) throw Error('Clustering requires finite, equally sized vectors.');
    const distances = Array.from({length: 2 * n - 1}, () => new Float64Array(2 * n - 1));
    const nodes = vectors.map((_, id) => ({id, size: 1, first: id, distance: 0}));
    const active = new Set(nodes.map(node => node.id));
    for (let i = 0; i < n; i++) for (let j = 0; j < i; j++) {
      distances[i][j] = distances[j][i] = Math.sqrt(vectors[i].reduce((sum, value, k) => sum + (value - vectors[j][k]) ** 2, 0));
    }
    while (active.size > 1) {
      const ids = [...active].sort((a, b) => a - b);
      let best = Infinity, a, b;
      for (let i = 0; i < ids.length; i++) for (let j = i + 1; j < ids.length; j++) {
        if (distances[ids[i]][ids[j]] < best) { best = distances[ids[i]][ids[j]]; a = ids[i]; b = ids[j]; }
      }
      const children = [nodes[a], nodes[b]].sort((x, y) => x.first - y.first);
      const node = {id: nodes.length, size: nodes[a].size + nodes[b].size, first: children[0].first, distance: best, children};
      for (const c of active) if (c !== a && c !== b) {
        distances[node.id][c] = distances[c][node.id] = (distances[a][c] * nodes[a].size + distances[b][c] * nodes[b].size) / node.size;
      }
      nodes.push(node); active.delete(a); active.delete(b); active.add(node.id);
    }
    return nodes[nodes.length - 1];
  }
  function leaves(tree) { return !tree ? [] : tree.children ? tree.children.flatMap(leaves) : [tree.id]; }
  function prepare(matrix) {
    if (!matrix.length || !matrix[0].length || !matrix.every(row => row.length === matrix[0].length)) throw Error('Heatmap matrix must be rectangular and nonempty.');
    let missing = 0, emptyGenes = 0;
    const finite = matrix.map(row => {
      const observed = row.filter(Number.isFinite).sort((a, b) => a - b), half = Math.floor(observed.length / 2);
      if (!observed.length) emptyGenes++;
      const median = observed.length ? (observed.length % 2 ? observed[half] : (observed[half - 1] + observed[half]) / 2) : 0;
      return row.map(value => { if (Number.isFinite(value)) return value; missing++; return median; });
    });
    return {genes: averageLinkage(finite), profiles: averageLinkage(finite[0].map((_, j) => finite.map(row => row[j]))), missing, emptyGenes};
  }
  function order(prepared, mode, rows, columns) {
    if (!['both', 'genes', 'profiles', 'none'].includes(mode)) throw Error('Unknown heatmap clustering mode.');
    return {rows: ['both', 'genes'].includes(mode) ? leaves(prepared.genes) : Array.from({length: rows}, (_, i) => i),
      columns: ['both', 'profiles'].includes(mode) ? leaves(prepared.profiles) : Array.from({length: columns}, (_, i) => i)};
  }
  return {averageLinkage, leaves, prepare, order};
});
