const assert = require('node:assert/strict');
const {averageLinkage, leaves, prepare, order} = require('../app/static/heatmap-clustering.js');
const tree = averageLinkage([[0], [2], [10]]);
assert.deepEqual(leaves(tree), [0, 1, 2]);
assert.equal(tree.children[0].distance, 2);
assert.equal(tree.distance, 9, 'average of cross-cluster pair distances, not centroid distance');
const weighted = averageLinkage([[0], [1], [3], [10]]);
assert(Math.abs(weighted.distance - 26 / 3) < 1e-12, 'cluster sizes weight the distance update');
assert.equal(averageLinkage([]), null);
assert.deepEqual(leaves(averageLinkage([[1, 2]])), [0]);
assert.deepEqual(leaves(averageLinkage([[0], [0], [0], [0]])), [0, 1, 2, 3], 'ties deterministic');
assert.throws(() => averageLinkage([[NaN]]));
assert.throws(() => averageLinkage([[1], [1, 2]]));
const matrix = [[1, null, 3], [2, 0, 6], [null, null, null]], copy = JSON.stringify(matrix);
const prepared = prepare(matrix);
assert.equal(prepared.missing, 4);
assert.equal(prepared.emptyGenes, 1);
assert.equal(JSON.stringify(matrix), copy, 'imputation must not change displayed values');
for (const mode of ['both', 'genes', 'profiles', 'none']) {
 const layout = order(prepared, mode, 3, 3);
 assert.deepEqual([...layout.rows].sort(), [0, 1, 2]);
 assert.deepEqual([...layout.columns].sort(), [0, 1, 2]);
 if (mode === 'profiles' || mode === 'none') assert.deepEqual(layout.rows, [0, 1, 2]);
 if (mode === 'genes' || mode === 'none') assert.deepEqual(layout.columns, [0, 1, 2]);
}
assert.throws(() => order(prepared, 'unknown', 3, 3));
assert.throws(() => prepare([[1], [1, 2]]));
assert.throws(() => prepare([]));
console.log('Heatmap clustering passed: average linkage, weighted merges, ties, singleton, invalid matrices, missing data, all four orders, original values preserved.');
