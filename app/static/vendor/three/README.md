# Three.js 0.187.0

These files are from the official `three@0.187.0` npm package. The MIT license is retained in `LICENSE`.

- `three.module.js` and `three.core.js` are unchanged upstream builds.
- `OrbitControls.js` is the upstream control module with its bare `three` import changed to `./three.module.js` so it resolves locally without an inline import map.

All three modules must be updated together. They are loaded from the application's own origin only when the spatial analysis view is used. No CDN or additional network permission is required.
