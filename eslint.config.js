// Minimal lint config: the point is `no-undef`.
//
// `node --check` validates syntax only. It happily accepts a name that is
// used but never declared -- which is exactly how a call site gained a
// fourth argument while the function kept three parameters, turning every
// table search into "avoidLast is not defined" and a permanent retry loop.
// Syntax was fine. Nothing caught it until it ran.
export default [
    {
        files: ["src/belotmd/platform/*.js"],
        languageOptions: {
            ecmaVersion: 2022,
            sourceType: "module",
            // Listed by hand rather than taken from a `globals` preset, so that
            // a misspelled global is still an error. The cost is that using a
            // new platform API in bridge.js means adding it here as well --
            // `npm run lint` names the ones that are missing.
            globals: {
                console: "readonly",
                fetch: "readonly",
                AbortSignal: "readonly",      // fetch deadlines: AbortSignal.timeout
                setInterval: "readonly",
                clearInterval: "readonly",
                setTimeout: "readonly",
                clearTimeout: "readonly",
                URLSearchParams: "readonly",
                URL: "readonly",
                process: "readonly",
                global: "readonly",
                globalThis: "readonly",
            },
        },
        rules: {
            "no-undef": "error",
            // `catch (_) {}` is deliberate here: several teardown paths must
            // not care why a leave() failed.
            "no-unused-vars": ["warn", { args: "none", caughtErrors: "none" }],
        },
    },
];
