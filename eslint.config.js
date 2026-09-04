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
            globals: {
                console: "readonly",
                fetch: "readonly",
                setInterval: "readonly",
                clearInterval: "readonly",
                URLSearchParams: "readonly",
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
