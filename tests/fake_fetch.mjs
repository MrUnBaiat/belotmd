// Loaded with `node --import` ahead of the real bridge.js: replaces fetch with
// canned answers and logs every request, so the shipped bridge is tested end
// to end without a single request reaching belot.md.
//
//   FAKE_FETCH_ANSWERS  a JSON file of [needle, status, body] triples; the
//                       first needle found in "<url> <body>" gives the answer
//   FAKE_FETCH_LOG      each request is appended here, one JSON line apiece
import { appendFileSync, readFileSync } from "node:fs";

const answers = JSON.parse(readFileSync(process.env.FAKE_FETCH_ANSWERS, "utf-8"));
const log = process.env.FAKE_FETCH_LOG;

globalThis.fetch = async (url, init = {}) => {
    const body = init.body === undefined ? "" : String(init.body);
    appendFileSync(log, JSON.stringify({
        url: String(url),
        method: init.method || "GET",
        body,
        form: init.body instanceof URLSearchParams,
        headers: init.headers || {},
    }) + "\n");
    const hit = answers.find(([needle]) => `${url} ${body}`.includes(needle));
    const [, status, text] = hit || [null, 404, ""];
    return new Response(text, { status });
};
