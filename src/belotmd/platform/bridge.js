import { Client } from 'colyseus.js';
import WebSocket, { WebSocketServer } from 'ws';

// AUDIT: the port was hardcoded at 8765, which made two accounts on one
// machine unsafe: the second daemon could not bind, and its Python client then
// connected to the FIRST one's bridge -- receiving another account's STATE
// frames, hand included. Port and token now come from argv, one bridge per
// client, and the token lets the client prove it reached its OWN daemon.
const BRIDGE_PORT = Number(process.argv[2]) || 8765;
const BRIDGE_TOKEN = process.argv[3] || "";

// Every HTTP call gets a deadline. Without one a stalled request hangs the
// bridge forever with no error and no log line -- the bot simply stops, which
// is indistinguishable from a quiet lobby. A timeout surfaces as an ERROR,
// which Python already knows how to recover from.
const HTTP_TIMEOUT_MS = 20_000;
const USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36";

// AUDIT: `activeRoom` / `pingInterval` were MODULE globals, so (a) a second
// Python client silently hijacked the first one's room and (b) neither was
// cleared on leave, so post-leave SENDs vanished into a dead room object.
const wss = new WebSocketServer({ port: BRIDGE_PORT });

// Without this a bind failure (port taken) is an unhandled 'error' event: the
// daemon dies with a stack trace that says nothing about the cause, and the
// Python side reports it as a timeout.
wss.on('error', (err) => {
    console.error(`[Bridge] Cannot listen on 127.0.0.1:${BRIDGE_PORT}: ${err.message}`);
    process.exit(1);
});
console.log(`[Bridge] Daemon listening on ws://127.0.0.1:${BRIDGE_PORT}`);

// One Python agent per daemon. Sharing a bridge is exactly the accident this
// file now exists to prevent, so a second attach is refused rather than served.
let attached = false;

wss.on('connection', (pySocket) => {
    if (attached) {
        console.warn("[Bridge] A second Python agent tried to attach -- refusing.");
        try {
            pySocket.send(JSON.stringify({ event: "ERROR",
                                           message: "this bridge already has a Python agent" }));
        } catch (_) {}
        pySocket.close();
        return;
    }
    attached = true;
    console.log("[Bridge] Python Agent connected.");

    // `lastTableId` lets Python say "anywhere but there" after a kick:
    // the lobby pick below takes the FIRST open table, and the table that
    // just ejected us has a free seat again, so it is a prime candidate to
    // be chosen straight back.
    const session = { room: null, ping: null, closed: false,
                      lastTableId: null };

    // AUDIT: every send was unguarded. onStateChange/onMessage/onError fire
    // from Colyseus internals, so a throw became an unhandled rejection while
    // the room stayed joined.
    const say = (obj) => {
        if (session.closed || pySocket.readyState !== WebSocket.OPEN) return;
        try { pySocket.send(JSON.stringify(obj)); }
        catch (err) { console.error("[Bridge] send failed:", err.message); }
    };

    // Always the first frame: the client checks this token before it sends a
    // single cookie, so it can tell its own daemon from one that happens to be
    // listening on the same port.
    say({ event: "HELLO", token: BRIDGE_TOKEN });

    const dropRoom = () => {
        if (session.ping) { clearInterval(session.ping); session.ping = null; }
        if (session.room) { try { session.room.leave(); } catch (_) {} session.room = null; }
    };

    pySocket.on('message', async (rawMsg) => {
        try {
            const cmd = JSON.parse(rawMsg);
            if (cmd.action === "CONNECT") {
                // AUDIT: a second CONNECT leaked the previous ping timer.
                if (session.room || session.ping) {
                    console.warn("[Bridge] CONNECT while a session is open — tearing the old one down first.");
                    dropRoom();
                }
                await connectToGame(cmd.cookies, say, session,
                                    cmd.avoidLast === true, cmd.table || null);
            } else if (cmd.action === "LOBBY") {
                // WHICH table is ours is decided in Python, where the rule is
                // unit-tested and where the partner's name is configured. The
                // bridge only fetches the list.
                say({ event: "LOBBY", mese: await fetchLobby(cmd.cookies) });
            } else if (cmd.action === "SEND") {
                if (!session.room) { say({ event: "ERROR", message: "SEND with no active room" }); return; }
                session.room.send(cmd.type, cmd.payload);
            } else if (cmd.action === "LEAVE") {
                dropRoom();
            }
        } catch (err) {
            console.error("[Bridge Error]:", err.message);
            say({ event: "ERROR", message: err.message });
        }
    });

    const cleanup = (why) => {
        attached = false;          // a restarted client may attach again
        if (session.closed) return;
        session.closed = true;
        console.log(`[Bridge] Cleaning up session (${why}).`);
        dropRoom();
    };
    pySocket.on('close', () => cleanup("python agent disconnected"));
    pySocket.on('error', (e) => cleanup(`python socket error: ${e.message}`));
});

async function connectToGame(cookieString, say, session, avoidLast = false,
                             table = null) {
    const headers = buildHeaders(cookieString);

    // AUDIT: a `global.WebSocket = CustomWebSocket` used to sit here, carrying
    // this connection's cookie into every later Colyseus connect -- a
    // cross-account hazard with nothing to gain. It never did anything either:
    // colyseus.js binds `globalThis.WebSocket || NodeWebSocket` ONCE at import
    // (transport/WebSocketTransport.mjs:4), long before this runs. The room
    // handshake authenticates with `playerToken`, not with cookies; the cookies
    // below are for the PHP calls.

    try {
        console.log("[Bridge] Checking for an ongoing active game...");
        let { wsUrl, roomId, playerToken, playerId } = await readTokens(headers);

        if (wsUrl && roomId) {
            // This branch is also how the host gets back to the table it
            // created: belot.md refuses to create a table for an account that
            // is already seated, so rejoining must be tried FIRST.
            console.log(`[Bridge] Rejoining active game room: ${roomId}`);
        } else if (table && table.mode === "create") {
            console.log("[Bridge] Creating a table...");
            // The site answers 302 -> gameplay_new.php and seats the creator,
            // so we FOLLOW the redirect and land on the page we need anyway.
            //
            // AUDIT: this used `redirect: "manual"`. That returns an opaque
            // response whose body is never read, and the next request on the
            // same pooled connection then stalls -- no error, no timeout, no
            // log line. The first live run hung here for minutes, having
            // created nothing.
            const createRes = await fetch("https://belot.md/gameTables.php", {
                method: "POST", headers, body: table.body, redirect: "follow",
                signal: AbortSignal.timeout(HTTP_TIMEOUT_MS) });
            const html = await createRes.text();      // always drain the body
            if (createRes.status >= 400) {
                throw new Error(`Creating a table failed (HTTP ${createRes.status}).`);
            }
            ({ wsUrl, roomId, playerToken, playerId } = scrapeTokens(html));
            if (!wsUrl || !roomId) {
                // Landed somewhere other than the gameplay page; ask for it.
                ({ wsUrl, roomId, playerToken, playerId } = await readTokens(headers));
            }
            if (!wsUrl || !roomId) throw new Error("Created a table, but its tokens did not parse.");
            console.log(`[Bridge] Created a table; seated in room ${roomId}.`);
        } else if (table && table.mode === "join") {
            console.log(`[Bridge] Joining table ${table.tableId}...`);
            session.lastTableId = table.tableId;
            await reserveSeat(headers, table.tableId, table.password || "");
            ({ wsUrl, roomId, playerToken, playerId } = await readTokens(headers));
            if (!wsUrl || !roomId) throw new Error("Failed to parse Colyseus tokens from gameplay page.");
        } else {
            console.log("[Bridge] No active game. Searching public tables...");
            const lobbyParams = new URLSearchParams({ getMeseNew: "1", nrJucatori: "4", minStatus: "0" });
            const lobbyRes = await fetch("https://belot.md/gameTables.php", { method: "POST", headers, body: lobbyParams });
            const lobbyData = await lobbyRes.json();
            
            const avoid = avoidLast ? session.lastTableId : null;
            const isOpen = (t) => t.full === 0 && t.pass === false && t.minStatus === undefined;
            let openTable = lobbyData.mese?.find(t => isOpen(t) && String(t.id) !== String(avoid));
            if (!openTable && avoid) {
                // The only thing on offer is the table we were just thrown
                // out of. Report it as nothing available rather than walking
                // back in to be kicked again; Python will wait and re-ask.
                console.log(`[Bridge] Only table ${avoid} is open, and we were just removed from it.`);
                throw new Error("No public open tables available.");
            }
            if (!openTable) throw new Error("No public open tables available.");

            session.lastTableId = openTable.id;
            await reserveSeat(headers, openTable.id, "");
            ({ wsUrl, roomId, playerToken, playerId } = await readTokens(headers));
            if (!wsUrl || !roomId) throw new Error("Failed to parse Colyseus tokens from gameplay page.");
        }

        console.log(`[Bridge] Connecting Colyseus client to ${wsUrl}...`);
        const client = new Client(wsUrl);
        const room = await client.joinById(roomId, { token: playerToken, playerId: playerId });

        // AUDIT: if Python vanished during the join handshake we would sit in
        // the room forever with nobody listening.
        if (session.closed) { try { room.leave(); } catch (_) {} return; }
        session.room = room;

        session.ping = setInterval(() => {
            fetch("https://belot.md/api/game.php?ping",
                  { headers, signal: AbortSignal.timeout(HTTP_TIMEOUT_MS) })
                .then(r => r.text())          // drain it; an unread body stalls the pool
                .catch(() => {});
        }, 30_000);

        room.onStateChange((state) => say({ event: "STATE", data: state, myPlayerId: playerId }));
        room.onMessage("*", (type, message) => say({ event: "MESSAGE", type, data: message }));
        room.onError((code, message) => say({ event: "ERROR", code, message }));
        room.onLeave((code) => {
            say({ event: "LEAVE", code });
            if (session.ping) { clearInterval(session.ping); session.ping = null; }
            session.room = null;
        });

        say({ event: "CONNECTED", roomId, playerId });

    } catch (err) {
        // AUDIT: on a partial failure the room may already be joined.
        if (session.ping) { clearInterval(session.ping); session.ping = null; }
        if (session.room) { try { session.room.leave(); } catch (_) {} session.room = null; }
        // `code` lets Python tell "our partner's table is not listed yet"
        // (seconds away) from "the lobby is empty" (a five-minute wait).
        say({ event: "ERROR", code: err.code, message: err.message });
    }
}

function buildHeaders(cookieString) {
    return {
        'Cookie': cookieString,
        'User-Agent': USER_AGENT,
        'Accept': 'application/json, text/plain, */*',
        'Referer': 'https://belot.md/'
    };
}

async function fetchLobby(cookieString) {
    const params = new URLSearchParams({ getMeseNew: "1", nrJucatori: "4", minStatus: "0" });
    const res = await fetch("https://belot.md/gameTables.php",
                            { method: "POST", headers: buildHeaders(cookieString), body: params,
                              signal: AbortSignal.timeout(HTTP_TIMEOUT_MS) });
    const data = await res.json();
    return data.mese || [];
}

// The four values every join needs, scraped from `window.customData` on the
// gameplay page. Reaching this page is also what proves a create or a seat
// reservation actually worked.
async function readTokens(headers) {
    const res = await fetch("https://belot.md/gameplay_new.php",
                            { headers, signal: AbortSignal.timeout(HTTP_TIMEOUT_MS) });
    return scrapeTokens(await res.text());
}

function scrapeTokens(html) {
    return {
        wsUrl: html.match(/wsUrl:\s*"([^"]+)"/)?.[1],
        roomId: html.match(/roomId:\s*"([^"]+)"/)?.[1],
        playerToken: html.match(/playerToken:\s*"([^"]+)"/)?.[1],
        playerId: html.match(/playerId:\s*"([^"]+)"/)?.[1],
    };
}

async function reserveSeat(headers, gameId, password) {
    console.log(`[Bridge] Reserving seat at table ${gameId}...`);
    const params = new URLSearchParams({ enterGame: "4", gameId, password: password || "" });
    const res = await fetch("https://belot.md/gameTables.php",
                            { method: "POST", headers, body: params,
                              signal: AbortSignal.timeout(HTTP_TIMEOUT_MS) });
    const data = await res.json();
    if (data.success !== "gameplay_new.php") {
        const err = new Error(`Seat reservation failed at table ${gameId}.`);
        err.code = "TABLE_NOT_FOUND";      // gone, full, or never existed
        throw err;
    }
}