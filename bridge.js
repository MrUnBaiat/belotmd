import { Client } from 'colyseus.js';
import WebSocket, { WebSocketServer } from 'ws';

const BRIDGE_PORT = 8765;
const USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36";

// AUDIT: `activeRoom` / `pingInterval` were MODULE globals, so (a) a second
// Python client silently hijacked the first one's room and (b) neither was
// cleared on leave, so post-leave SENDs vanished into a dead room object.
const wss = new WebSocketServer({ port: BRIDGE_PORT });
console.log(`[Bridge] Daemon listening on ws://127.0.0.1:${BRIDGE_PORT}`);

wss.on('connection', (pySocket) => {
    console.log("[Bridge] Python Agent connected.");

    const session = { room: null, ping: null, closed: false };

    // AUDIT: every send was unguarded. onStateChange/onMessage/onError fire
    // from Colyseus internals, so a throw became an unhandled rejection while
    // the room stayed joined.
    const say = (obj) => {
        if (session.closed || pySocket.readyState !== WebSocket.OPEN) return;
        try { pySocket.send(JSON.stringify(obj)); }
        catch (err) { console.error("[Bridge] send failed:", err.message); }
    };

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
                await connectToGame(cmd.cookies, say, session);
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
        if (session.closed) return;
        session.closed = true;
        console.log(`[Bridge] Cleaning up session (${why}).`);
        dropRoom();
    };
    pySocket.on('close', () => cleanup("python agent disconnected"));
    pySocket.on('error', (e) => cleanup(`python socket error: ${e.message}`));
});

async function connectToGame(cookieString, say, session) {
    const headers = {
        'Cookie': cookieString,
        'User-Agent': USER_AGENT,
        'Accept': 'application/json, text/plain, */*',
        'Referer': 'https://belot.md/'
    };

    class CustomWebSocket extends WebSocket {
        constructor(url, protocols) {
            super(url, protocols, { headers: { 'Origin': 'https://belot.md', 'User-Agent': USER_AGENT, 'Cookie': cookieString } });
        }
    }
    global.WebSocket = CustomWebSocket;

    try {
        let wsUrl, roomId, playerToken, playerId;

        console.log("[Bridge] Checking for an ongoing active game...");
        let pageRes = await fetch("https://belot.md/gameplay_new.php", { headers });
        let html = await pageRes.text();

        wsUrl = html.match(/wsUrl:\s*"([^"]+)"/)?.[1];
        roomId = html.match(/roomId:\s*"([^"]+)"/)?.[1];
        playerToken = html.match(/playerToken:\s*"([^"]+)"/)?.[1];
        playerId = html.match(/playerId:\s*"([^"]+)"/)?.[1];

        if (wsUrl && roomId) {
            console.log(`[Bridge] Rejoining active game room: ${roomId}`);
        } else {
            console.log("[Bridge] No active game. Searching public tables...");
            const lobbyParams = new URLSearchParams({ getMeseNew: "1", nrJucatori: "4", minStatus: "0" });
            const lobbyRes = await fetch("https://belot.md/gameTables.php", { method: "POST", headers, body: lobbyParams });
            const lobbyData = await lobbyRes.json();
            
            const openTable = lobbyData.mese?.find(t => t.full === 0 && t.pass === false && t.minStatus === undefined); // && t.creator.includes("UnBaiat1") 
            if (!openTable) throw new Error("No public open tables available.");

            console.log(`[Bridge] Reserving seat at table ${openTable.id}...`);
            const joinParams = new URLSearchParams({ enterGame: "4", gameId: openTable.id, password: "" });
            const joinRes = await fetch("https://belot.md/gameTables.php", { method: "POST", headers, body: joinParams });
            const joinData = await joinRes.json();
            if (joinData.success !== "gameplay_new.php") throw new Error("Seat reservation failed.");

            pageRes = await fetch("https://belot.md/gameplay_new.php", { headers });
            html = await pageRes.text();

            wsUrl = html.match(/wsUrl:\s*"([^"]+)"/)?.[1];
            roomId = html.match(/roomId:\s*"([^"]+)"/)?.[1];
            playerToken = html.match(/playerToken:\s*"([^"]+)"/)?.[1];
            playerId = html.match(/playerId:\s*"([^"]+)"/)?.[1];

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
            fetch("https://belot.md/api/game.php?ping", { headers }).catch(() => {});
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
        say({ event: "ERROR", message: err.message });
    }
}