import { Client } from 'colyseus.js';
import WebSocket, { WebSocketServer } from 'ws';

const BRIDGE_PORT = 8765;
const USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36";

let activeRoom = null;
let pingInterval = null;

const wss = new WebSocketServer({ port: BRIDGE_PORT });
console.log(`[Bridge] Daemon listening on ws://127.0.0.1:${BRIDGE_PORT}`);

wss.on('connection', (pySocket) => {
    console.log("[Bridge] Python Agent connected.");

    pySocket.on('message', async (rawMsg) => {
        try {
            const cmd = JSON.parse(rawMsg);
            if (cmd.action === "CONNECT") {
                await connectToGame(cmd.cookies, pySocket);
            } else if (cmd.action === "SEND" && activeRoom) {
                activeRoom.send(cmd.type, cmd.payload);
            } else if (cmd.action === "LEAVE" && activeRoom) {
                activeRoom.leave();
            }
        } catch (err) {
            console.error("[Bridge Error]:", err.message);
        }
    });

    pySocket.on('close', () => {
        console.log("[Bridge] Python Agent disconnected. Cleaning up room...");
        if (activeRoom) activeRoom.leave();
        if (pingInterval) clearInterval(pingInterval);
    });
});

async function connectToGame(cookieString, pySocket) {
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
        activeRoom = await client.joinById(roomId, { token: playerToken, playerId: playerId });

        pingInterval = setInterval(() => {
            fetch("https://belot.md/api/game.php?ping", { headers }).catch(() => {});
        }, 30_000);

        activeRoom.onStateChange((state) => {
            pySocket.send(JSON.stringify({ event: "STATE", data: state, myPlayerId: playerId }));
        });

        activeRoom.onMessage("*", (type, message) => {
            pySocket.send(JSON.stringify({ event: "MESSAGE", type, data: message }));
        });

        activeRoom.onError((code, message) => {
            pySocket.send(JSON.stringify({ event: "ERROR", code, message }));
        });

        activeRoom.onLeave((code) => {
            pySocket.send(JSON.stringify({ event: "LEAVE", code }));
            if (pingInterval) clearInterval(pingInterval);
        });

        pySocket.send(JSON.stringify({ event: "CONNECTED", roomId, playerId }));

    } catch (err) {
        pySocket.send(JSON.stringify({ event: "ERROR", message: err.message }));
    }
}