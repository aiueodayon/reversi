// ==========================================
// AI Worker Logic
// ==========================================

// --- Storage Manager ---
class StorageManager {
    constructor() {
        this.dbName = 'ReversiNeuralDB_v3';
        this.db = null;
    }
    async init() {
        return new Promise((resolve, reject) => {
            const req = indexedDB.open(this.dbName, 1);
            req.onupgradeneeded = (e) => {
                const db = e.target.result;
                if (!db.objectStoreNames.contains('brain')) db.createObjectStore('brain', { keyPath: 'id' });
            };
            req.onsuccess = (e) => { this.db = e.target.result; resolve(); };
            req.onerror = (e) => reject(e);
        });
    }
    async saveData(weights, totalGames) {
        if(!this.db) await this.init();
        const tx = this.db.transaction(['brain'], 'readwrite');
        const data = { id: 'current', weights: weights, totalGames: totalGames };
        tx.objectStore('brain').put(data);
        
        const jsonStr = JSON.stringify(data);
        const sizeKB = (new Blob([jsonStr]).size / 1024).toFixed(2);
        
        return new Promise(r => {
            tx.oncomplete = () => r(sizeKB);
        });
    }
    async loadData() {
        if(!this.db) await this.init();
        return new Promise(resolve => {
            const req = this.db.transaction(['brain'], 'readonly').objectStore('brain').get('current');
            req.onsuccess = () => resolve(req.result ? req.result : null);
        });
    }
}

// --- Ultimate AI Engine ---
class UltimateAI {
    constructor() {
        this.storage = new StorageManager();
        this.totalGames = 0;
        
        // Initial heuristics
        this.defaultWeights = [
            [100, -40, 10,  5,  5, 10, -40, 100],
            [-40, -80, -2, -2, -2, -2, -80, -40],
            [ 10,  -2, -1, -1, -1, -1,  -2,  10],
            [  5,  -2, -1, -1, -1, -1,  -2,   5],
            [  5,  -2, -1, -1, -1, -1,  -2,   5],
            [ 10,  -2, -1, -1, -1, -1,  -2,  10],
            [-40, -80, -2, -2, -2, -2, -80, -40],
            [100, -40, 10,  5,  5, 10, -40, 100]
        ];
        this.weights = JSON.parse(JSON.stringify(this.defaultWeights));
        this.directions = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]];
        
        this.dangerZones = [
            {r:0, c:0, targets: [{r:1, c:1, type:'X'}, {r:0, c:1, type:'C'}, {r:1, c:0, type:'C'}]},
            {r:0, c:7, targets: [{r:1, c:6, type:'X'}, {r:0, c:6, type:'C'}, {r:1, c:7, type:'C'}]},
            {r:7, c:0, targets: [{r:6, c:1, type:'X'}, {r:7, c:1, type:'C'}, {r:6, c:0, type:'C'}]},
            {r:7, c:7, targets: [{r:6, c:6, type:'X'}, {r:7, c:6, type:'C'}, {r:6, c:7, type:'C'}]}
        ];
    }

    async load() {
        const data = await this.storage.loadData();
        let size = "0.00";
        if(data) {
            this.weights = data.weights;
            this.totalGames = data.totalGames || 0;
            const jsonStr = JSON.stringify(data);
            size = (new Blob([jsonStr]).size / 1024).toFixed(2);
        } else {
            size = await this.storage.saveData(this.weights, 0);
        }
        return { total: this.totalGames, size: size, weights: this.weights };
    }

    reset() {
        this.weights = JSON.parse(JSON.stringify(this.defaultWeights));
        this.totalGames = 0;
        return this.storage.saveData(this.weights, 0);
    }

    async train(board, winner, aiPlayer) {
        if(winner === 0) return null;
        const learningRate = 2.0;
        const sign = (winner === aiPlayer) ? 1 : -1;
        
        for(let r=0; r<8; r++) {
            for(let c=0; c<8; c++) {
                if(board[r][c] === aiPlayer) {
                    this.weights[r][c] += sign * learningRate;
                    // Symmetry update
                    this.weights[r][7-c] += sign * learningRate;
                    this.weights[7-r][c] += sign * learningRate;
                    this.weights[7-r][7-c] += sign * learningRate;
                    // Clamp values
                    this.weights[r][c] = Math.max(-300, Math.min(300, this.weights[r][c]));
                }
            }
        }
        this.totalGames++;
        return await this.storage.saveData(this.weights, this.totalGames);
    }

    getBestMove(board, player) {
        const moves = this.getValidMoves(board, player);
        if(moves.length === 0) return null;
        
        // Sort by current knowledge (Depth 0)
        moves.sort((a,b) => this.weights[b.r][b.c] - this.weights[a.r][a.c]);

        // WLD / Minimax Logic
        const empty = this.countEmpty(board);
        let maxDepth = (empty <= 14) ? empty : 4; 
        
        let bestMove = moves[0];
        let alpha = -Infinity;
        let beta = Infinity;

        for (let m of moves) {
            const nb = this.clone(board);
            this.execute(nb, m.r, m.c, player);
            const val = this.minimax(nb, maxDepth - 1, alpha, beta, false, player);
            if (val > alpha) {
                alpha = val;
                bestMove = m;
            }
        }
        return bestMove;
    }

    minimax(board, depth, alpha, beta, isMax, player) {
        if(depth === 0) return this.evaluate(board, player);
        const curP = isMax ? player : 3-player;
        const moves = this.getValidMoves(board, curP);
        if(moves.length === 0) return this.evaluate(board, player);

        if(isMax) {
            let max = -Infinity;
            for(let m of moves) {
                const nb = this.clone(board);
                this.execute(nb, m.r, m.c, curP);
                const v = this.minimax(nb, depth-1, alpha, beta, false, player);
                max = Math.max(max, v);
                alpha = Math.max(alpha, v);
                if(beta <= alpha) break;
            }
            return max;
        } else {
            let min = Infinity;
            for(let m of moves) {
                const nb = this.clone(board);
                this.execute(nb, m.r, m.c, curP);
                const v = this.minimax(nb, depth-1, alpha, beta, true, player);
                min = Math.min(min, v);
                beta = Math.min(beta, v);
                if(beta <= alpha) break;
            }
            return min;
        }
    }

    evaluate(board, player) {
        const opp = 3-player;
        let score = 0;
        for(let r=0; r<8; r++) {
            for(let c=0; c<8; c++) {
                if(board[r][c] === player) score += this.weights[r][c];
                else if(board[r][c] === opp) score -= this.weights[r][c];
            }
        }
        return score;
    }

    clone(b) { return b.map(r => [...r]); }
    countEmpty(b) { return b.flat().filter(x=>x===0).length; }
    isValid(b, r, c, p) {
        if(b[r][c]!==0) return false;
        for(let d of this.directions) if(this.canFlip(b,r,c,d,p)) return true;
        return false;
    }
    canFlip(b, r, c, d, p) {
        let nr=r+d[0], nc=c+d[1], opp=3-p, hasOpp=false;
        while(nr>=0 && nr<8 && nc>=0 && nc<8) {
            if(b[nr][nc]===opp) hasOpp=true;
            else if(b[nr][nc]===p) return hasOpp;
            else return false;
            nr+=d[0]; nc+=d[1];
        }
        return false;
    }
    execute(b, r, c, p) {
        b[r][c] = p;
        for(let d of this.directions) {
            if(this.canFlip(b,r,c,d,p)) {
                let nr=r+d[0], nc=c+d[1];
                while(b[nr][nc]!==p) { b[nr][nc]=p; nr+=d[0]; nc+=d[1]; }
            }
        }
    }
    getValidMoves(b, p) {
        const ms=[];
        for(let r=0;r<8;r++) for(let c=0;c<8;c++) if(this.isValid(b,r,c,p)) ms.push({r,c});
        return ms;
    }
}

// --- Worker Controller ---
const ai = new UltimateAI();
let isTraining = false;

self.onmessage = async (e) => {
    const { type, payload } = e.data;

    if (type === 'INIT') {
        const info = await ai.load();
        self.postMessage({ type: 'READY', totalGames: info.total, size: info.size, weights: info.weights });
    } 
    else if (type === 'RESET_BRAIN') {
        const size = await ai.reset();
        self.postMessage({ type: 'READY', totalGames: 0, size: size, weights: ai.weights });
    }
    else if (type === 'GET_MOVE') {
        const move = ai.getBestMove(payload.board, payload.player);
        self.postMessage({ type: 'MOVE_RESULT', move });
    }
    else if (type === 'TRAIN_END_GAME') {
        const size = await ai.train(payload.board, payload.winner, payload.aiPlayer);
        if(size) self.postMessage({ type: 'DATA_UPDATED', totalGames: ai.totalGames, size: size, weights: ai.weights });
    }
    else if (type === 'START_TRAINING') {
        isTraining = true;
        runTrainingLoop();
    }
    else if (type === 'STOP_TRAINING') {
        isTraining = false;
    }
};

async function runTrainingLoop() {
    let gamesInSession = 0;
    let blackWins = 0;
    
    while(isTraining) {
        let board = Array(8).fill(0).map(()=>Array(8).fill(0));
        board[3][3]=2; board[4][4]=2; board[3][4]=1; board[4][3]=1;
        let cur = 1;
        let passCount = 0;
        
        while(true) {
            const moves = ai.getValidMoves(board, cur);
            if(moves.length === 0) {
                passCount++;
                if(passCount >= 2) break;
            } else {
                passCount = 0;
                // Epsilon-Greedy: 10% Random
                let move;
                if (Math.random() < 0.1) move = moves[Math.floor(Math.random() * moves.length)];
                else {
                    move = moves.sort((a,b) => ai.weights[b.r][b.c] - ai.weights[a.r][a.c])[0];
                }
                ai.execute(board, move.r, move.c, cur);
            }
            cur = 3 - cur;
        }

        let b=0, w=0;
        board.flat().forEach(x => x===1?b++:x===2?w++:0);
        const winner = b>w ? 1 : (w>b ? 2 : 0);
        if(winner === 1) blackWins++;

        let newSize = null;
        if(winner !== 0) {
            newSize = await ai.train(board, winner, winner); 
        }

        gamesInSession++;
        
        if(gamesInSession % 10 === 0) {
            self.postMessage({ 
                type: 'TRAIN_UPDATE', 
                count: gamesInSession, 
                totalGames: ai.totalGames,
                size: newSize,
                lastWinner: winner === 1 ? 'Black' : (winner===2?'White':'Draw'),
                winRate: Math.round((blackWins/gamesInSession)*100),
                weights: ai.weights 
            });
        }
    }
    self.postMessage({ type: 'TRAIN_STOPPED', total: ai.totalGames });
}
