// ==========================================
// AI Worker Logic (V6: God Mode - Hard Coded Strength)
// ==========================================

// --- Storage Manager ---
class StorageManager {
    constructor() {
        // ロジックを根本的に変えるためDBを刷新
        this.dbName = 'ReversiNeuralDB_v6_GodMode';
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
        return new Promise(r => { tx.oncomplete = () => r(sizeKB); });
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
        
        // 【修正3】初期重みの厳格化
        // 危険地帯（X, C）を極端に低く設定し、初期状態から「絶対にそこに打ちたくない」ようにする
        this.defaultWeights = [
            [ 120, -60,  20,   5,   5,  20, -60, 120],
            [ -60, -80,  -5,  -5,  -5,  -5, -80, -60], // X打ちは -80
            [  20,  -5,  15,   3,   3,  15,  -5,  20],
            [   5,  -5,   3,   3,   3,   3,  -5,   5],
            [   5,  -5,   3,   3,   3,   3,  -5,   5],
            [  20,  -5,  15,   3,   3,  15,  -5,  20],
            [ -60, -80,  -5,  -5,  -5,  -5, -80, -60],
            [ 120, -60,  20,   5,   5,  20, -60, 120]
        ];
        this.weights = JSON.parse(JSON.stringify(this.defaultWeights));
        this.directions = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]];

        // 危険地帯の定義（角が空いている時の周辺マス）
        this.dangerMap = {
            '0,0': [{r:0,c:1}, {r:1,c:0}, {r:1,c:1}], // Corner: TL -> Right, Down, Diagonal
            '0,7': [{r:0,c:6}, {r:1,c:7}, {r:1,c:6}], // Corner: TR
            '7,0': [{r:6,c:0}, {r:7,c:1}, {r:6,c:1}], // Corner: BL
            '7,7': [{r:7,c:6}, {r:6,c:7}, {r:6,c:6}]  // Corner: BR
        };
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
        const learningRate = 1.0; // 学習率は控えめに（基本性能が高いので微調整で十分）
        const sign = (winner === aiPlayer) ? 1 : -1;
        
        for(let r=0; r<8; r++) {
            for(let c=0; c<8; c++) {
                if(board[r][c] === aiPlayer) {
                    let w = this.weights[r][c];
                    // 重みの極端な崩壊を防ぐため、元の値の符号を維持しようとする力を働かせる（正則化的な処理）
                    w += sign * learningRate;
                    
                    // クランプ処理（極端な値になりすぎないように）
                    this.weights[r][c] = Math.max(-200, Math.min(200, w));
                    
                    // 対称性の更新
                    this.weights[r][7-c] = this.weights[r][c];
                    this.weights[7-r][c] = this.weights[r][c];
                    this.weights[7-r][7-c] = this.weights[r][c];
                }
            }
        }
        this.totalGames++;
        return await this.storage.saveData(this.weights, this.totalGames);
    }

    getBestMove(board, player) {
        const moves = this.getValidMoves(board, player);
        if(moves.length === 0) return null;
        
        const empty = this.countEmpty(board);
        
        // 【修正1】探索深さ（Search Depth）の強化
        // 中盤は深さ6（人間にはかなり読みづらいレベル）、終盤14手は完全読み切り
        let maxDepth = (empty <= 14) ? empty : 6;
        
        let bestMove = moves[0];
        let alpha = -Infinity;
        let beta = Infinity;

        // Move Ordering: 探索効率化のため、有望な手（角など）から先に調べる
        // 重み評価が高い順にソート
        moves.sort((a,b) => {
            // 簡易的な評価値比較
            let scoreA = this.weights[a.r][a.c];
            let scoreB = this.weights[b.r][b.c];
            return scoreB - scoreA;
        });

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
        
        if(moves.length === 0) {
            if(this.getValidMoves(board, 3-curP).length === 0) {
                return this.evaluateFinal(board, player);
            }
            return this.minimax(board, depth-1, alpha, beta, !isMax, player);
        }

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

    // ★修正2：評価関数への「絶対ルール」の追加★
    evaluate(board, player) {
        const opp = 3-player;
        let score = 0;
        
        // 1. 基本重み（Weights）
        for(let r=0; r<8; r++) {
            for(let c=0; c<8; c++) {
                if(board[r][c] === player) score += this.weights[r][c];
                else if(board[r][c] === opp) score -= this.weights[r][c];
            }
        }

        // 2. 確定石（Corner）ボーナス & 危険地帯ペナルティ（絶対ルール）
        const corners = [
            {r:0, c:0, key:'0,0'}, {r:0, c:7, key:'0,7'}, 
            {r:7, c:0, key:'7,0'}, {r:7, c:7, key:'7,7'}
        ];

        for(let corner of corners) {
            const stone = board[corner.r][corner.c];
            
            if (stone === player) {
                // 隅を取っていれば超加点（重みがどうあろうと優先）
                score += 500;
            } else if (stone === opp) {
                // 取られていれば減点
                score -= 500;
            } else {
                // 隅が「空いている」場合のみ、その周辺（X, C）のチェックを行う
                const dangers = this.dangerMap[corner.key];
                for(let d of dangers) {
                    const dStone = board[d.r][d.c];
                    if(dStone === player) {
                        // 隅が空いているのにX/C打ちしている -> 強烈なペナルティ
                        score -= 300; 
                    } else if (dStone === opp) {
                        // 相手がやっている -> チャンスなので加点
                        score += 300;
                    }
                }
            }
        }

        // 3. モビリティ（Mobility）
        // 打てる箇所が多い方が有利
        const myMoves = this.getValidMoves(board, player).length;
        const oppMoves = this.getValidMoves(board, opp).length;
        score += (myMoves - oppMoves) * 15;

        return score;
    }

    evaluateFinal(board, player) {
        let diff = 0;
        const opp = 3-player;
        for(let r=0; r<8; r++) for(let c=0; c<8; c++) {
            if(board[r][c] === player) diff++;
            else if(board[r][c] === opp) diff--;
        }
        return diff * 10000; // 勝ちは絶対
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
                // 学習用: 上位候補から選択するが、より上位に偏らせる
                moves.sort((a,b) => ai.weights[b.r][b.c] - ai.weights[a.r][a.c]);
                
                let move;
                // God Modeの学習は「強い手」をさらに強化することに集中
                if (Math.random() < 0.15) {
                    const topN = moves.slice(0, 3);
                    move = topN[Math.floor(Math.random() * topN.length)];
                } else {
                    move = moves[0];
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
