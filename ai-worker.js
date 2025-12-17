// ==========================================
// AI Worker Logic (Mobility + Stability + Smarter Training)
// ==========================================

// --- Storage Manager ---
class StorageManager {
    constructor() {
        // ロジックが変わったのでDBバージョンを変更してリセット
        this.dbName = 'ReversiNeuralDB_v5_Final'; 
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
        
        // 初期の重み
        // 確定石ボーナスをevaluateで加算するため、角の基礎点は少し抑えめでもOKですが
        // 学習の指針として高めにしておきます。
        this.defaultWeights = [
            [100, -20, 20,  5,  5, 20, -20, 100],
            [-20, -40, -5, -5, -5, -5, -40, -20],
            [ 20,  -5, 15,  3,  3, 15,  -5,  20],
            [  5,  -5,  3,  3,  3,  3,  -5,   5],
            [  5,  -5,  3,  3,  3,  3,  -5,   5],
            [ 20,  -5, 15,  3,  3, 15,  -5,  20],
            [-20, -40, -5, -5, -5, -5, -40, -20],
            [100, -20, 20,  5,  5, 20, -20, 100]
        ];
        this.weights = JSON.parse(JSON.stringify(this.defaultWeights));
        this.directions = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]];
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
        const learningRate = 1.5;
        const sign = (winner === aiPlayer) ? 1 : -1;
        
        for(let r=0; r<8; r++) {
            for(let c=0; c<8; c++) {
                if(board[r][c] === aiPlayer) {
                    // 角は重みが固定されがちなので、変動幅を少し小さくしても良いが
                    // ここでは単純化して一律適用
                    let update = sign * learningRate;
                    this.weights[r][c] += update;
                    this.weights[r][7-c] += update;
                    this.weights[7-r][c] += update;
                    this.weights[7-r][7-c] += update;
                    
                    this.weights[r][c] = Math.max(-500, Math.min(500, this.weights[r][c]));
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
        // 終盤14手読み切り、それ以外は4手読み
        let maxDepth = (empty <= 14) ? empty : 4; 
        
        let bestMove = moves[0];
        let alpha = -Infinity;
        let beta = Infinity;

        // Move Ordering: 重み順でソートして枝刈り効率アップ
        moves.sort((a,b) => this.weights[b.r][b.c] - this.weights[a.r][a.c]);

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

        // Move Ordering (Simple)
        // 深い探索では毎回ソートすると重いので、上位階層や残り手数が少ない時のみソート等の工夫も可。
        // ここではコードを単純に保つためソートなし、または単純な重みソートのみ適用可。
        // 高速化のため、Minimax内部ではソートを省略します（getBestMoveでの初期ソートが効くため）

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

    // ★評価関数の改善：重み + モビリティ + 確定石★
    evaluate(board, player) {
        const opp = 3-player;
        let score = 0;
        
        // 1. 位置の重み
        for(let r=0; r<8; r++) {
            for(let c=0; c<8; c++) {
                if(board[r][c] === player) score += this.weights[r][c];
                else if(board[r][c] === opp) score -= this.weights[r][c];
            }
        }

        // 2. 確定石（Corner）ボーナス
        // 重みテーブルでも表現できますが、絶対的な安全地帯として
        // 明示的に加点することで、読みの抜けを防ぎます。
        const corners = [[0,0], [0,7], [7,0], [7,7]];
        for(const [cr, cc] of corners) {
            if(board[cr][cc] === player) score += 150; 
            else if(board[cr][cc] === opp) score -= 150;
        }

        // 3. モビリティ（Mobility）
        // 相手の手数を減らす戦略
        const myMoves = this.getValidMoves(board, player).length;
        const oppMoves = this.getValidMoves(board, opp).length;
        
        // モビリティの重要度は非常に高い
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
        return diff * 1000;
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
                
                // ★学習用AIの思考ロジック改善（Top-3 Random）★
                // 重み順にソート
                moves.sort((a,b) => ai.weights[b.r][b.c] - ai.weights[a.r][a.c]);

                let move;
                // 20%の確率で「上位3手」からランダムに選ぶ（探索）
                // これにより、極端な悪手（リストの最後の方にある手）を排除しつつ多様性を確保
                if (Math.random() < 0.2) {
                    const topN = moves.slice(0, 3); // 上位3つ（手が3つ未満なら全て）
                    move = topN[Math.floor(Math.random() * topN.length)];
                } else {
                    // 80%は現在の一番良い手を選ぶ（活用）
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
