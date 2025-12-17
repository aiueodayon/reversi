// ==========================================
// AI Worker Logic (Mobility Enhanced)
// ==========================================

// --- Storage Manager ---
class StorageManager {
    constructor() {
        this.dbName = 'ReversiNeuralDB_v4_Mobility'; // バージョン変更（DBを新規作成させるため）
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
        
        // 初期の重み（ヒューリスティック）
        // モビリティ導入に伴い、極端なマイナス値を少しマイルドに調整
        this.defaultWeights = [
            [120, -20, 20,  5,  5, 20, -20, 120],
            [-20, -40, -5, -5, -5, -5, -40, -20],
            [ 20,  -5, 15,  3,  3, 15,  -5,  20],
            [  5,  -5,  3,  3,  3,  3,  -5,   5],
            [  5,  -5,  3,  3,  3,  3,  -5,   5],
            [ 20,  -5, 15,  3,  3, 15,  -5,  20],
            [-20, -40, -5, -5, -5, -5, -40, -20],
            [120, -20, 20,  5,  5, 20, -20, 120]
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

    // 学習: 勝敗に応じて重みを更新
    async train(board, winner, aiPlayer) {
        if(winner === 0) return null;
        
        // 学習率（徐々に下げるのが定石だが、簡易的に固定）
        const learningRate = 1.5;
        const sign = (winner === aiPlayer) ? 1 : -1;
        
        for(let r=0; r<8; r++) {
            for(let c=0; c<8; c++) {
                if(board[r][c] === aiPlayer) {
                    this.weights[r][c] += sign * learningRate;
                    
                    // 対称性を使って学習効率を4倍にする
                    this.weights[r][7-c] += sign * learningRate;
                    this.weights[7-r][c] += sign * learningRate;
                    this.weights[7-r][7-c] += sign * learningRate;

                    // 重みが発散しないように制限
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
        
        // 序盤の定石DBがあれば強いが、ここでは探索深さでカバー
        // 空きマス数に応じて深さを可変にする
        const empty = this.countEmpty(board);
        
        // 終盤完全読み切り（14手以下なら最後まで読む）
        // 中盤は4手読み（重い場合は3に減らす）
        let maxDepth = (empty <= 14) ? empty : 4; 
        
        let bestMove = moves[0];
        let alpha = -Infinity;
        let beta = Infinity;

        // Move Ordering: 探索効率を上げるため、評価値が高そうな手から調べる
        moves.sort((a,b) => this.weights[b.r][b.c] - this.weights[a.r][a.c]);

        for (let m of moves) {
            const nb = this.clone(board);
            this.execute(nb, m.r, m.c, player);
            
            // 再帰呼び出し
            const val = this.minimax(nb, maxDepth - 1, alpha, beta, false, player);
            
            if (val > alpha) {
                alpha = val;
                bestMove = m;
            }
        }
        return bestMove;
    }

    // Minimax法（Alpha-Beta法による枝刈り付き）
    minimax(board, depth, alpha, beta, isMax, player) {
        // 葉ノードまたは終局
        if(depth === 0) return this.evaluate(board, player);
        
        const curP = isMax ? player : 3-player;
        const moves = this.getValidMoves(board, curP);
        
        // パスの処理
        if(moves.length === 0) {
            // 相手も打てないなら終局
            if(this.getValidMoves(board, 3-curP).length === 0) {
                return this.evaluateFinal(board, player); // 最終石差
            }
            // パスして探索継続
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
                if(beta <= alpha) break; // Beta Cut
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
                if(beta <= alpha) break; // Alpha Cut
            }
            return min;
        }
    }

    // ★強化された評価関数★
    evaluate(board, player) {
        const opp = 3-player;
        let score = 0;
        
        // 1. 位置の重み（Positional Strategy）
        for(let r=0; r<8; r++) {
            for(let c=0; c<8; c++) {
                if(board[r][c] === player) score += this.weights[r][c];
                else if(board[r][c] === opp) score -= this.weights[r][c];
            }
        }

        // 2. モビリティ（Mobility Strategy）
        // 自分の打てる手が多く、相手の打てる手が少ないほど有利
        const myMoves = this.getValidMoves(board, player).length;
        const oppMoves = this.getValidMoves(board, opp).length;
        
        // モビリティの重み係数（盤面状況によるが、1手あたり10〜15点相当の価値）
        const mobilityWeight = 12; 
        score += (myMoves - oppMoves) * mobilityWeight;

        return score;
    }

    // 終局時の評価（石の差そのもの）
    evaluateFinal(board, player) {
        let diff = 0;
        const opp = 3-player;
        for(let r=0; r<8; r++) for(let c=0; c<8; c++) {
            if(board[r][c] === player) diff++;
            else if(board[r][c] === opp) diff--;
        }
        return diff * 1000; // 勝ちは絶対的なので大きな値を返す
    }

    // --- Utility Methods ---
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
    
    // トレーニングループでは高速化のため、探索深さを浅くするか
    // あるいは「学習した重み」のみで打つ（Greedy）かを選択します。
    // ここでは「重み+モビリティ」の1手読み（深さ1）程度がバランス良いですが、
    // 数をこなすために今まで通り重みベースのGreedyで行います。
    // ※実戦（GET_MOVE）ではMinimaxを使うので強くなります。
    
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
                
                // 10%ランダム（探索）
                let move;
                if (Math.random() < 0.1) {
                    move = moves[Math.floor(Math.random() * moves.length)];
                } else {
                    // 重みベースの高速選択
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
