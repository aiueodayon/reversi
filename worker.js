// ai_core.js (AIロジック) を読み込みます
// ※ このファイルが同じフォルダに必要です
importScripts('ai_core.js');

/**
 * Worker内で完結して高速に動くオセロエンジン
 * (UIの描画などを気にせず計算だけに集中するためのクラス)
 */
class FastOthello {
    constructor() {
        // 10x10のボード（番兵法用）
        // インデックス: 11~18, 21~28, ... 81~88 が実盤面
        this.board = new Int8Array(100); 
        this.directions = [-11, -10, -9, -1, 1, 9, 10, 11];
        this.reset();
    }

    reset() {
        this.board.fill(2); // 2は壁(番兵)
        for (let y = 1; y <= 8; y++) {
            for (let x = 1; x <= 8; x++) {
                this.board[y * 10 + x] = 0; // 0は空
            }
        }
        // 初期配置
        this.board[44] = -1; this.board[45] = 1;
        this.board[54] = 1;  this.board[55] = -1;
        this.turn = 1; // 1:黒, -1:白
        this.passCount = 0;
    }

    // 8x8の1次元配列(長さ64)を取得 (AI入力用)
    getFlatBoard64() {
        const flat = new Int8Array(64);
        let idx = 0;
        for (let y = 1; y <= 8; y++) {
            for (let x = 1; x <= 8; x++) {
                flat[idx++] = this.board[y * 10 + x];
            }
        }
        return flat;
    }

    // 合法手生成 (戻り値は10x10のインデックス)
    getLegalMoves() {
        const moves = [];
        for (let y = 1; y <= 8; y++) {
            for (let x = 1; x <= 8; x++) {
                const idx = y * 10 + x;
                if (this.board[idx] !== 0) continue;

                let canHit = false;
                for (let dir of this.directions) {
                    let curr = idx + dir;
                    if (this.board[curr] === -this.turn) {
                        while (this.board[curr] === -this.turn) curr += dir;
                        if (this.board[curr] === this.turn) {
                            canHit = true;
                            break;
                        }
                    }
                }
                if (canHit) moves.push(idx);
            }
        }
        return moves;
    }

    // 着手実行 (idxは10x10形式)
    makeMove(idx) {
        if (idx === -1) { // パス
            this.passCount++;
            this.turn = -this.turn;
            return;
        }

        this.passCount = 0;
        this.board[idx] = this.turn;
        const opponent = -this.turn;

        for (let dir of this.directions) {
            let curr = idx + dir;
            if (this.board[curr] === opponent) {
                let flippables = [];
                while (this.board[curr] === opponent) {
                    flippables.push(curr);
                    curr += dir;
                }
                if (this.board[curr] === this.turn) {
                    for (let f of flippables) this.board[f] = this.turn;
                }
            }
        }
        this.turn = -this.turn;
    }

    isGameOver() {
        return this.passCount >= 2 || !this.getFlatBoard64().includes(0);
    }

    getResult() {
        let b = 0, w = 0;
        for (let i = 0; i < 100; i++) {
            if (this.board[i] === 1) b++;
            if (this.board[i] === -1) w++;
        }
        if (b > w) return 1;
        if (w > b) return -1;
        return 0;
    }
}

// -----------------------------------------
// Worker メイン処理
// -----------------------------------------

let ai = null;
const game = new FastOthello();
let isTraining = false;

self.onmessage = async function(e) {
    const { type, payload } = e.data;

    switch (type) {
        case 'INIT':
            // AIインスタンス生成 (ai_core.js内のクラス)
            if (typeof UltimateAI !== 'undefined') {
                ai = new UltimateAI();
                console.log('Worker: AI Initialized');
            } else {
                console.error('Worker Error: ai_core.js not loaded properly.');
            }
            break;

        case 'START_TRAINING':
            if (!ai) return;
            isTraining = true;
            // 非同期で学習ループを開始
            trainLoop(payload.batchSize || 1000);
            break;

        case 'STOP_TRAINING':
            isTraining = false;
            break;

        case 'GET_MOVE':
            if (!ai) return;
            // UIからの対局リクエスト (payload.boardは64要素の配列)
            // AIに考えさせて手を返す
            const bestMoveIndex = ai.getBestMove(payload.board, payload.turn);
            self.postMessage({ type: 'MOVE_DECIDED', move: bestMoveIndex });
            break;
            
        case 'RESET_BRAIN':
            if(ai) {
                ai = new UltimateAI(); // 重みリセット
                self.postMessage({ type: 'DATA_UPDATED', weights: null, size: 0 });
            }
            break;
    }
};

// --- 学習ループ (自己対戦) ---
async function trainLoop(batchSize) {
    let gamesCount = 0;
    let winStats = { black: 0, white: 0, draw: 0 };
    const startTime = performance.now();

    // 座標変換ヘルパー
    const idx10to8 = (idx10) => {
        const y = Math.floor(idx10 / 10);
        const x = idx10 % 10;
        return (y - 1) * 8 + (x - 1);
    };

    while (isTraining && gamesCount < batchSize) {
        game.reset();
        const history = []; // 1局の履歴 (TD学習用)

        // 1ゲーム実行
        while (!game.isGameOver()) {
            const moves10 = game.getLegalMoves(); // 10x10形式の合法手
            let move10 = -1;
            
            const currentBoard64 = game.getFlatBoard64();
            history.push({
                board: currentBoard64,
                turn: game.turn
            });

            if (moves10.length > 0) {
                // ε-greedy法: 10%の確率でランダム、90%はAIの最善手
                // これにより多様な局面を学習させる
                if (Math.random() < 0.1) {
                    move10 = moves10[Math.floor(Math.random() * moves10.length)];
                } else {
                    // AIに選ばせる (内部で評価関数を使用)
                    // AIは0-63のインデックスを返すので、10x10に変換が必要
                    // ここでは簡易的に、現在のAI評価で最も高い手をmoves10の中から選ぶ処理を実装
                    
                    let bestScore = -Infinity;
                    let bestM = moves10[0];
                    
                    for(let m of moves10) {
                        // 1手進めた状態を仮想的に作るのが本来だが、
                        // 高速化のため「着手位置の静的評価」や「AIのgetBestMove」を利用
                        // ここでは単純に ai.getBestMove を呼ぶ
                        // (ただし getBestMove は全合法手を探索するので少し重い)
                        
                        // ※高速化のため、ここではAIに全探索させず、ランダムと混ぜて進める
                        // 本格的に強くするには、ここで ai.getBestMove を呼んでください
                    }
                    
                    // 今回はシンプルに AIクラスの getBestMove を利用 (多少遅くなるが正確)
                    const aiMoveIdx8 = ai.getBestMove(currentBoard64, game.turn);
                    
                    // 8x8 -> 10x10 変換
                    const r = Math.floor(aiMoveIdx8 / 8);
                    const c = aiMoveIdx8 % 8;
                    move10 = (r + 1) * 10 + (c + 1);
                }
            }

            game.makeMove(move10);
        }

        // 終局処理
        const result = game.getResult(); // 1(黒勝), -1(白勝), 0
        
        if (result === 1) winStats.black++;
        else if (result === -1) winStats.white++;
        else winStats.draw++;

        // 学習 (TD Learning: 履歴に基づいて重みを更新)
        ai.train(history, result);

        gamesCount++;

        // 100局ごとにUIへ進捗通知
        if (gamesCount % 100 === 0) {
            const now = performance.now();
            const elapsed = (now - startTime) / 1000;
            const speed = Math.floor(gamesCount / elapsed);

            self.postMessage({
                type: 'PROGRESS',
                data: {
                    games: gamesCount,
                    speed: speed,
                    stats: winStats
                }
            });
            // イベントループをブロックしないよう一瞬休む
            await new Promise(r => setTimeout(r, 0));
        }
    }

    self.postMessage({ type: 'TRAINING_COMPLETE' });
    isTraining = false;
}
