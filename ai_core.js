/**
 * ai_core.js
 * N-Tuple Network + TD Learning (Temporal Difference Learning)
 * * 特徴:
 * - 盤面の特定パターン(Tuple)の形を見て評価値を算出します。
 * - 8方向の対称性を利用して学習効率を高めています。
 * - 強化学習(TD法)により、自己対戦の結果から自動で重みを更新します。
 */

class UltimateAI {
    constructor() {
        // --- 設定 ---
        this.learningRate = 0.01; // 学習率 (高いと早く変わるが安定しない)
        this.gamma = 1.0;         // 割引率 (1.0 = 将来の報酬を重視)
        
        // --- N-Tuple 定義 ---
        // 盤面のどの位置を見るかのパターンセット (0-63のインデックス)
        // ここでは代表的な4つの形状を定義 (これを8方向対称展開して使用)
        this.rawTuples = [
            // 1. 角周りの強力なパターン (8マス)
            [0, 1, 2, 3, 8, 9, 10, 11],
            // 2. 辺のパターン (8マス)
            [0, 1, 2, 3, 4, 5, 6, 7],
            // 3. 斜めのライン (8マス)
            [0, 9, 18, 27, 36, 45, 54, 63],
            // 4. 中心のブロック (ボックス)
            [18, 19, 26, 27, 20, 21, 28, 29]
        ];

        // --- 対称性の事前計算 ---
        this.symmetryMaps = this.generateSymmetryMaps();
        
        // --- タプルの展開と重み配列の初期化 ---
        // 各タプルは3^8(6561)通りの状態を持つ
        // これを 4パターン × 8対称形 分用意する代わりに、
        // 重みを共有させる設計にしてメモリを節約・学習を高速化します。
        
        // 重みテーブル: weights[パターンID][3進数インデックス]
        // rawTuplesの数だけテーブルを作る
        this.weights = this.rawTuples.map(tuple => {
            const size = Math.pow(3, tuple.length); // 3^8 = 6561
            // Float32Arrayで高速化 (初期値は0付近の乱数)
            const arr = new Float32Array(size);
            for(let i=0; i<size; i++) arr[i] = (Math.random() - 0.5) * 0.01;
            return arr;
        });
    }

    /**
     * 盤面を受け取り、最善手(0-63)を返す
     * @param {Int8Array} board - 64要素の1次元配列 (1:黒, -1:白, 0:空)
     * @param {number} turn - 手番 (1 or -1)
     */
    getBestMove(board, turn) {
        const legalMoves = this.getLegalMoves(board, turn);
        
        if (legalMoves.length === 0) return -1; // パス

        let bestScore = -Infinity;
        let bestMove = -1;

        // 全ての合法手について、1手進めた後の盤面を評価する
        for (const move of legalMoves) {
            // 仮想的に手を打つ
            const nextBoard = this.simulateMove(board, move, turn);
            
            // 評価値を取得 (相手番の盤面価値なので、符号を反転させる考え方もあるが
            // ここでは evaluate() が「黒にとっての価値」を返すと定義する)
            let score = this.evaluate(nextBoard);

            // 手番が白(-1)なら、評価値(黒有利度)が低いほど良い
            if (turn === -1) score = -score;

            if (score > bestScore) {
                bestScore = score;
                bestMove = move;
            }
        }

        return bestMove;
    }

    /**
     * 盤面の評価値を計算 (黒有利: プラス, 白有利: マイナス)
     */
    evaluate(board) {
        let totalScore = 0;

        // 定義されたパターンごとに計算
        for (let t = 0; t < this.rawTuples.length; t++) {
            const rawTuple = this.rawTuples[t];
            const weightTable = this.weights[t];

            // 8つの対称形すべてについて評価を足し合わせる
            for (let s = 0; s < 8; s++) {
                const map = this.symmetryMaps[s];
                let index = 0;
                let power = 1;

                // 3進数インデックスの計算 (Base3: 空=0, 黒=1, 白=2)
                for (let i = 0; i < rawTuple.length; i++) {
                    const boardIdx = map[rawTuple[i]]; // 対称変換後の座標
                    const cell = board[boardIdx];
                    
                    // 0:空 -> 0, 1:黒 -> 1, -1:白 -> 2
                    const val = (cell === 0) ? 0 : (cell === 1 ? 1 : 2);
                    
                    index += val * power;
                    power *= 3;
                }
                
                totalScore += weightTable[index];
            }
        }
        return totalScore; // tanhなどは使わず、リニアな和を返す
    }

    /**
     * 学習実行 (TD(0)法)
     * ゲーム終了後に履歴を逆順にたどり、予測値と結果の誤差を修正する
     * @param {Array} history - {board: Int8Array, turn: number} の配列
     * @param {number} result - 1(黒勝), -1(白勝), 0(引分)
     */
    train(history, result) {
        // 最終的な正解信号 (報酬)
        // 勝敗結果そのものをターゲットにする
        let currentTarget = result; // 1.0, -1.0, or 0

        // 履歴を後ろから遡る (終了直前の手 -> 初手)
        for (let i = history.length - 1; i >= 0; i--) {
            const state = history[i];
            const board = state.board;
            
            // 現在の評価値 V(s)
            const currentVal = this.evaluate(board);

            // 誤差 δ = Target - V(s)
            const error = currentTarget - currentVal;

            // 重みの更新
            this.updateWeights(board, error);

            // TD学習: 次のステップ(時間的には前の手)のターゲットは
            // 現在の評価値に近づける (Target <- V(s))
            // これにより、結果の報酬が徐々に手前の局面に伝播する
            currentTarget = currentVal;
        }
    }

    /**
     * 誤差に基づいて重みを更新する
     */
    updateWeights(board, error) {
        // 全パターンの全対称形について、今回参照したインデックスの重みを修正
        const delta = error * this.learningRate; // 共通の修正量

        for (let t = 0; t < this.rawTuples.length; t++) {
            const rawTuple = this.rawTuples[t];
            const weightTable = this.weights[t];

            for (let s = 0; s < 8; s++) {
                const map = this.symmetryMaps[s];
                let index = 0;
                let power = 1;

                for (let i = 0; i < rawTuple.length; i++) {
                    const boardIdx = map[rawTuple[i]];
                    const cell = board[boardIdx];
                    const val = (cell === 0) ? 0 : (cell === 1 ? 1 : 2);
                    index += val * power;
                    power *= 3;
                }

                // 重みを更新
                // ※ 本来は「特徴が出現した回数」で割るなど正規化することもあるが
                // ここでは単純加算で実装 (出現した特徴全ての評価を少しずつ正解に寄せる)
                weightTable[index] += delta;
            }
        }
    }

    // --- ヘルパー関数 ---

    // 8通りの対称マップ(座標変換テーブル)を生成
    generateSymmetryMaps() {
        const maps = [];
        for (let s = 0; s < 8; s++) {
            const map = new Int8Array(64);
            for (let i = 0; i < 64; i++) {
                let r = Math.floor(i / 8);
                let c = i % 8;
                
                // 対称変換
                // sのビットで変換を組み合わせる (0-7)
                if (s & 1) c = 7 - c; // 左右反転
                if (s & 2) r = 7 - r; // 上下反転
                if (s & 4) {          // xy入替(対角反転)
                    const temp = r; r = c; c = temp;
                }
                map[i] = r * 8 + c;
            }
            maps.push(map);
        }
        return maps;
    }

    // 合法手列挙 (AI用: 64配列ベース)
    getLegalMoves(board, turn) {
        const moves = [];
        for (let i = 0; i < 64; i++) {
            if (board[i] === 0) {
                if (this.canFlip(board, i, turn)) {
                    moves.push(i);
                }
            }
        }
        return moves;
    }

    // 仮想的に手を打って新しい盤面配列を返す
    simulateMove(board, moveIdx, turn) {
        const newBoard = new Int8Array(board); // コピー
        newBoard[moveIdx] = turn;
        const r = Math.floor(moveIdx / 8);
        const c = moveIdx % 8;
        const dirs = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]];

        for (const d of dirs) {
            let nr = r + d[0], nc = c + d[1];
            let flippables = [];
            while (nr >= 0 && nr < 8 && nc >= 0 && nc < 8) {
                const idx = nr * 8 + nc;
                if (newBoard[idx] === -turn) {
                    flippables.push(idx);
                } else if (newBoard[idx] === turn) {
                    for (const f of flippables) newBoard[f] = turn;
                    break;
                } else {
                    break;
                }
                nr += d[0]; nc += d[1];
            }
        }
        return newBoard;
    }

    canFlip(board, idx, turn) {
        const r = Math.floor(idx / 8);
        const c = idx % 8;
        const dirs = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]];
        
        for (const d of dirs) {
            let nr = r + d[0], nc = c + d[1];
            let hasOpp = false;
            while (nr >= 0 && nr < 8 && nc >= 0 && nc < 8) {
                const val = board[nr * 8 + nc];
                if (val === -turn) {
                    hasOpp = true;
                } else if (val === turn) {
                    if (hasOpp) return true;
                    break;
                } else {
                    break;
                }
                nr += d[0]; nc += d[1];
            }
        }
        return false;
    }
}
