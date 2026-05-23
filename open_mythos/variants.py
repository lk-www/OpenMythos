from open_mythos.main import MythosConfig

# Parameter budget breakdown per variant:
#   total ≈ embed + prelude/coda dense blocks + recurrent MLA + MoE
#   MoE   = 3 * dim * expert_dim * (n_experts + n_shared * n_experts_per_tok)
# expert_dim is solved from the residual budget after all other terms.
def mythos_1m() -> MythosConfig:
    """1M (約1M〜2M) 極小ナノ構成。
    パイプラインのテスト、デバッグ、あるいは最小限のメモリ環境での動作検証用。
    """
    return MythosConfig(
        vocab_size=8000,        # 32000 -> 8000 に削減（埋め込み層のパラメータを最小化、テスト用トークナイザー想定）
        dim=64,                 # 256 -> 64 に極小化（1Mに抑えるための最大のレバー）
        n_heads=4,              # ヘッドあたり 64/4 = 16 次元
        n_kv_heads=1,           # KVは1ヘッドのみ（Grouped/Multi-Query構造の最小値）
        max_seq_len=512,        # 1024 -> 512 に短縮（コンテキスト窓のメモリ消費をほぼゼロに）
        max_loop_iters=2,       # 4 -> 2 に削減（再帰処理を最小限の2回に制限）
        prelude_layers=1,
        coda_layers=1,
        attn_type="mla",
        kv_lora_rank=16,        # dim=64 に合わせて極小化
        q_lora_rank=32,         # dim=64 に合わせて極小化
        qk_rope_head_dim=8,     # ヘッド次元に合わせて調整
        qk_nope_head_dim=16,    # ヘッド次元に合わせて調整
        v_head_dim=16,
        n_experts=4,            # エキスパート数を 16 -> 4 に厳選
        n_shared_experts=1,
        n_experts_per_tok=1,    # Top-1 ルーティングを維持
        expert_dim=128,         # dimの2倍（64 * 2）
        act_threshold=0.99,
        rope_theta=500000.0,
        lora_rank=2,            # LoRAランクも最小限の2に設定
    )
def mythos_30m() -> MythosConfig:
    """0.03B (約30M) 超軽量構成。
    極小の検証環境や、VRAMが極めて制限された環境向けのミニマムモデル。
    """
    return MythosConfig(
        vocab_size=32000,
        dim=256,                # 1024 -> 256 に極小化（計算量とメモリが劇的に減少）
        n_heads=8,              # ヘッド数を減らし、1ヘッドあたり 256/8 = 32 次元を維持
        n_kv_heads=2,           # KVヘッドも最小限に
        max_seq_len=1024,       # 2048 -> 1024 に短縮（コンテキスト窓のVRAMを最小化）
        max_loop_iters=4,       # 8 -> 4 に削減（再帰処理のアクティベーション保持を最小限に）
        prelude_layers=1,       # 前段ブロックを最小の1層に
        coda_layers=1,          # 後段ブロックを最小の1層に
        attn_type="mla",
        kv_lora_rank=64,        # dim=256 に合わせてLoRAランクを縮小
        q_lora_rank=128,        # dim=256 に合わせてLoRAランクを縮小
        qk_rope_head_dim=16,    # ヘッド次元の縮小に合わせて調整
        qk_nope_head_dim=32,    # ヘッド次元の縮小に合わせて調整
        v_head_dim=32,
        n_experts=16,           # 32 -> 16 に削減
        n_shared_experts=1,
        n_experts_per_tok=1,    # 2 -> 1 に削減（1トークンあたり1エキスパートのみ駆動で最軽量化）
        expert_dim=512,         # dimの2倍（256 * 2）程度に抑えてバランスを維持
        act_threshold=0.99,
        rope_theta=500000.0,
        lora_rank=4,            # 8 -> 4 に縮小
    )
def mythos_300m() -> MythosConfig:
    """ダウンサイジング版（約0.3B〜0.4B）。VRAM消費を大幅に削減し、24GB環境での学習・推論を安定化。"""
    return MythosConfig(
        vocab_size=32000,
        dim=1024,               # 2048 -> 1024 に半減（モデル全体の重みが4分の1近くに減少）
        n_heads=16,             # 各ヘッドの次元（dim/n_heads）を 1024/16 = 64 に維持
        n_kv_heads=4,
        max_seq_len=2048,       # 4096 -> 2048 に削減（アテンションのコンテキストVRAMを大幅節約）
        max_loop_iters=8,       # 16 -> 8 に半減（再帰ループによるアクティベーションキャッシュを削減）
        prelude_layers=2,
        coda_layers=2,
        attn_type="mla",
        kv_lora_rank=128,       # dimの縮小に合わせてLoRAランクも半減
        q_lora_rank=256,        # dimの縮小に合わせてLoRAランクも半減
        qk_rope_head_dim=32,
        qk_nope_head_dim=64,
        v_head_dim=64,
        n_experts=32,           # 64 -> 32 に削減（MoE全体の容量を削減）
        n_shared_experts=1,     # 2 -> 1 に削減
        n_experts_per_tok=2,    # 4 -> 2 に削減（1トークンあたりの計算量とトポロジーを軽量化）
        expert_dim=1024,        # dimに合わせて 2048 -> 1024 に削減
        act_threshold=0.99,
        rope_theta=500000.0,
        lora_rank=8,
    )
def mythos_1b() -> MythosConfig:
    """1B parameter config. Small research/fine-tuning model. dim=2048, 64 experts, 16 loop iters, 4k context."""
    return MythosConfig(
        vocab_size=32000,
        dim=2048,
        n_heads=16,
        n_kv_heads=4,
        max_seq_len=4096,
        max_loop_iters=16,
        prelude_layers=2,
        coda_layers=2,
        attn_type="mla",
        kv_lora_rank=256,
        q_lora_rank=512,
        qk_rope_head_dim=32,
        qk_nope_head_dim=64,
        v_head_dim=64,
        n_experts=64,
        n_shared_experts=2,
        n_experts_per_tok=4,
        expert_dim=2048,
        act_threshold=0.99,
        rope_theta=500000.0,
        lora_rank=8,
    )


def mythos_3b() -> MythosConfig:
    """3B parameter config. Compact inference model. dim=3072, 64 experts, 16 loop iters, 4k context."""
    return MythosConfig(
        vocab_size=32000, # 語彙数
        dim=3072, # 隠れ層の次元数(各ベクトルの「幅」(要素数)) 他のパラメータ(n_heads、expert_dim、qk_rope_head_dimなど)との整合性が必要
        n_heads=24,
        n_kv_heads=6,
        max_seq_len=4096,
        max_loop_iters=16, # Recurrent Blockでのloop回数
        prelude_layers=2,
        coda_layers=2,
        attn_type="mla",
        kv_lora_rank=384,
        q_lora_rank=768,
        qk_rope_head_dim=32,
        qk_nope_head_dim=96,
        v_head_dim=96,
        n_experts=64,
        n_shared_experts=2,
        n_experts_per_tok=4,
        expert_dim=4096,
        act_threshold=0.99, # 各トークン位置において累積確率質量がこの閾値を超えた時点で反復ループを早期終了する
        rope_theta=500000.0,
        lora_rank=8,
    )


def mythos_10b() -> MythosConfig:
    """10B parameter config. Mid-scale general model. dim=4096, 128 experts, 24 loop iters, 8k context."""
    return MythosConfig(
        vocab_size=32000,
        dim=4096,
        n_heads=32,
        n_kv_heads=8,
        max_seq_len=8192,
        max_loop_iters=24,
        prelude_layers=2,
        coda_layers=2,
        attn_type="mla",
        kv_lora_rank=512,
        q_lora_rank=1024,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        n_experts=128,
        n_shared_experts=2,
        n_experts_per_tok=4,
        expert_dim=5632,
        act_threshold=0.99,
        rope_theta=500000.0,
        lora_rank=16,
    )


def mythos_50b() -> MythosConfig:
    """50B parameter config. Large reasoning model. dim=6144, 256 experts, 32 loop iters, 8k context."""
    return MythosConfig(
        vocab_size=32000,
        dim=6144,
        n_heads=48,
        n_kv_heads=8,
        max_seq_len=8192,
        max_loop_iters=32,
        prelude_layers=3,
        coda_layers=3,
        attn_type="mla",
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        n_experts=256,
        n_shared_experts=4,
        n_experts_per_tok=4,
        expert_dim=9728,
        act_threshold=0.99,
        rope_theta=500000.0,
        lora_rank=32,
    )


def mythos_100b() -> MythosConfig:
    """100B parameter config. Frontier-class model. dim=8192, 256 experts, 32 loop iters, 1M context, 128k output."""
    return MythosConfig(
        vocab_size=32000,
        dim=8192,
        n_heads=64,
        n_kv_heads=8,
        max_seq_len=1000000,
        max_loop_iters=32,
        prelude_layers=4,
        coda_layers=4,
        attn_type="mla",
        kv_lora_rank=512,
        q_lora_rank=2048,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        n_experts=256,
        n_shared_experts=4,
        n_experts_per_tok=8,
        expert_dim=13568,
        act_threshold=0.99,
        rope_theta=1000000.0,
        lora_rank=64,
        max_output_tokens=131072,
    )


def mythos_500b() -> MythosConfig:
    """500B parameter config. Ultra-scale MoE model. dim=12288, 512 experts, 48 loop iters, 1M context, 128k output."""
    return MythosConfig(
        vocab_size=100000,
        dim=12288,
        n_heads=96,
        n_kv_heads=16,
        max_seq_len=1000000,
        max_loop_iters=48,
        prelude_layers=4,
        coda_layers=4,
        attn_type="mla",
        kv_lora_rank=1024,
        q_lora_rank=3072,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        n_experts=512,
        n_shared_experts=8,
        n_experts_per_tok=8,
        expert_dim=23040,
        act_threshold=0.99,
        rope_theta=1000000.0,
        lora_rank=128,
        max_output_tokens=131072,
    )


def mythos_1t() -> MythosConfig:
    """1T parameter config. Maximum scale. dim=16384, 512 experts, 64 loop iters, 1M context, 128k output."""
    return MythosConfig(
        vocab_size=100000,
        dim=16384,
        n_heads=128,
        n_kv_heads=16,
        max_seq_len=1000000,
        max_loop_iters=64,
        prelude_layers=6,
        coda_layers=6,
        attn_type="mla",
        kv_lora_rank=1024,
        q_lora_rank=4096,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        n_experts=512,
        n_shared_experts=8,
        n_experts_per_tok=8,
        expert_dim=34560,
        act_threshold=0.99,
        rope_theta=2000000.0,
        lora_rank=256,
        max_output_tokens=131072,
    )
