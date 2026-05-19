# `input_ids (B, T)` とは

言語モデルへの**入力テンソル**で、テキストをトークンID列に変換した整数配列です。

## 各記号の意味

| 記号 | 意味 | 例 |
|---|---|---|
| **B** | Batch size — 同時に処理する文(系列)の本数 | 2 (= 2文を並列処理) |
| **T** | sequence length — 各文のトークン数 | 16 (= 16トークン) |
| **input_ids** | 各位置のトークンID(語彙表のインデックス、整数) | 整数 ∈ [0, vocab_size) |

つまり `input_ids` は **形状 `(B, T)` の `torch.LongTensor`** で、要素が「語彙のどの単語か」を指す番号です。

## 具体例

`vocab_size=1000`, `B=2`, `T=16` のとき:

```python
input_ids = torch.tensor([
    [ 5, 423,  77,  12,  ...,   8],   # 1文目: 16個のトークンID
    [89,  91, 222, 999,  ...,   3],   # 2文目: 16個のトークンID
])  # shape: (2, 16)
```

実際のテキストとの対応 (例として `"Hello world"`):

```
"Hello world"
   │
   ▼ tokenizer.encode
[15496, 995]            ← T = 2
   │
   ▼ batch化
[[15496, 995]]          ← shape (B=1, T=2) のテンソル = input_ids
```

## このリポジトリでの流れ

`OpenMythos.forward()` の中で:

```python
def forward(self, input_ids: torch.Tensor, ...):
    T = input_ids.shape[1]              # 系列長
    x = self.embed(input_ids)           # (B, T) → (B, T, dim) 連続ベクトルへ
    ...
    return self.head(...)               # (B, T, vocab_size) ロジット
```

| ステージ | テンソル形状 | dtype |
|---|---|---|
| 入力 | `input_ids: (B, T)` | int64 |
| 埋め込み後 | `x: (B, T, dim)` | float |
| 最終出力 | `logits: (B, T, vocab_size)` | float |

`input_ids` の値は `0 ≤ id < cfg.vocab_size` を満たす必要があり、`MythosTokenizer.encode("...")` がこの整数列を生成します。

## 参考リンク

- [PyTorch nn.Embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html) — `(B, T)` の整数を `(B, T, dim)` のベクトルに変換するレイヤ
- [HuggingFace Glossary: input_ids](https://huggingface.co/docs/transformers/glossary#input-ids) — 公式用語集の解説
- [Tokenization 入門 (HF course)](https://huggingface.co/learn/nlp-course/chapter2/4) — テキスト → input_ids への変換手順