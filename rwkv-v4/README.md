https://github.com/BlinkDL/RWKV-LM

RWKV is a RNN with transformer-level performance. It can be directly trained like a GPT (parallelizable). So it's combining the best of RNN and transformer - great performance, fast inference, saves VRAM, fast training, "infinite" `ctx_len`, and free `sentence embedding`.


# RWKV: RNN with Transformer-level Performance

RWKV is a RNN with Transformer-level performance, which can also be directly trained like a GPT transformer (parallelizable). And it's attention-free. You only need the hidden state at position `t` to compute the state at position `t+1`. You can use the "GPT" mode to quickly computer the hidden state for the "RNN" mode.

So it's combining the best of RNN and transformer - great performance, fast inference, saves VRAM, fast training, "infinite" ctx_len, and free sentence embedding (using the final hidden state).

Training speed: RWKV-4 1.5B BF16 ctxlen1024 = 106K tokens/s on 8xA100 40G.

You are welcome to join the RWKV discord https://discord.gg/bDSBUMeFpc to build upon it. We have plenty of potential compute (A100 40Gs) now (thanks to Stability and EleutherAI), so if you have interesting ideas I can run them.

## Inference without matrix-matrix multiplications

All of the trained models will be open-source. Inference is very fast (only matrix-vector multiplications, no matrix-matrix multiplications) even on CPUs, so you can even run a LLM on your phone.

RWKV is parallelizable because the **time-decay of each channel is data-independent** (and trainable). For example, in usual RNN you can adjust the time-decay of a channel from say 0.8 to 0.5 (these are called "gates"), while in RWKV you simply move the information from a W-0.8-channel to a W-0.5-channel to achieve the same effect. Moreover, you can fine-tune RWKV into a non-parallelizable RNN (then you can use outputs of later layers of the previous token) if you want extra performance.

RWKV is a RNN and very friendly for edge devices. Let's make it possible to run a LLM on your phone.

## Quick start

Colab for RWKV-4 Pile 1.5B https://colab.research.google.com/drive/1F7tZoPZaWJf1fsCmZ5tjw6sYHiFOYVWM

### Run RWKV-4 Pile models

Download models from https://huggingface.co/BlinkDL. Set `TOKEN_MODE = 'pile'` in run.py and run it. It's fast even on CPU (the default mode).

### Training / Fine-tuning

Training RWKV-4 from scratch: run train.py, which by default is using the enwik8 dataset (unzip https://data.deepai.org/enwik8.zip).

You will be training the "GPT" version because it's paralleziable and faster to train. RWKV-4 can extrapolate, so training with ctxLen 1024 can work for ctxLen of 2500+. You can fine-tune the model with longer ctxLen and it can quickly adapt to longer ctxLens.

## How it works

RWKV is inspired by [Apple's AFT](https://arxiv.org/abs/2105.14103)

Moreover it's using a number of tricks, such as:

- [`SmallInitEmb`](https://github.com/BlinkDL/SmallInitEmb)
- [`Token-shift`](https://github.com/BlinkDL/RWKV-LM#token-shift-time-shift-mixing)
- [`Head-QK`](https://github.com/BlinkDL/RWKV-LM#the-head-qk-trick-learning-to-copy-and-avoid-tokens)
- Extra R-gate in the FFN (applicable to all transformers)

More info https://github.com/BlinkDL/RWKV-LM#the-pseudocode-execution-from-top-to-bottom