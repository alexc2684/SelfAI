# SelfAI

Using my Facebook message data to make a generative deep learning model to talk like me.

## Usage
1. To preprocess the input data, separate "contexts" and "responses" to respective .txt files
for training and test data.
```python preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/model```

2. To train the model, run this command. Further details on training parameters are in the [OpenNMT](https://github.com/OpenNMT/OpenNMT-py) documentation.
```python train.py -data data/model -dropout .2 -global_attention mlp -start_decay_steps 8 -tensorboard -tensorboard_log_dir self_v1 -save_model checkpoints/model_v1 -src_word_vec_size 256 -tgt_word_vec_size 256 -rnn_size 256```

3. Test out the model on test data! Adjust beam size to see next highest probable text sequences.
```python translate.py -model checkpoints/model_v1_step_<STEP_SIZE>.pt -src data/src-val.txt -beam_size 3 -replace_unk -verbose```

## Examples
The model yielded some decent results:
```
INPUT: can u come at 930
OUTPUT: ill head to ur place

INPUT: im at the library with George
OUTPUT: ok wordd ill come thru dude

INPUT: hey alex do u know if we haf the math test monday for bartlett
OUTPUT: yeah yo

```

And not so great (but none the less funny) ones:
```
INPUT: nope
OUTPUT: il bring sunglasses for kidz bop

INPUT: okk cya my phones dead tho i should probably go charget itt
OUTPUT: truuu haha we are currently in heated debate

INPUT: 9
OUTPUT: i hate math
```

Credit to [OpenNMT](https://github.com/OpenNMT/OpenNMT-py) for their Seq2Seq implementation.
