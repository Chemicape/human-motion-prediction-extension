# 3D Human Motion Prediction with RNNs and Transformer Extensions

This project is based on paper:

Julieta Martinez, Michael J. Black, Javier Romero. On human motion prediction using recurrent neural networks. In CVPR 17.

The original code for the paper is in [Github link](https://github.com/una-dinosauria/human-motion-prediction).

This is an RNN-based human motion prediction system for the Human3.6M dataset, uses a GRU-based seq2seq model to predict future 3D human poses from past motion. Our version extends the original pipeline by adding a Transformer-based model while keeping the original preprocessing, batching, evaluation, sampling, and visualization pipeline.

---

## 1. Project Structure

Main files:

```text
src/
  translate.py              # Main training, evaluation, and sampling script
  seq2seq_model.py          # Original GRU/RNN seq2seq model
  transformer_model.py      # Added Transformer model
  data_utils.py             # Human3.6M data loading and preprocessing
  forward_kinematics.py     # Convert predicted angles to 3D skeleton visualization
  viz.py                    # 3D skeleton plotting utilities
  baselines.py              # Zero-velocity / running average baseline
```

Important output files:

```text
experiments/                # Model checkpoints and TensorBoard logs
samples.h5                  # Sampled prediction results
walking_prediction.gif      # Generated GIF visualization
```

---

## 2. Environment Setup

This project uses the original TensorFlow 1.x style code. A Python 3.7 64-bit environment is recommended.

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install numpy scipy h5py matplotlib tensorflow==1.15 protobuf==3.20.*
```

For GIF saving, also install:

```bash
pip install pillow
```

---

## 3. Dataset

This project expects the Human3.6M dataset to be placed in the same format used by the original project. The link for Human3.6M URL exist no more, but available on WayBackMachine: [Human3.6M](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip)

A typical structure is:

```text
data/
  h3.6m/
    S1/
    S5/
    S6/
    S7/
    S8/
    S9/
    S11/
```

Place it under the main folder.

---

## 4. Main Training Script

The main script is:

```bash
python src/translate.py [options]
```

The same script is used for:

- training
- validation
- SRNN evaluation
- sampling predictions into `samples.h5`

---

## 5. Important Command-Line Parameters

### Model selection

Use the original GRU/RNN seq2seq model:

```bash
--model_type rnn
```

Use the added Transformer model:

```bash
--model_type transformer
```

---

### Action selection

Train and evaluate only on the walking action:

```bash
--action walking
```

Other actions:
```
["directions", "discussion", "eating", "greeting", "phoning",
 "posing", "purchases", "sitting", "sittingdown", "smoking",
 "takingphoto", "waiting", "walking", "walkingdog", "walkingtogether"]
```

Train and evaluate on all actions:

```bash
--action all
```

---

### Prediction length

Train for 25 future frames:

```bash
--seq_length_out 25
```

Train for 100 future frames:

```bash
--seq_length_out 100
```

Note: In the original project logic, sample mode may force 100 output frames. To make training and sampling consistent for 100-frame GIFs, train with:

```bash
--seq_length_out 100
```

---

### Model size

For RNN, `--size` is the hidden state size.

For Transformer, `--size` is the hidden dimension / `d_model`.

Examples:

```bash
--size 128
--size 256
--size 512
```

---

### Number of layers

For RNN, `--num_layers` controls the number of recurrent layers.

For Transformer, `--num_layers` controls the number of Transformer blocks.

Examples:

```bash
--num_layers 2
--num_layers 4
--num_layers 8
```

A Transformer configuration such as `256/4` means:

```text
--size 256 --num_layers 4
```

---

### Residual velocity

Use residual velocity modeling:

```bash
--residual_velocities
```

---

### Training iterations

Total number of training steps:

```bash
--iterations 10000
```

---

### Evaluation and saving frequency

Evaluate and save checkpoints every 1000 steps:

```bash
--test_every 1000
--save_every 1000
```

---

### Learning rate

Recommended default:

```bash
--learning_rate 0.0005
```

---

### CPU mode

Force the project to run on CPU:

```bash
--use_cpu
```

---

### Sampling mode

Load checkpoint at step 10000 and generate `samples.h5`:

```bash
--sample --load 10000
```

---

## 6. Training Examples

### 6.1 RNN without residual velocity, walking

```bash
python src/translate.py --model_type rnn --action walking --seq_length_out 100 --iterations 10000 --test_every 1000 --save_every 1000 --size 128 --num_layers 2 --learning_rate 0.0005 --use_cpu
```

---

### 6.2 RNN with residual velocity, walking

```bash
python src/translate.py --model_type rnn --action walking --seq_length_out 100 --iterations 10000 --test_every 1000 --save_every 1000 --size 128 --num_layers 2 --learning_rate 0.0005 --residual_velocities --use_cpu
```

---

### 6.3 Transformer without residual velocity, walking

```bash
python src/translate.py --model_type transformer --action walking --seq_length_out 100 --iterations 10000 --test_every 1000 --save_every 1000 --size 128 --num_layers 2 --learning_rate 0.0005 --use_cpu
```

---

### 6.4 Transformer with residual velocity, walking

```bash
python src/translate.py --model_type transformer --action walking --seq_length_out 100 --iterations 10000 --test_every 1000 --save_every 1000 --size 128 --num_layers 2 --learning_rate 0.0005 --residual_velocities --use_cpu
```

---

## 7. Sampling and GIF Generation

After training, generate predictions with `--sample`.

### Example: sample RNN with residual velocity

```bash
del samples.h5

[Original_training_command] --sample --load [Your_--iterations_number]

copy samples.h5 [The_name_you_want].h5
```

---

### Generate GIF

After `samples.h5` is created, run:

```bash
python src/forward_kinematics.py
```

Rename it to avoid overwriting:

```bash
rename walking_prediction.gif [The_name_you_want].gif
```

---

## 8. Baseline

Run the original baseline script:

```bash
python src/baselines.py
```

This includes simple baselines such as zero-velocity / running average prediction.

Zero-velocity baseline means all future frames are copied from the last observed pose. It is simple but strong for short-term prediction because human poses usually change slowly over small time intervals.

---

## 9. How to output loss curves

For each model, copy command line output from:

```text
step 0000; step_loss: 2.8822
step 0010; step_loss: 2.7015

...

step 9990; step_loss: 0.3517

milliseconds     |    80 |   160 |   320 |   400 |   560 |  1000 |
walking          | 0.367 | 0.636 | 0.951 | 1.087 | 1.290 | 1.725 |

============================
Global step:         10000
Learning rate:       0.0005
Step-time (ms):     32.3257
Train loss avg:      0.3489
--------------------------
Val loss:            0.5847
srnn loss:           0.5068
============================
```

Paste it to a txt file and rename the txt file to a name you need. Then, change file path input and output in plot_logs.py. Run:

```bash
python plot_logs.py
```

To get outputs.