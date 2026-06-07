```
REM ============================================================
REM Human Motion Prediction Project - Experiment Commands
REM Run all commands from project root:
REM human-motion-prediction-master/
REM ============================================================


REM ============================================================
REM 0. Activate virtual environment
REM ============================================================

.\.venv\Scripts\Activate.ps1


REM ============================================================
REM 1. Zero-velocity / running-average baseline
REM No training needed. This gives simple baseline errors.
REM ============================================================

python src/baselines.py


REM ============================================================
REM 2. RNN Seq2Seq baseline, 25-step prediction
REM Purpose: original non-residual RNN baseline
REM ============================================================

python src/translate.py --model_type rnn --action walking --seq_length_out 25 --iterations 10000 --test_every 1000 --save_every 1000 --size 128 --num_layers 2 --learning_rate 0.0005 --use_cpu

REM Generate samples.h5 from the selected checkpoint.
REM Replace --load 10000 with the best checkpoint if needed, e.g. 7000 or 9000.
del samples.h5

python src/translate.py --model_type rnn --action walking --seq_length_out 25 --iterations 10000 --test_every 1000 --save_every 1000 --size 128 --num_layers 2 --learning_rate 0.0005 --sample --load 10000 --use_cpu

copy samples.h5 samples_rnn_25.h5

REM Generate GIF. Make sure forward_kinematics.py saves to a unique GIF name.
python src/forward_kinematics.py

rename walking_prediction.gif walking_rnn_25.gif


REM ============================================================
REM 3. Residual RNN baseline, 25-step prediction
REM Purpose: reproduce original paper's residual-velocity idea
REM ============================================================

python src/translate.py --model_type rnn --action walking --seq_length_out 25 --iterations 10000 --test_every 1000 --save_every 1000 --size 128 --num_layers 2 --learning_rate 0.0005 --residual_velocities --use_cpu

del samples.h5

python src/translate.py --model_type rnn --action walking --seq_length_out 25 --iterations 10000 --test_every 1000 --save_every 1000 --size 128 --num_layers 2 --learning_rate 0.0005 --residual_velocities --sample --load 10000 --use_cpu

copy samples.h5 samples_residual_rnn_25.h5

python src/forward_kinematics.py

rename walking_prediction.gif walking_residual_rnn_25.gif


REM ============================================================
REM 4. Transformer without velocity loss, 25-step prediction
REM Purpose: test architecture change alone
REM Note: this assumes transformer_model.py uses normal pose MSE loss only.
REM ============================================================

python src/translate.py --model_type transformer --action walking --seq_length_out 25 --iterations 10000 --test_every 1000 --save_every 1000 --size 128 --num_layers 2 --learning_rate 0.0005 --residual_velocities --use_cpu

del samples.h5

python src/translate.py --model_type transformer --action walking --seq_length_out 25 --iterations 10000 --test_every 1000 --save_every 1000 --size 128 --num_layers 2 --learning_rate 0.0005 --residual_velocities --sample --load 10000 --use_cpu

copy samples.h5 samples_transformer_25.h5

python src/forward_kinematics.py

rename walking_prediction.gif walking_transformer_25.gif


REM ============================================================
REM 5. Transformer + velocity loss, 25-step prediction
REM Purpose: final proposed short-term model
REM Note: this assumes transformer_model.py loss has been changed to:
REM pose_loss + velocity_loss
REM ============================================================

python src/translate.py --model_type transformer --action walking --seq_length_out 25 --iterations 10000 --test_every 1000 --save_every 1000 --size 128 --num_layers 2 --learning_rate 0.0005 --residual_velocities --use_cpu

REM Choose the best checkpoint according to validation / SRNN / 80-400ms error.
REM Example below uses 9000. Change to 7000, 8000, or 10000 if better.
del samples.h5

python src/translate.py --model_type transformer --action walking --seq_length_out 25 --iterations 10000 --test_every 1000 --save_every 1000 --size 128 --num_layers 2 --learning_rate 0.0005 --residual_velocities --sample --load 9000 --use_cpu

copy samples.h5 samples_transformer_velocity_25.h5

python src/forward_kinematics.py

rename walking_prediction.gif walking_transformer_velocity_25.gif


REM ============================================================
REM 6. Transformer + velocity loss, 100-step prediction
REM Purpose: horizon ablation, compare 25-step vs 100-step
REM This studies long-term prediction stability and motion decay.
REM ============================================================

python src/translate.py --model_type transformer --action walking --seq_length_out 100 --iterations 10000 --test_every 1000 --save_every 1000 --size 128 --num_layers 2 --learning_rate 0.0005 --residual_velocities --use_cpu

REM Again, choose the best checkpoint based on validation / SRNN / error table.
REM Example below uses 9000.
del samples.h5

python src/translate.py --model_type transformer --action walking --seq_length_out 100 --iterations 10000 --test_every 1000 --save_every 1000 --size 128 --num_layers 2 --learning_rate 0.0005 --residual_velocities --sample --load 9000 --use_cpu

copy samples.h5 samples_transformer_velocity_100.h5

python src/forward_kinematics.py

rename walking_prediction.gif walking_transformer_velocity_100.gif


REM ============================================================
REM 7. Optional: Transformer + velocity loss on another action
REM Purpose: show the method is not only tested on walking
REM Suggested actions: eating or smoking
REM ============================================================

python src/translate.py --model_type transformer --action eating --seq_length_out 25 --iterations 10000 --test_every 1000 --save_every 1000 --size 128 --num_layers 2 --learning_rate 0.0005 --residual_velocities --use_cpu

del samples.h5

python src/translate.py --model_type transformer --action eating --seq_length_out 25 --iterations 10000 --test_every 1000 --save_every 1000 --size 128 --num_layers 2 --learning_rate 0.0005 --residual_velocities --sample --load 9000 --use_cpu

copy samples.h5 samples_transformer_velocity_eating_25.h5

python src/forward_kinematics.py

rename walking_prediction.gif eating_transformer_velocity_25.gif


REM ============================================================
REM 8. Optional: check motion amplitude from samples.h5
REM Use this after each sample generation.
REM This helps determine whether the model predicts real motion or near-static poses.
REM ============================================================

python check_samples.py


REM ============================================================
REM 9. Important notes
REM ============================================================

REM A. Every sample command must use the same parameters as its training command:
REM --model_type
REM --action
REM --seq_length_out
REM --iterations
REM --size
REM --num_layers
REM --learning_rate
REM --residual_velocities
REM --omit_one_hot, if used

REM B. Do not always use the final checkpoint automatically.
REM Check the printed validation/SRNN errors at 1000, 2000, ..., 10000.
REM Pick the checkpoint with the best validation/SRNN or 80-400ms error.

REM C. samples.h5 is overwritten every time you run --sample.
REM Always copy it immediately after generation.

REM D. walking_prediction.gif is overwritten every time you run forward_kinematics.py.
REM Always rename it immediately after generation.

REM E. For the final report, the most important experiments are:
REM 1) Zero-velocity baseline
REM 2) RNN 25-step
REM 3) Residual RNN 25-step
REM 4) Transformer + velocity loss 25-step
REM 5) Transformer + velocity loss 100-step
```