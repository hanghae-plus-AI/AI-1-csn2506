## 목표

이번 실습에서는 LoRA에서 rank를 변화시켰을 때, 성능 및 메모리 사용량 차이를 살펴볼 것입니다. 기존의 LoRA 실습 코드를 그대로 사용하되, 다음 부분들을 report 하시면 됩니다:

- [ ]  `lora_r`를 `[8, 128, 256]`로 변화시켜가며 학습
    - Deepspeed 없이 순수 LoRA만을 가지고 기존과 같은 LLM(`facebook/opt-350m`)과 dataset(`lucasmccabe-lmi/CodeAlpaca-20k`)를 활용합니다.
    - Rank를 8, 128, 256로 바꿔가며 학습을 진행해봅니다.
    - SFTTrainer는 다음과 같이 변경합니다:
        
        ```python
        trainer = SFTTrainer(
            model,
            train_dataset=dataset,
            args=SFTConfig(output_dir="/tmp/clm-instruction-tuning", **max_seq_length=128**),
            formatting_func=formatting_prompts_func,
            data_collator=collator,
        )
        trainer.train()
        ```
        
- [ ]  Rank에 따른 loss, 학습 속도, 그리고 메모리 점유율 공유
    - Loss는 wandb를 활용하여 다음과 같은 log를 공유합니다.
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/83c75a39-3aba-4ba4-a792-7aefe4b07895/b3073297-aa25-459d-bee9-0ed6843b447b/image.png)
        
    - 학습 속도 또한 wandb의 `Runtime` 항목을 공유합니다.
    - 메모리 점유율은 다음 코드를 적절히 추가하여 print한 후, 공유합니다.
        
        ```python
        print('Max Alloc:', round(torch.cuda.max_memory_allocated(0)/1024**3, 1), 'GB')
        ```
### 결과

https://wandb.ai/csn2506-diem/lora-rank-comparison/reports/train-loss-24-11-08-07-04-53---VmlldzoxMDA3NTY1MA

LoRA rank: 8로 학습 시작
Finishing last run (ID:h3r707di) before initializing another...
VBox(children=(Label(value='0.016 MB of 0.016 MB uploaded\r'), FloatProgress(value=1.0, max=1.0)))
View run rank_8 at: https://wandb.ai/csn2506-diem/lora-rank-comparison/runs/h3r707di
View project at: https://wandb.ai/csn2506-diem/lora-rank-comparison
Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
Find logs at: ./wandb/run-20241107_175009-h3r707di/logs
Successfully finished last run (ID:h3r707di). Initializing new run:
Tracking run with wandb version 0.18.3
Run data is saved locally in /kaggle/working/wandb/run-20241107_175147-wm0tl1v2
Syncing run rank_8 to Weights & Biases (docs)
View project at https://wandb.ai/csn2506-diem/lora-rank-comparison
View run at https://wandb.ai/csn2506-diem/lora-rank-comparison/runs/wm0tl1v2
/opt/conda/lib/python3.10/site-packages/trl/trainer/sft_trainer.py:463: UserWarning: You passed a dataset that is already processed (contains an `input_ids` field) together with a valid formatting function. Therefore `formatting_func` will be ignored.
  warnings.warn(
init Max Alloc: 5.3 GB
 [7509/7509 33:55, Epoch 3/3]
Step	Training Loss
500	2.935200
1000	2.440000
1500	2.334100
2000	2.303600
2500	2.262100
3000	2.253500
3500	2.216000
4000	2.210700
4500	2.198800
5000	2.178200
5500	2.183100
6000	2.163600
6500	2.168300
7000	2.175700
7500	2.163800
End Max Alloc: 5.3 GB
VBox(children=(Label(value='0.026 MB of 0.026 MB uploaded\r'), FloatProgress(value=1.0, max=1.0)))
Run history:

train/epoch	▁▁▂▂▃▃▄▄▅▅▆▆▇▇██
train/global_step	▁▁▂▂▃▃▄▄▅▅▆▆▇▇██
train/grad_norm	▃▁▁▂▂▂▂█▅▆▅▅▆▆▆
train/learning_rate	██▇▇▆▅▅▄▄▃▃▃▂▁▁
train/loss	█▄▃▂▂▂▁▁▁▁▁▁▁▁▁

Run summary:

total_flos	1.4030389685256192e+16
train/epoch	3
train/global_step	7509
train/grad_norm	2.58725
train/learning_rate	0.0
train/loss	2.1638
train_loss	2.27891
train_runtime	2036.2942
train_samples_per_second	29.498
train_steps_per_second	3.688

View run rank_8 at: https://wandb.ai/csn2506-diem/lora-rank-comparison/runs/wm0tl1v2
View project at: https://wandb.ai/csn2506-diem/lora-rank-comparison
Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
Find logs at: ./wandb/run-20241107_175147-wm0tl1v2/logs
wandb: Currently logged in as: csn2506 (csn2506-diem). Use `wandb login --relogin` to force relogin
LoRA rank: 128로 학습 시작
Tracking run with wandb version 0.18.3
Run data is saved locally in /kaggle/working/wandb/run-20241107_182551-uaeqv3q6
Syncing run rank_128 to Weights & Biases (docs)
View project at https://wandb.ai/csn2506-diem/lora-rank-comparison
View run at https://wandb.ai/csn2506-diem/lora-rank-comparison/runs/uaeqv3q6
/opt/conda/lib/python3.10/site-packages/trl/trainer/sft_trainer.py:463: UserWarning: You passed a dataset that is already processed (contains an `input_ids` field) together with a valid formatting function. Therefore `formatting_func` will be ignored.
  warnings.warn(
init Max Alloc: 5.3 GB
 [7509/7509 34:55, Epoch 3/3]
Step	Training Loss
500	2.917000
1000	2.430500
1500	2.327200
2000	2.298700
2500	2.258100
3000	2.249200
3500	2.212000
4000	2.206700
4500	2.194900
5000	2.174500
5500	2.179100
6000	2.160000
6500	2.165100
7000	2.172500
7500	2.160300
End Max Alloc: 5.3 GB
VBox(children=(Label(value='0.026 MB of 0.026 MB uploaded\r'), FloatProgress(value=1.0, max=1.0)))
Run history:

train/epoch	▁▁▂▂▃▃▄▄▅▅▆▆▇▇██
train/global_step	▁▁▂▂▃▃▄▄▅▅▆▆▇▇██
train/grad_norm	▃▁▁▂▂▂▁▇▄▅▄█▆▆▆
train/learning_rate	██▇▇▆▅▅▄▄▃▃▃▂▁▁
train/loss	█▄▃▂▂▂▁▁▁▁▁▁▁▁▁

Run summary:

total_flos	1.4574569423634432e+16
train/epoch	3
train/global_step	7509
train/grad_norm	0.65752
train/learning_rate	0.0
train/loss	2.1603
train_loss	2.27352
train_runtime	2096.2382
train_samples_per_second	28.654
train_steps_per_second	3.582

View run rank_128 at: https://wandb.ai/csn2506-diem/lora-rank-comparison/runs/uaeqv3q6
View project at: https://wandb.ai/csn2506-diem/lora-rank-comparison
Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
Find logs at: ./wandb/run-20241107_182551-uaeqv3q6/logs
LoRA rank: 256로 학습 시작
VBox(children=(Label(value='Waiting for wandb.init()...\r'), FloatProgress(value=0.011112508922208993, max=1.0…
Tracking run with wandb version 0.18.3
Run data is saved locally in /kaggle/working/wandb/run-20241107_190053-fsexlw48
Syncing run rank_256 to Weights & Biases (docs)
View project at https://wandb.ai/csn2506-diem/lora-rank-comparison
View run at https://wandb.ai/csn2506-diem/lora-rank-comparison/runs/fsexlw48
/opt/conda/lib/python3.10/site-packages/trl/trainer/sft_trainer.py:463: UserWarning: You passed a dataset that is already processed (contains an `input_ids` field) together with a valid formatting function. Therefore `formatting_func` will be ignored.
  warnings.warn(
init Max Alloc: 5.3 GB
 [7509/7509 36:18, Epoch 3/3]
Step	Training Loss
500	2.917400
1000	2.428900
1500	2.325900
2000	2.297300
2500	2.256300
3000	2.247300
3500	2.210000
4000	2.204800
4500	2.192700
5000	2.172300
5500	2.177000
6000	2.157900
6500	2.163000
7000	2.170300
7500	2.158000
End Max Alloc: 5.3 GB
VBox(children=(Label(value='0.026 MB of 0.026 MB uploaded\r'), FloatProgress(value=1.0, max=1.0)))
Run history:

train/epoch	▁▁▂▂▃▃▄▄▅▅▆▆▇▇██
train/global_step	▁▁▂▂▃▃▄▄▅▅▆▆▇▇██
train/grad_norm	▃▁▁▂▂▂▁█▄▆▅▇▆▆▆
train/learning_rate	██▇▇▆▅▅▄▄▃▃▃▂▁▁
train/loss	█▃▃▂▂▂▁▁▁▁▁▁▁▁▁

Run summary:

total_flos	1.5155027811237888e+16
train/epoch	3
train/global_step	7509
train/grad_norm	0.46337
train/learning_rate	0.0
train/loss	2.158
train_loss	2.27174
train_runtime	2179.3394
train_samples_per_second	27.562
train_steps_per_second	3.446

View run rank_256 at: https://wandb.ai/csn2506-diem/lora-rank-comparison/runs/fsexlw48
View project at: https://wandb.ai/csn2506-diem/lora-rank-comparison
Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
Find logs at: ./wandb/run-20241107_190053-fsexlw48/logs

### 장단점 분석

loss는 세 랭크 값(rank_8, rank_128, rank_256)에서 손실의 변화가 거의 비슷한 것을 확인할 수 있음
메모리 사용량에 대한 측정이 정확하지 않아 변화를 확인하지 못하였음
랭크값이 낮을수록 학습속도가 낮았음

