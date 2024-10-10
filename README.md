# Pacman RL Training Overview

## Visualization demo of Pacman agent playing

https://github.com/user-attachments/assets/5c644c69-022d-4b90-8fe2-f1b7630e5792

## Execution Time Across Processors
- Mac M1 CPU (8 cores): `2.5 minutes per episode`.
- Mac M1 GPU: `17 seconds per episode` (significantly faster). Ensure MPS is enabled.
- learning by memory replay at every 4th frame instead of every single frame made `7 seconds per episode`

## Chronology of Updates and Strategies at various episode checkpoints

### Episode 5 Checkpoint
- Starting point for fresh runs.

### Episode 430
- **avg score 477**

#### added prioritized rewards memory strategy logic
- `priority value` added to each sample reward in memory.

- High reward samples (`high_reward_threshold`) get a boost of `high_priority_boost` to their priority.

- current values are `high_reward_threshold = 10` and `high_priority_boost = 10`.

### Episode 500
- **avg score 411**
- No improvement with the prioritized reward strategy.

#### updates
- Increased `boost_priority` to 50

### Episode 600
- **avg score 382**
- No improvement with prioritized reward strategy `boost_priority = 50`

### updates
- removed the `priority reward` strategy and added the `bucketed memory` strategy.
- continue from Episode 430 with the New Bucketed Memory Strategy

#### reward buckets memory logic
- 5 buckets: [0, 10], [10, 50], [50, 100], [10, 400] [400, ∞]

- Each bucket has a fixed size of 2000 samples.

- Samples are drawn equally across buckets during training.

### Episode 500
- started again from the 430th episode checkpoint using a bucketed memory strategy.

- **avg score 468** vs 430th checkpoint avg score 477. same but at least not dropped as it did at 500th with prioritised strategy.

Once buckets are filled, we will be able to decide whether bucketed memory is working or not. Buckets take time to fill up, especially higher rewards buckets.

- currently, around `1 high reward comes in 7 episodes`. so, to get 50 samples in high reward buckets. it will require `50*7 = 350 episodes` around. also, as it learns it'll get more high rewards. so, let's say we average `1 high reward in 3 episodes` in future episodes, then we still need `50*3 = 150 episodes`. i.e starting from 430th, bucket should filled up at `430 + 150 = 580th ep`. so, will wait for up to 580th episodes and then see if it improves over the next 100 episodes or not.

### Episode 580th
- **avg score 427** dropped since 430th ep. but not as much as compared to the priority strategy
#### buckets info at 580th ep. The high reward bucket is not filled up much.
```bucketmem[50, 100] 60
bucketmem[100, 400] 40
bucketmem[400, +inf] 8
```

### updates
1. added `t_step` to the agent to learn at every `4th step` instead of every step. it will reduce learning time and also will reduce the learning noise.
2. `soft_updates` to target_network with `interpolation factor 0.001`. it will make the target network learn slowly and will make learning stable.
3. increasing `sample_size/minibatch_size from 64 to 100`.

### Episode 680th
**avg score 533**. `highest so far`. improved from 580th ep. so, the changes at 580th ep seems to be working.

#### How do 580th ep three changes probably work?
1. `t_step` improved speed. `now avg 7sec/ep. earlier best 17sec/ep`.
2. `soft_updates` to `target_network` seems to be working. it's learning slowly but steadily and persist more stable learning.
3. increasing `sample_size/minibatch_size` from 64 to 100 seems to be working. now, more samples to learn from.
out of all, **soft_updates must be the most significant in terms of Pacman score improvement**. since, persisting stable learning is the key to success in RL.

### Episode 840th
- **avg score 537**. `highest so far`. not much improvement from the 680th ep. but at least not dropped.

#### buckets info at 840th ep
```
bucketmem[50, 100] 210
bucketmem[100, 400] 122
bucketmem[400, +inf] 49
```
### Episode 1120th
- **avg score 540**. `highest so far`. not much improvement from the 800th ep.

- it fluctuated from the 800th ep 537 score to the 980th ep 428 score then rose back to 540 scores at the 1120th ep. `once max 580 scores reached at 1160th ep`

### updates, will start it from the 840th ep since not been much improvement since then
- Penalty for deaths: Adding a `-100 penalty` for 3rd Pacman death since for 1st, 2nd death there is no way to detect it.
- Updated buckets: `[0, 10], [10, 50], [50, 100], [100, ∞], [-∞, 0]` to incorporate the death penalty in bucket memory.

- can't detect Pacman's death except for 3rd death in each episode. so, will add a penalty of -100 on each death. it will make Pacman to avoid death as much as possible. to maintain the design as it is of bucketed memory. will combine the highest 2 buckets in one. i.e. [100, 400] and [400, +inf] in one bucket. it will make the last bucket free to use for the death penalty. so, will have 4 buckets. [0, 10], [10, 50], [50, 100], [100, +inf] and [-inf, 0]. will start from the 840th ep.

### We tried by -100 penalty for each death. but it didn't work. so, tried again with a -500 penalty

```NOTE: now score x is equivalent to (x-500) of earlier scores. since -100 penalty added for each death.```

### Episode 1200th
it fluctuates from 0 to 100 which is equivalent to 500 to 600 since we negative bucket has a -500 added reward for each episode.

### thoughts on the next approach to improving
it's possible that conv layers are not learning well. can debug by creating heatMap or other observability tools or can use pre-trained conv layer and check with that once how it performs.


## Visualisation of pacman playing
[120th ep](./src/ep%20120.mp4)

[200th ep](./src/ep%20200.mp4)

[400th ep](./src/ep%20400.mp4)

[600th ep](./src/ep%20600.mp4)

[800th ep](./src/ep%20800.mp4)

[1000th ep](./src/ep%201000%20bktmem%20pos%20reward%20only.mp4)

[1200th ep](./src/ep%201200%20nbktmem.mp4)


#### abbreviations
1. `ep` - episode
2. `chkpt` - checkpoint
3. `mem` - memory
4. `bktmem` - bucketed memory
5. `prtymem` - prioritised memory
