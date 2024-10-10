


## execution time in different processors
### in mac M1 cpu
it has 8 cores. it took `2.5 minutes to run 1 episode`.

### in mac M1 gpu
it took `17 seconds to run 1 episode`. singnificant difference. to use it make sure mps is enabled.

## changes introduced at checkpoint timelines
### oldest checkpoint
episode 5 checkpoints are beginning two checkpoints. if we want to start from the beginning we can use these checkpoints.

### 430 episode
**avg score 477**
after episode 430 checkpoint, prioritised reward strategy implemented and used.
if prioritised strategy won't work out well. we can use this checkpoint to continue from where it left.
#### prioritised strategy logic
we added `priority value` to each sample. it's an addons to memory. `memory` strategy as it is though
if `reward > 10`, we consider it high reward sample and boost its priority by 10.
for other samples, `priority` simply equals to `reward`.
`high_reward_threshold = 10` at the moment. it decides which are high reward samples.
`boost_priority = 10` at the moment. it decides how much we boost the priority of high reward samples.

### 500 episode
**avg score 411**. dropped since 430th chkpt
resumed from 430 checkpoint.
didn't see any improvement after checkpoint 430 changes. avg score even dropped a bit
so, post 500 episode, changed `boost_priority to 50`

### 600 episode
**avg score 382**. dropped since 500th chkpt
resumed from 500 checkpoint.
no improvement after 500 checkpoint changes.

### updates and start from 430 checkpoint
will remove the `priority value` strategy and add `reward buckets memory` strategy.
#### reward buckets memory logic
pacman has < 10 rewards/score category only. i.e. 0 on nothing, 10 on dots, 50 on ghost, 100 on power pellet, etc...
will create a `5=bucket_count` buckets of [0], [10], [50, 100], [200, 400] and [500, inf] rewards.
size of each bucket is `bucket_size = total_memory_size/bucket_count`. 10000/5 = 2000.
each bucket will have 2000 samples.
when we sample, we will sample from each bucket with equal probability. `bucket_sample_count = sample_size/bucket_count` samples from each bucket. i.e. 64/5 ~ 12 samples from each bucket.

will start from 430 checkpoint and see how it works. later will try bucket_size depends on reward value. higher reward will have large bucket_size and so more samples.
### 500 episode
again from 430 checkpoint by bucketed memory strategy.
**avg score 468**. vs 430th checkpoint avg score 477. same but atleast not dropped at it was at 500th with prioritised strategy.
once buckets filled up then will able to decide if bucketed memory working or not.
buckets take time to fill up. especially higer rewards buckets.
currently, around 1 higher rewards comes in 7 episodes. so, to get 50 samples in higher reward buckets. it will require 50*7 = 350 episodes around. also, as it learn it'll get more higher rewards. so, let's say average including future episodes. it will take 1 higher rewards 3 episodes. then we still need 50*3 = 150 episodes. i.e.e starting from 430th, bucket should filled up at 430+ 150 = 580th ep so, will wait for upto 580th episodes then will see if it improve over next 100 episodes or not.

### 580th episode
**avg score 427**. dropped since 430th ep. but not as much as compared to priority strategy
#### buckets info at 580th ep
```bucketmem[50, 100] 60
bucketmem[100, 400] 40
bucketmem[400, +inf] 8
```

### updates at 580th ep
1. added t_step to agent to learn at every 4th step instead of every step. it will reduce learning time and also will reduce the learning noise.
2. soft_updates to target_network with interpolation factor 0.001. it will make target network to learn slowly and will make learning stable.
3. increasing sample_size/minibatch_size from 64 to 100.

### 680th ep
**avg score 533**. `highest so far`. improved from 580th ep. so, the changes at 580th ep seems to be working.
#### how 580th ep three changes probably worked?
1. t_step improved speed. `now avg 7sec/ep. earlier best 17sec/ep`.
2. soft_updates to target_network seems to be working. it's learning slowly but steadily and persist more stable learning.
3. increasing sample_size/minibatch_size from 64 to 100 seems to be working. now, more samples to learn from.
out of all, soft_updates must be the most significant in terms of pacman score improvement. since, persisting stable learning which is the key to success in RL.

### 840th ep
**avg score 537**. `highest so far`. not much improvement from 680th ep. but atleast not dropped.
#### buckets info at 580th ep
```
bucketmem[50, 100] 210
bucketmem[100, 400] 122
bucketmem[400, +inf] 49
```
### 1120th ep
**avg score 540**. `highest so far`. not much improvement from 800th ep.
it fluctuated from 800th ep 537 score to 980th ep 4428 score then rise back to 540 score at 1120th ep. `once max 580 score reached at 1160th ep`

### updates, will start it from 840th ep since not much improvement since then
can't detect pacman death except 3rd death in each episode. so, will add a penalty of -100 on each death. it will make pacman to avoid death as much as possible. to maintain the design as it is of bucketed memory. will combine highest 2 buckets in one. i.e. [100, 400] and [400, +inf] in one bucket. it will make last bucket free to use for death penalty. so, will have 4 buckets. [0, 10], [10, 50], [50, 100], [100, +inf] and [-inf, 0]. will start from 840th ep.

### 1200th ep
it fluctutates 0 to 100 which is equivalent to 500 to 600 since we negative bucket has -500 added reward for each episode.

### thoughts on next approach to improve
posiible that conv layers are not learning well. can debug by creating heatMap or other observability tools or can use pretrained conv layer and check with that once how it performs.


## visualisation of pacman playing
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