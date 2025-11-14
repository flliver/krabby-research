.PHONY: train-teacher train-student play-teacher play-student help

help:
	@echo "Available targets:"
	@echo "  make train-teacher  - Train the teacher policy"
	@echo "  make train-student  - Train the student policy"
	@echo "  make train-student-resume - Resume training the student policy"
	@echo "  make play-teacher   - Play the teacher policy"
	@echo "  make play-student   - Play the student policy"


train-teacher:
	python scripts/rsl_rl/train.py --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-v0 --seed 1 --headless --num_envs 12288 $(ARGS)


train-teacher-resume:
	python scripts/rsl_rl/train.py --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-v0 --seed 1 --headless --num_envs 12288 --resume $(ARGS)

# --load_run 2025-11-13_21-26-55 --num_envs 6144
train-student:
	python scripts/rsl_rl/train.py --task Isaac-Extreme-Parkour-Student-Unitree-Go2-v0 --seed 1 --headless --num_envs 6144 $(ARGS)


train-student-resume:
	python scripts/rsl_rl/train.py --task Isaac-Extreme-Parkour-Student-Unitree-Go2-v0 --seed 1 --headless --resume $(ARGS)


play-teacher:
	python scripts/rsl_rl/play.py --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0 --num_envs 16 $(ARGS)


play-student:
	python scripts/rsl_rl/play.py --task Isaac-Extreme-Parkour-Student-Unitree-Go2-Play-v0 --num_envs 16 $(ARGS)
