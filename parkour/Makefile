.PHONY: train-teacher train-student play-teacher play-student help

help:
	@echo "Available targets:"
	@echo "  make train-teacher  - Train the teacher policy"
	@echo "  make train-teacher-resume - Resume training the teacher policy"
	@echo "  make train-student  - Train the student policy"
	@echo "  make train-student-resume - Resume training the student policy"
	@echo "  make play-teacher   - Play the teacher policy"
	@echo "  make play-student   - Play the student policy"
	@echo "  make eval-teacher   - Evaluate the teacher policy"
	@echo "  make eval-student   - Evaluate the student policy"


train-teacher:
	python scripts/rsl_rl/train.py --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-v0 --seed 1 --headless --num_envs 12288 $(ARGS)


train-teacher-resume:
	python scripts/rsl_rl/train.py --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-v0 --seed 1 --headless --num_envs 12288 --resume $(ARGS)

# make train-student ARGS="--num_envs 12288
train-student:
	python scripts/rsl_rl/train.py --task Isaac-Extreme-Parkour-Student-Unitree-Go2-v0 --seed 1 --headless $(ARGS)


# make train-student-resume ARGS="--load_run 2025-11-13_21-26-55 --num_envs 6144"
train-student-resume:
	python scripts/rsl_rl/train.py --task Isaac-Extreme-Parkour-Student-Unitree-Go2-v0 --seed 1 --headless --resume $(ARGS)


play-teacher:
	python scripts/rsl_rl/play.py --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0 --num_envs 16 $(ARGS)


play-student:
	python scripts/rsl_rl/play.py --task Isaac-Extreme-Parkour-Student-Unitree-Go2-Play-v0 --num_envs 16 $(ARGS)

# make eval-student ARGS="--checkpoint /home/brian/patina/krabby-research/parkour/logs/rsl_rl/unitree_go2_parkour_teacher/2025-11-13_21-26-55/model_7400.pt"
eval-teacher:
	python scripts/rsl_rl/evaluation.py --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Eval-v0 $(ARGS)


# make eval-student ARGS="--checkpoint /home/brian/patina/krabby-research/parkour/logs/rsl_rl/unitree_go2_parkour_student/2025-11-14_12-43-18/model_8500.pt"
eval-student:
	python scripts/rsl_rl/evaluation.py --task Isaac-Extreme-Parkour-Student-Unitree-Go2-Eval-v0 $(ARGS)


eval-house-01:
	PARKOUR_SCENE_USD=assets/scenes/house_model_01.usd python scripts/rsl_rl/evaluation.py --task Isaac-Extreme-Parkour-Student-Unitree-Go2-Eval-v0 $(ARGS)


eval-flat:
	PARKOUR_SCENE_USD=assets/scenes/flat.usd python scripts/rsl_rl/evaluation.py --task Isaac-Extreme-Parkour-Student-Unitree-Go2-Eval-v0 $(ARGS)


eval-student-headless:
	python scripts/rsl_rl/evaluation.py --task Isaac-Extreme-Parkour-Student-Unitree-Go2-Eval-v0 --headless $(ARGS)
