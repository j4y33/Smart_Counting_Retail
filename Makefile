PYTHON = python3.6

# ========== Linux (Debian) ==========

.PHONY: install venv update clean remove


# ----- Install -----

install:
	$(if $(shell apt-cache search $(PYTHON)), , \
		sudo apt-get -q update \
		&& apt-get install --no-install-recommends -y apt-utils software-properties-common \
		&& add-apt-repository -y ppa:jonathonf/python-3.6 \
		&& apt-get -q update)
	sudo apt-get install --no-install-recommends -y \
		build-essential pkg-config \
		$(PYTHON) $(PYTHON)-dev $(PYTHON)-venv \
		ffmpeg


# ----- Virtualenv -----

venv:
	@if [ ! -f "venv/bin/activate" ]; then $(PYTHON) -m venv venv ; fi;


# ----- Update -----

update:
	@echo "----- Updating requirements -----"
	@pip install --upgrade wheel pip setuptools
	@pip install --upgrade --requirement requirements.txt

	# install detectron
	@pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.7/index.html

	# install torch-reid
	@mkdir -p third_party
	@if [ ! -d "third_party/deep-person-reid" ]; then git clone https://github.com/KaiyangZhou/deep-person-reid.git third_party/deep-person-reid; fi;
	@pip install --upgrade -r third_party/deep-person-reid/requirements.txt
	@cd third_party/deep-person-reid && python3 setup.py develop

	# pull weights
	@dvc pull assets/models/osnet_ain.pth
	@dvc pull assets/models/keypoint_rcnn_r_50_fpn_3x.pkl


update-dev: update
	@pip install --upgrade --requirement requirements-dev.txt



# ----- Tests -----

test:
	@echo "----- Tests Running -----"
	pytest tests -vv

test-verb:
	@echo "----- Tests Running Verbose-----"
	pytest tests -s -v

test-cov: update-dev
	@echo "----- Tests Coverage -----"
	pytest tests -v --flake8 --cov spyglass --cov-report html:.cov_html --cov-report term
	@python -m webbrowser -t "file://`pwd`/.cov_html/index.html" &


# ----- Clean -----

clean:
	-@find . \( \
		-name "__pycache__" -o \
		-name "*.pyc" -o \
		-name ".cache" -o \
		-name ".eggs" -o \
		-name "*.egg-info" \) \
		-prune \
		-exec rm -rf {} \;
	@rm -f .coverage
	@rm -rf .pytest_cache
	@rm -rf .cov_html
	@rm -rf build
	@rm -rf dist


# ----- Remove -----

remove: clean
	rm -rf venv