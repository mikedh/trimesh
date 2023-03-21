# See: https://clarkgrubb.com/makefile-style-guide
MAKEFLAGS += --warn-undefined-variables
SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := all
.DELETE_ON_ERROR:
.SUFFIXES:

# get the git short hash and trimesh semver.
VERSION := $(shell python trimesh/version.py)
GIT_SHA := $(shell git rev-parse --short HEAD)

# for coverage reports
GIT_SHA_FULL := $(shell git rev-parse HEAD)
GIT_REPO := "mikedh/trimesh"

# the name of the docker images
NAME=trimesh/trimesh
REPO=docker.io
IMAGE=$(REPO)/$(NAME)

# the tags we'll be applying
TAG_LATEST=$(IMAGE):latest
TAG_GIT_SHA=$(IMAGE):$(GIT_SHA)
TAG_VERSION=$(IMAGE):$(VERSION)

# This will output the help for each task
# See: https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
.PHONY: help
help: ## Print usage help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
.DEFAULT_GOAL := help

# build the output stage image using buildkit
.PHONY: build
build: ## Build the docker images
	DOCKER_BUILDKIT=1 \
	docker build \
		--progress=plain \
		--target output \
		--tag $(TAG_LATEST) \
		--tag $(TAG_VERSION) \
		--tag $(TAG_GIT_SHA) \
		.

# build the tests stage of the image
.PHONY: tests
tests: ## Run unit tests inside docker images.
	DOCKER_BUILDKIT=1 \
	docker build \
		--target tests \
		--progress=plain \
		--build-arg "CODECOV_TOKEN=$(CODECOV_TOKEN)" \
		.

# build the docs inside our image and eject the contents
# into `./html` directory
.PHONY: docs
docs: ## Build trimesh's sphinx docs
	DOCKER_BUILDKIT=1 \
	docker build \
		--target docs \
		--progress=plain \
		--output html \
		.

.PHONY: bash
bash: build ## Start a bash terminal in the image.
	docker run -it $(TAG_LATEST) /bin/bash

.PHONY: publish-docker
publish-docker: build ## Publish Docker images.
	docker push $(TAG_LATEST)
	docker push $(TAG_VERSION)
	docker push $(TAG_GIT_SHA)
